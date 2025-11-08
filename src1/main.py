# main.py
import os
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘制图表

from src.transformer_model import TransformerModel
from src.dataset import CNNDailyMailDataset
from src.bpe_tokenization import load_tokenizer

# ======================================================
# ⚙️ 设备选择
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 获取当前文件所在目录（跨平台路径根目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ======================================================
# 🔧 Mask 工具函数
# ======================================================
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


# ======================================================
# 🏋️ 训练函数
# ======================================================
def train_epoch(model, dataloader, optimizer, criterion, tokenizer, clip=1.0):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        src, tgt_inp, tgt_out = batch["article"], batch["decoder_input"], batch["decoder_output"]
        src, tgt_inp, tgt_out = src.to(DEVICE), tgt_inp.to(DEVICE), tgt_out.to(DEVICE)

        tgt_mask = generate_square_subsequent_mask(tgt_inp.size(1)).to(DEVICE)

        optimizer.zero_grad()
        output = model(src, tgt_inp, tgt_mask=tgt_mask)

        output_dim = output.shape[-1]
        loss = criterion(output.view(-1, output_dim), tgt_out.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ======================================================
# 🧪 验证函数
# ======================================================
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src, tgt_inp, tgt_out = batch["article"], batch["decoder_input"], batch["decoder_output"]
            src, tgt_inp, tgt_out = src.to(DEVICE), tgt_inp.to(DEVICE), tgt_out.to(DEVICE)
            tgt_mask = generate_square_subsequent_mask(tgt_inp.size(1)).to(DEVICE)

            output = model(src, tgt_inp, tgt_mask=tgt_mask)
            output_dim = output.shape[-1]
            loss = criterion(output.view(-1, output_dim), tgt_out.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


# ======================================================
# 🧠 推理（生成摘要）
# ======================================================
def generate_summary(model, tokenizer, src_text, max_len=80, top_k=50, top_p=0.9, temperature=1.0):
    """
    生成文本摘要，使用 Top-k Sampling 或 Nucleus Sampling
    """
    model.eval()
    src_ids = torch.tensor([tokenizer.encode(src_text, out_type=int)], dtype=torch.long).to(DEVICE)
    tgt_ids = torch.tensor([[tokenizer.bos_id()]], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        for _ in range(max_len):
            tgt_mask = generate_square_subsequent_mask(tgt_ids.size(1)).to(DEVICE)
            output = model(src_ids, tgt_ids, tgt_mask=tgt_mask)

            # 获取下一个 token 的 logits
            next_token_logits = output[:, -1, :] / temperature  # 使用温度控制分布

            # 应用 Top-k Sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                top_k_probs = F.softmax(top_k_logits, dim=-1)
                next_token = torch.multinomial(top_k_probs, 1).squeeze(0)
            # 应用 Nucleus Sampling (Top-p Sampling)
            else:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_logits[sorted_indices_to_remove] = -float("Inf")
                next_token = torch.argmax(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 将采样的 token 追加到 tgt_ids
            tgt_ids = torch.cat([tgt_ids, next_token.unsqueeze(0)], dim=1)

            # 如果生成的 token 是 EOS，则提前结束
            if next_token.item() == tokenizer.eos_id():
                break

    # 解码生成的 token ID 为文本
    decoded = tokenizer.decode(tgt_ids[0].tolist())
    return decoded


# ======================================================
# 🚀 主程序入口
# ======================================================
def main():
    print(f"🚀 Using device: {DEVICE}")

    # ==== 加载 tokenizer ====
    tokenizer_path = os.path.join(BASE_DIR, "bpe_tokenizer.model")
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_piece_size()
    print(f"✅ Tokenizer loaded ({vocab_size} vocab size)")

    # ==== 加载数据集 ====
    train_data_path = os.path.join(BASE_DIR, "train_sample_filtered.parquet")
    val_data_path = os.path.join(BASE_DIR, "validation_sample_filtered.parquet")

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"❌ Missing training file: {train_data_path}")
    if not os.path.exists(val_data_path):
        raise FileNotFoundError(f"❌ Missing validation file: {val_data_path}")

    train_dataset = CNNDailyMailDataset(train_data_path, tokenizer_path)
    val_dataset = CNNDailyMailDataset(val_data_path, tokenizer_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=val_dataset.collate_fn)

    # ==== 初始化模型 ====
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=2,
        num_layers=2,
        d_ff=512
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    # ==== 训练循环 ====
    num_epochs = 50
    best_val_loss = float('inf')
    model_save_path = os.path.join(BASE_DIR, "best_transformer.pth")

    # 用于存储每个epoch的损失
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, tokenizer)
        val_loss = evaluate(model, val_loader, criterion)

        # 保存每个 epoch 的损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"📅 Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 Model saved at {model_save_path}")

    # ==== 绘制训练和验证损失图表 ====
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 保存图表
    chart_save_path = os.path.join(BASE_DIR, "loss_plot.png") 
    plt.savefig(chart_save_path)
    print(f"💾 Loss plot saved at {chart_save_path}")

    def save_losses_to_csv(train_losses, val_losses, file_path):
        # 打开文件以写入
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # 写入标题行
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

            # 写入每个 epoch 的损失
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
                writer.writerow([epoch, train_loss, val_loss])

        print(f"💾 Losses saved to {file_path}")

    # 在训练完成后保存损失
    losses_file_path = os.path.join(BASE_DIR, "losses.csv")  # 指定 CSV 文件的保存路径
    save_losses_to_csv(train_losses, val_losses, losses_file_path)

    # ==== 推理测试 ====
    test_article = "Police and FBI agents are investigating the discovery of an empty rocket launcher tube on the front lawn of a Jersey City, New Jersey, home, FBI spokesman Sean Quinn said. Niranjan Desai discovered the 20-year-old AT4 anti-tank rocket launcher tube, a one-time-use device, lying on her lawn Friday morning, police said.The launcher has been turned over to U.S. Army officials at the 754th Ordnance Company, an explosive ordnance disposal unit, at Fort Monmouth, New Jersey, Army officials said.The launcher is no longer operable and not considered to be a hazard to public safety, police said, adding there was no indication the launcher had been fired recently.Army officials said they could not determine if the launcher had been fired, but indicated they should know once they find out where it came from. The nearest military base, Fort Dix, is more than 70 miles from Jersey City.The Joint Terrorism Task Force division of the FBI and Jersey City police are investigating the origin of the rocket launcher and the circumstance that led to its appearance on residential property.Al Qaeda doesn't leave a rocket launcher on the lawn of middle-aged ladies, said Paul Cruickshank of New York University Law School's Center on Law and Security.A neighbor, Joe Quinn, said the object lying on Desai's lawn looked military, was brown, had a handle and strap, and both ends were open, like you could shoot something with it. Quinn also said the device had a picture of a soldier on it and was 3 to 4 feet long.An Army official said the device is basically a shoulder-fired, direct-fire weapon used against ground targets -- a modern-day bazooka -- and it is not wire-guided.According to the Web site Globalsecurity.org, a loaded M136 AT4 anti-tank weapon has a 40-inch-long fiberglass-wrapped tube and weighs just 4 pounds. Its 84 millimeter shaped-charge missile can penetrate 14 inches of armor from a maximum of 985 feet. It is used once and discarded."
    summary = generate_summary(model, tokenizer, test_article)
    print("\n📝 Generated Summary:")
    print(summary)


if __name__ == "__main__":
    main()
