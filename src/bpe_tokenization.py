import os
import re
import pandas as pd
import sentencepiece as spm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^A-Za-z0-9.,!?;:'\"()\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data(train_path="train_sample_filtered.parquet", val_path="validation_sample_filtered.parquet"):
    train_full = os.path.join(BASE_DIR, train_path)
    val_full = os.path.join(BASE_DIR, val_path)

    print(f"Loading train: {train_full}")
    df_train = pd.read_parquet(train_full)
    print(f"Loading val: {val_full}")
    df_val = pd.read_parquet(val_full)

    df = pd.concat([df_train, df_val], ignore_index=True)
    print(f"Total samples combined: {len(df)} (train + val)")
    return df



def build_corpus(df, output_path="corpus.txt"):
    output_full_path = os.path.join(BASE_DIR, output_path)
    print("Building corpus file (train + val) ...")

    with open(output_full_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            article = clean_text(row.get("article", ""))
            summary = clean_text(row.get("highlights", ""))
            if article or summary:
                f.write(f"{article} {summary}\n")

    size_kb = os.path.getsize(output_full_path) / 1024
    print(f"Corpus saved to {output_full_path} ({size_kb:.2f} KB)")

def train_sentencepiece(
    input_file="corpus.txt",
    model_prefix="bpe_tokenizer",
    vocab_size=32000,
    model_type="bpe",
):
    input_full_path = os.path.join(BASE_DIR, input_file)
    model_prefix_full = os.path.join(BASE_DIR, model_prefix)

    print("Training SentencePiece tokenizer ...")

    spm.SentencePieceTrainer.train(
        input=input_full_path,
        model_prefix=model_prefix_full,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<sep>", "<cls>"],
        input_sentence_size=2000000,
        shuffle_input_sentence=True,
        normalization_rule_name="identity",
        train_extremely_large_corpus=True
    )

    print(f"Model trained: {model_prefix_full}.model / {model_prefix_full}.vocab")

def test_tokenizer(model_path="bpe_tokenizer.model", df=None):
    model_full_path = os.path.join(BASE_DIR, model_path)
    print("\nTesting tokenizer ...")

    sp = spm.SentencePieceProcessor(model_file=model_full_path)

    if df is not None and "article" in df.columns:
        sample = df["article"].dropna().iloc[0]
    else:
        sample = "LeBron James Jnr appears to be following in his dad's footsteps."

    ids = sp.encode(sample, out_type=int)
    tokens = sp.encode(sample, out_type=str)
    decoded = sp.decode(ids)

    print("Tokenizer Test")
    print("Original:", sample[:200], "...")
    print("Tokens:", tokens[:20])
    print("IDs:", ids[:20])
    print("Decoded:", decoded[:300], "...")
    print(f"Vocab size: {sp.get_piece_size()}")


def load_tokenizer(model_path="bpe_tokenizer.model"):
    model_full_path = os.path.join(BASE_DIR, model_path)
    if not os.path.exists(model_full_path):
        raise FileNotFoundError(f"Tokenizer model not found: {model_full_path}")
    tokenizer = spm.SentencePieceProcessor(model_file=model_full_path)
    print(f"Loaded tokenizer from {model_full_path}, vocab size = {tokenizer.get_piece_size()}")
    return tokenizer


def main():
    df = load_data("train_sample_filtered.parquet", "validation_sample_filtered.parquet")
    build_corpus(df, "corpus.txt")
    train_sentencepiece(
        input_file="corpus.txt",
        model_prefix="bpe_tokenizer",
        vocab_size=32000,
        model_type="bpe"
    )
    test_tokenizer("bpe_tokenizer.model", df)

if __name__ == "__main__":
    main()
