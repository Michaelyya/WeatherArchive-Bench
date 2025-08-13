#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Concatenate chunks from all CSVs under ./embedding_loaders/raw_csv/ into one CSV.

- Input CSV schema: Date,Text  (Date ignored)
- Output CSV: ./embedding_loaders/concatenated_chunks.csv with a single column: Text
- Chunking replicates the provided sliding-window approach with safe sentence tokenization.
"""

import os
import re
import glob
import argparse
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken

# ----------------------------
# NLTK resources
# ----------------------------
def _ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    # Some envs require punkt_tab as well; try to fetch if present
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

# ----------------------------
# Tokenizer (OpenAI cl100k_base)
# ----------------------------
OPENAI_TOKENIZER = "cl100k_base"
tokenizer = tiktoken.get_encoding(OPENAI_TOKENIZER)

def tokenize(text: str):
    return tokenizer.encode(text)

def detokenize(tokens):
    return tokenizer.decode(tokens)

def force_split_tokens(text: str, max_tokens: int = 254):
    tokens = tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(detokenize(chunk_tokens))
    return chunks

# ----------------------------
# Safe sentence tokenize (replicating the given logic)
# ----------------------------
LANGUAGE_ENGLISH = "english"

def safe_sent_tokenize(text: str):
    protected = {
        "No.": "No<dot>",
        "Mr.": "Mr<dot>",
        "Ms.": "Ms<dot>",
        "Dr.": "Dr<dot>",
        "Mine.": "Mr<dot>",  # kept as-is to mirror the original snippet
        "Ste.": "Ste<dot>",
    }
    for k, v in protected.items():
        text = text.replace(k, v)

    # Protect decimals and currency like $12.34
    text = re.sub(r"\$(\d+)\.(\d+)", r"$\1<dot>\2", text)
    text = re.sub(r"(?<=\d)\.(?=\d)", "<dot>", text)

    sentences = sent_tokenize(text, language=LANGUAGE_ENGLISH)
    return [s.replace("<dot>", ".") for s in sentences]

# ----------------------------
# Sliding window chunking (replicates the provided logic)
# ----------------------------
def chunk_text(text: str, max_tokens: int = 256, overlap: int = 100):
    sentences = safe_sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        token_len = len(tokenize(sentence))

        if token_len > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            forced_chunks = force_split_tokens(sentence, max_tokens)
            chunks.extend(forced_chunks)
            i += 1
            continue

        if current_len + token_len > max_tokens:
            if current_chunk:
                chunk_text_str = " ".join(current_chunk)
                chunks.append(chunk_text_str)

                if overlap > 0:
                    overlap_tokens = 0
                    new_chunk = []
                    for sent in reversed(current_chunk):
                        sent_tokens = len(tokenize(sent))
                        if overlap_tokens + sent_tokens > overlap:
                            break
                        new_chunk.insert(0, sent)
                        overlap_tokens += sent_tokens

                    if new_chunk == current_chunk:
                        i += 1
                        current_chunk = []
                        current_len = 0
                    else:
                        current_chunk = new_chunk
                        current_len = sum(len(tokenize(s)) for s in current_chunk)
                else:
                    current_chunk = []
                    current_len = 0
            else:
                # Fallback (shouldn't normally happen due to the >max_tokens branch)
                chunks.append(sentence)
                i += 1
                continue
        else:
            current_chunk.append(sentence)
            current_len += token_len
            i += 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ----------------------------
# Main routine
# ----------------------------
def process_all_csvs(raw_dir: str, output_csv: str, max_tokens: int, overlap: int, pattern: str):
    _ensure_nltk_resources()

    files = sorted(glob.glob(os.path.join(raw_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSV files found with pattern '{pattern}' under {raw_dir}")

    all_chunks = []

    print(f"Found {len(files)} file(s) under {raw_dir} matching '{pattern}'.")
    for idx, file_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Processing: {file_path}")
        try:
            df = pd.read_csv(file_path, on_bad_lines="skip")
        except TypeError:
            # pandas < 1.4 compatibility (on_bad_lines introduced in 1.3, behavior changed later)
            df = pd.read_csv(file_path)

        if "Text" not in df.columns:
            raise ValueError(f"{file_path} has no 'Text' column")

        for _, row in df.iterrows():
            text = str(row["Text"]) if pd.notna(row["Text"]) else ""
            if not text.strip():
                continue
            chunks = chunk_text(text, max_tokens=max_tokens, overlap=overlap)
            all_chunks.extend(chunks)

    print(f"Total chunks produced: {len(all_chunks)}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    out_df = pd.DataFrame({"Text": all_chunks})
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved concatenated chunks to: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Chunk and concatenate CSV texts.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="./embedding_loaders/raw_csv",
        help="Directory containing input CSV files (default: ./embedding_loaders/raw_csv)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for CSV files (default: *.csv)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./embedding_loaders/concatenated_chunks.csv",
        help="Output CSV path (default: ./embedding_loaders/concatenated_chunks.csv)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens per chunk (default: 256)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Token overlap between adjacent chunks (default: 100)",
    )
    args = parser.parse_args()

    process_all_csvs(
        raw_dir=args.raw_dir,
        output_csv=args.output_csv,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        pattern=args.pattern,
    )

if __name__ == "__main__":
    main()
