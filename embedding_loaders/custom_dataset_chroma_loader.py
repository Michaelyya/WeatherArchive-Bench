import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["ONNX_NUM_THREADS"] = "4"
from nltk.tokenize import sent_tokenize
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import re

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
import nltk
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import glob

nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize
import tiktoken
from constant.constants import (
    FILE_1386_ADDRESS,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_TOKENIZER,
    CHROMADB_CLIENT_ADDRESS,
    CHROMADB_COLLECTION_NAME,
    LANGUAGE_ENGLISH,
)

# from datasets import load_dataset
embedding_fn = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name=OPENAI_EMBEDDING_MODEL
)

# Localize Chroma
client = chromadb.PersistentClient(path=CHROMADB_CLIENT_ADDRESS)
collection = client.get_or_create_collection(
    name=CHROMADB_COLLECTION_NAME, embedding_function=embedding_fn
)

# Initialize tokenizer (ChatGPT tokenizer)
tokenizer = tiktoken.get_encoding(OPENAI_TOKENIZER)


def tokenize(text):
    return tokenizer.encode(text)


def detokenize(tokens):
    return tokenizer.decode(tokens)


def force_split_tokens(text, max_tokens=254):
    tokens = tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(detokenize(chunk_tokens))
    return chunks


def safe_sent_tokenize(text):
    protected = {
        "No.": "No<dot>",
        "Mr.": "Mr<dot>",
        "Ms.": "Ms<dot>",
        "Dr.": "Dr<dot>",
        "Mine.": "Mr<dot>",
        "Ste.": "Ste<dot>",
    }
    for k, v in protected.items():
        text = text.replace(k, v)

    text = re.sub(r"\$(\d+)\.(\d+)", r"$\1<dot>\2", text)
    text = re.sub(r"(?<=\d)\.(?=\d)", "<dot>", text)

    sentences = sent_tokenize(text, language=LANGUAGE_ENGLISH)
    return [s.replace("<dot>", ".") for s in sentences]


# sliding window chunking
def chunk_text(text, max_tokens=256, overlap=100):
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

        # If adding the sentence exceeds the max token limit
        if current_len + token_len > max_tokens:
            if current_chunk:
                # Finalize current chunk
                chunk_text_str = " ".join(current_chunk)

                chunks.append(chunk_text_str)

                # Create new chunk with overlap (from previous chunk's end)
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
                # Very long single sentence (skip or force chunk)
                chunks.append(sentence)
                i += 1
                continue
        else:

            current_chunk.append(sentence)
            current_len += token_len
            i += 1

    # Add remaining chunk
    if current_chunk:
        # print(current_chunk)
        chunks.append(" ".join(current_chunk))

    return chunks


print("Current collection count before adding noise:", collection.count())

# Process noise files and add them to the existing collection
noise_folder = r"./embedding_loaders/raw_csv"
files = glob.glob(os.path.join(noise_folder, "*_corrected.csv"))
noise_id_counter = collection.count()  # Start IDs after existing ones

for file_path in files:
    print(f"Processing file: {file_path}")
    df_noise = pd.read_csv(file_path, on_bad_lines="skip")

    if "Text" not in df_noise.columns:
        raise ValueError(f"{file_path} has no TEXT column")

    chunks_to_add = []
    ids_to_add = []
    metadatas_to_add = []

    for idx, row in df_noise.iterrows():
        text = str(row["Text"])
        chunks = chunk_text(text)
        for chunk in chunks:
            chunks_to_add.append(chunk)
            ids_to_add.append(f"noise_{noise_id_counter}")
            metadatas_to_add.append(
                {
                    "type": "noise",
                    "date": str(row.get("Date", "")),
                }
            )
            noise_id_counter += 1

    # Add in batches
    batch_size = 1000
    for i in range(0, len(chunks_to_add), batch_size):
        batch_end = min(i + batch_size, len(chunks_to_add))
        collection.add(
            documents=chunks_to_add[i:batch_end],
            ids=ids_to_add[i:batch_end],
            metadatas=metadatas_to_add[i:batch_end],
        )
        print(f"Added {batch_end} noise chunks so far")

print("Noise data added successfully")
print("Total chunks in collection:", collection.count())
