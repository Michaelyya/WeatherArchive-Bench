import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["ONNX_NUM_THREADS"] = "4"

from nltk.tokenize import sent_tokenize
import re

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
import nltk
nltk.download("punkt")

from nltk.tokenize import sent_tokenize
import tiktoken

#from datasets import load_dataset

# Localize Chroma
client = chromadb.PersistentClient(path="weather_chroma_store")



collection = client.get_or_create_collection(
    name="weather_records",
)

# Initialize tokenizer (ChatGPT tokenizer)
tokenizer = tiktoken.get_encoding("cl100k_base")

def tokenize(text):
    return tokenizer.encode(text)

def detokenize(tokens):
    return tokenizer.decode(tokens)

# sliding window chunking
def chunk_text(text, max_tokens=256, overlap=0):
    sentences = sent_tokenize(text, language="english")
    chunks = []
    current_chunk = []
    current_len = 0


    i = 0
    while i < len(sentences):

        sentence = sentences[i]
        token_len = len(tokenize(sentence))

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

        chunks.append(" ".join(current_chunk))

    return chunks



# Default embedding model
embedding_fn = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small"
)



# Process each article in 1386 datasets
df = pd.read_csv(r"C:/Users/14821/Desktop/RAG/MixedCTX_Dataset(1386).csv") #C:/Users/14821/Desktop/RAG/MixedCTX_Dataset(1386).csv
#r"/data/rech/mofengra/climateRAG/MixedCTX_Dataset(1386).csv"


current_id = collection.count()


for i, row in df.iterrows():
    article = str(row["Article"])
    chunks = chunk_text(article)
    metadata = {"date": str(row["Date"])}



    # Store each chunk into Chroma
    for j, chunk in enumerate(chunks):
        print(chunk)
        current_id += 1
        collection.add(
            documents=[chunk],
            ids=[str(current_id)],
            metadatas=[metadata]
        )
    print(f"{i + 1}")



print("Successfully write 1386 datasets into chroma. Total data size is:", collection.count())

'''
# Process each article in CORPUS datasets
dataset = load_dataset("NLP-RISE/guardian_climate_news_corpus")
df = dataset["train"].to_pandas()

# Optional take a small subset to test
subset = df.iloc[:2]
current_id = collection.count()

for i, row in subset.iterrows():
    article = str(row["body"])
    chunks = chunk_text(article)
    metadata = {"date": str(row["date"])}

    # Store each chunk into Chroma
    for j, chunk in enumerate(chunks):
        current_id += 1
        collection.add(
            documents=[chunk],
            ids=[str(current_id)],
            metadatas=[metadata]
        )

print("Successfully write CORPUS datasets into chroma. Total data size is:", collection.count())


# Load NER-formatted dataset
def parse_ner_conll_file(path):
    docs = []
    current_tokens = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                if current_tokens:
                    docs.append(" ".join(current_tokens))
                    current_tokens = []
            elif line:
                parts = line.split()
                if len(parts) >= 1:
                    word = parts[0]
                    current_tokens.append(word)
        if current_tokens:
            docs.append(" ".join(current_tokens))
    return docs

# Load and insert NER chunks to Chroma
ner_docs = parse_ner_conll_file("C:/Users/14821/Desktop/RAG/en.train.txt")[:2]   # replace with your NER file path
current_id = collection.count()

for text in ner_docs:
    chunks = chunk_text(text)

    metadata = {
        "date": "N/A"
    }

    for chunk in chunks:
        current_id += 1
        collection.add(
            documents=[chunk],
            ids=[str(current_id)],
            metadatas=[metadata]
        )

print("Successfully write NER datasets into chroma. Total data size is:", collection.count())

#[TSV files]Process each articles in climateTXT
df = pd.read_csv("C:/Users/14821/Desktop/RAG/AL-Wiki%20%28train%29.tsv", sep="\t")
current_id = collection.count()
subset = df.iloc[:2]

for i, row in subset.iterrows():
    text = str(row["sentence"])
    chunks = chunk_text(text)
    metadata = {
        "date": "N/A"
    }

    for chunk in chunks:
        current_id += 1
        collection.add(
            documents=[chunk],
            ids=[str(current_id)],
            metadatas=[metadata]
        )

print("Successfully write climateTXT_Wiki_Doc datasets into chroma. Total data size is:", collection.count())



df = pd.read_csv("C:/Users/14821/Desktop/RAG/AL-10Ks.tsv%20%3A%203000%20%2858%20positives%2C%202942%20negatives%29%20%28TSV%2C%20127138%20KB%29.tsv", sep="\t")
current_id = collection.count()
subset = df.iloc[:2]

for i, row in subset.iterrows():
    text = str(row["sentence"])
    chunks = chunk_text(text)
    metadata = {
        "date": "N/A"
    }

    for chunk in chunks:
        current_id += 1
        collection.add(
            documents=[chunk],
            ids=[str(current_id)],
            metadatas=[metadata]
        )

print("Successfully write climateTXT_10Ks datasets into chroma. Total data size is:", collection.count())

# Process Scientific abstract
df = pd.read_csv(r"C:/Users/14821/Desktop/RAG/SciDCC.csv")
subset = df.iloc[:2]
current_id = collection.count()

for i, row in subset.iterrows():
    article = str(row["Body"])
    chunks = chunk_text(article)
    metadata = {"date": str(row["Date"])}

    # Store each chunk into Chroma
    for j, chunk in enumerate(chunks):
        current_id += 1
        collection.add(
            documents=[chunk],
            ids=[str(current_id)],
            metadatas=[metadata]
        )

print("Successfully write Science Abstarct datasets into chroma. Total data size is:", collection.count())
'''