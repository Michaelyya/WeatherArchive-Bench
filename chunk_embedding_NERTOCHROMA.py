import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import nltk
nltk.download("punkt")
nltk.download("stopwords")
from nltk.tokenize import word_tokenize, sent_tokenize
import tiktoken
import numpy as np
import openai
import ast
from datasets import load_dataset

# Load Chroma client
client = chromadb.PersistentClient(path="weather_chroma_store")
collection = client.get_or_create_collection(name="weather_records")

# Initialize embedding function
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# Tokenizer for BM25
stopwords = set(nltk.corpus.stopwords.words("english"))
def preprocess(text):
    return [word for word in word_tokenize(text.lower()) if word.isalnum() and word not in stopwords]

# Tokenizer for embedding/chunking
tokenizer = tiktoken.get_encoding("cl100k_base")

def tokenize(text):
    return tokenizer.encode(text)

def detokenize(tokens):
    return tokenizer.decode(tokens)

def chunk_text(text, max_tokens=125, overlap=20):
    sentences = sent_tokenize(text, language="english")
    chunks = []
    current_chunk = []
    current_len = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        token_len = len(tokenize(sentence))

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
                    current_chunk = new_chunk
                    current_len = sum(len(tokenize(s)) for s in current_chunk)
                else:
                    current_chunk = []
                    current_len = 0
            else:
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

# === Load NER-formatted dataset ===
def parse_ner_file(path):
    docs = []
    current_tokens = []
    current_doc_id = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                if current_tokens:
                    docs.append({"doc_id": current_doc_id, "text": " ".join(current_tokens)})
                    current_tokens = []
                current_doc_id = line.split()[-1]
            elif line:
                word, *rest = line.split()
                current_tokens.append(word)
        if current_tokens:
            docs.append({"doc_id": current_doc_id, "text": " ".join(current_tokens)})
    return docs

# Load and insert NER chunks to Chroma
ner_docs = parse_ner_file("ner_dataset.txt")  # replace with your NER file path
current_id = collection.count()

for doc in ner_docs:
    doc_id = doc["doc_id"]
    text = doc["text"]
    chunks = chunk_text(text)

    metadata = {
        "source": "ner",
        "doc_id": doc_id
    }

    for chunk in chunks:
        current_id += 1
        collection.add(
            documents=[chunk],
            ids=[str(current_id)],
            metadatas=[metadata]
        )

# === Other datasets: e.g., CSV or Hugging Face ===
df = pd.read_csv("your_csv.csv")
subset1 = df.iloc[:100]
for i, row in subset1.iterrows():
    article = str(row["Article"])
    chunks = chunk_text(article)
    metadata = {"source": "csv", "date": str(row["Date"])}
    for chunk in chunks:
        current_id += 1
        collection.add(
            documents=[chunk],
            ids=[str(current_id)],
            metadatas=[metadata]
        )

# Hugging Face dataset
hf = load_dataset("NLP-RISE/guardian_climate_news_corpus")
df2 = hf["train"].to_pandas().iloc[:10]
for i, row in df2.iterrows():
    article = str(row["body"])
    try:
        tags = ast.literal_eval(row["tags"]) if isinstance(row["tags"], str) else []
    except:
        tags = []
    metadata = {
        "source": "huggingface",
        "label": "climate",
        "title": str(row["title"]),
        "tags": tags,
        "date": str(row["date"])
    }
    chunks = chunk_text(article)
    for chunk in chunks:
        current_id += 1
        collection.add(
            documents=[chunk],
            ids=[str(current_id)],
            metadatas=[metadata]
        )

print("✅ 插入完成，当前 Chroma 总数:", collection.count())
