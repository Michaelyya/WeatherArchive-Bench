import chromadb
import os
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from constant.constants import (
    CHROMADB_CLIENT_ADDRESS,
    CHROMADB_COLLECTION_NAME,
    LANGUAGE_ENGLISH,
    OPENAI_EMBEDDING_MODEL,
)
import dotenv
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()

nltk.download("punkt")
nltk.download("stopwords")
stopwords = set(nltk.corpus.stopwords.words(LANGUAGE_ENGLISH))

# Load Chroma client
client = chromadb.PersistentClient(path=CHROMADB_CLIENT_ADDRESS)
collection = client.get_collection(name=CHROMADB_COLLECTION_NAME)

disaster_prompts = [
    # Natural disaster
    "natural disaster",
    "earthquake",
    "flood",
    "hurricane",
    "tornado",
    "storm",
    "tsunami",
    "landslide",
    "wildfire",
    "volcanic eruption",
    # Climate
    "extreme weather",
    "heavy rain",
    "snowstorm",
    "hail",
    "drought",
    "heat wave",
    "cold wave",
    "weather damage",
    # Geographic related
    "mountain area",
    "coastal region",
    "river basin",
    "urban flooding",
    "rural area",
    "forest region",
    # Sup
    "emergency response",
    "evacuation",
    "rescue operation",
    "government aid",
    "disaster relief",
    "support troops",
    "civil protection",
    "humanitarian assistance",
]

# TODO, use the same embedding model, this is different from the one in embedding_loaders
# Otherwise the similarity is not comparable
embedding_func = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name=OPENAI_EMBEDDING_MODEL
)

disaster_embeddings = embedding_func(disaster_prompts)


# calculate disaster similarity's vector cosine
# def cosine_similarity(a, b):
#     a, b = np.array(a), np.array(b)
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


# def max_disaster_similarity(doc_emb):
#     return max(cosine_similarity(doc_emb, d_emb) for d_emb in disaster_embeddings)


def max_disaster_similarity(doc_emb):
    doc_emb = np.array(doc_emb).reshape(1, -1)
    disaster_matrix = np.array(disaster_embeddings)
    similarities = cosine_similarity(doc_emb, disaster_matrix)
    return np.max(similarities)


def preprocess(text):  # remove 1.stopwords 2.non algebra and number 3.Capital letters
    return [
        word
        for word in word_tokenize(text.lower())
        if word.isalnum() and word not in stopwords
    ]


# Hybrid retrieve function
def hybrid_retrieve(
    query,
    bm25_model,
    chroma_collection,
    top_k=10,
    bm25_weight=0.3,
    disaster_threshold=0.0,
):
    # BM25 retrieval
    tokenized_query = preprocess(query)
    bm25_scores = bm25_model.get_scores(tokenized_query)

    # Semantic retrieval
    semantic_results = chroma_collection.query(
        query_texts=[query], n_results=200  # full similarity for alignment
    )
    semantic_docs = semantic_results["documents"][0]
    semantic_scores = semantic_results["distances"][0]  # cosine distances

    # Normalize and convert distance to similarity, because we need to uniform comparing methods(both scores are within [0,1])
    semantic_scores = [1 - s for s in semantic_scores]

    # Match by ID
    semantic_id_score_map = dict(zip(semantic_results["ids"][0], semantic_scores))

    # Get every chunk's vector
    all_doc_embeddings = chroma_collection.get(ids=ids, include=["embeddings"])[
        "embeddings"
    ]

    # Merge: weighted sum of BM25 and semantic
    hybrid_scores = []
    for i, doc_id in enumerate(ids):
        bm25_score = bm25_scores[i]
        sem_score = semantic_id_score_map.get(doc_id, 0)
        final_score = bm25_weight * bm25_score + (1 - bm25_weight) * sem_score

        # Calculate disaster similarity and eliminate the chunks that not related to prompt
        doc_emb = np.array(all_doc_embeddings[i])
        disaster_sim = max_disaster_similarity(doc_emb)

        if disaster_sim >= disaster_threshold:
            hybrid_scores.append(
                (final_score, doc_id, documents[i], metadatas[i], disaster_sim)
            )
        else:
            pass

    # Sort and return top_k
    hybrid_scores.sort(reverse=True)
    return hybrid_scores[:top_k]


# Load documents from Chroma and build BM25 index
results = collection.get()
documents, metadatas, ids = results["documents"], results["metadatas"], results["ids"]
bm25 = BM25Okapi([preprocess(doc) for doc in documents])

# --- Usage Example
query = "What infrastructure and political challenges are being addressed in Montreal and New Edinburgh as a result of the ice conditions and how are they being managed according to the passage?"
top_docs = hybrid_retrieve(query, bm25, collection, top_k=10, bm25_weight=0)
# evaluate_accuracy("C:/Users/14821/Desktop/RAG/QACandidate_Pool.csv")
for i, (score, doc_id, doc, meta, dissim) in enumerate(top_docs):
    print(
        f"\nRank {i+1} | Score: {score:.4f} | DisasterSim: {dissim:.4f} | ID: {doc_id} | Date: {meta['date']} "
    )
    print(f"Chunk Text:\n{doc}")
    print("=" * 80)
