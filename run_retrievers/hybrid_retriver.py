import chromadb
import sys, os
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

# （可选）启用重排
try:
    from rerank import ClimateReranker

    _HAS_RERANKER = True
except Exception:
    _HAS_RERANKER = False

dotenv.load_dotenv()

# 仅首次下载
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stopwords = set(nltk.corpus.stopwords.words(LANGUAGE_ENGLISH))

# ---- Chroma 语料加载（全库）----
client = chromadb.PersistentClient(path=CHROMADB_CLIENT_ADDRESS)
collection = client.get_collection(name=CHROMADB_COLLECTION_NAME)

# 一次性拉取全库内容并准备 BM25
_results = collection.get()
documents, metadatas, ids = (
    _results["documents"],
    _results["metadatas"],
    _results["ids"],
)
bm25 = BM25Okapi(
    [  # 供外部 import 使用
        [w for w in word_tokenize(doc.lower()) if w.isalnum() and w not in stopwords]
        for doc in documents
    ]
)

# ---- OpenAI Embedding 函数 ----
embedding_func = OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY", ""), model_name=OPENAI_EMBEDDING_MODEL
)

# ---- 灾害主题 prompt（可按需扩展/替换）----
disaster_prompts = [
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
    "extreme weather",
    "heavy rain",
    "snowstorm",
    "hail",
    "drought",
    "heat wave",
    "cold wave",
    "weather damage",
    "mountain area",
    "coastal region",
    "river basin",
    "urban flooding",
    "rural area",
    "forest region",
    "emergency response",
    "evacuation",
    "rescue operation",
    "government aid",
    "disaster relief",
    "support troops",
    "civil protection",
    "humanitarian assistance",
]
disaster_embeddings = embedding_func(disaster_prompts)


def max_disaster_similarity(doc_emb):
    doc_emb = np.array(doc_emb).reshape(1, -1)
    disaster_matrix = np.array(disaster_embeddings)
    sims = cosine_similarity(doc_emb, disaster_matrix)
    return float(np.max(sims))


def preprocess(text: str):
    return [
        w for w in word_tokenize(text.lower()) if w.isalnum() and w not in stopwords
    ]


def hybrid_retrieve(
    query: str,
    bm25_model: BM25Okapi,
    chroma_collection,
    top_k: int = 10,
    bm25_weight: float = 0.0,
    disaster_threshold: float = 0.0,
    rerank: bool = False,
):
    """
    在 Chroma 全库上进行混合检索：
    - bm25_weight=1.0 -> 纯 BM25
    - bm25_weight=0.0 -> 纯向量语义
    - 其余 -> 融合
    返回列表 [(final_score, doc_id, doc_text, meta, disaster_sim), ...]，按得分降序。
    """
    # BM25 打分（基于全量 documents 的顺序，与 ids 对齐）
    tokenized_query = preprocess(query)
    bm25_scores = bm25_model.get_scores(tokenized_query)  # len == len(ids)

    # 语义检索（在全库上获取尽可能多的候选）
    embedding_query = embedding_func(query)
    sem = chroma_collection.query(
        query_embeddings=embedding_query,
        n_results=min(5000, len(ids)),  # 较大候选，避免截断
    )
    # Chroma 返回的距离是 cosine distance（越小越近）
    semantic_scores = [1 - d for d in sem["distances"][0]]  # 转换为相似度
    semantic_id_score_map = dict(zip(sem["ids"][0], semantic_scores))

    # 为灾害过滤准备每条 embedding（一次 get 全量）
    all_doc_embeddings = chroma_collection.get(ids=ids, include=["embeddings"])[
        "embeddings"
    ]

    hybrid_scores = []
    for i, doc_id in enumerate(ids):
        bm25_score = float(bm25_scores[i])
        sem_score = float(semantic_id_score_map.get(doc_id, 0.0))
        final_score = bm25_weight * bm25_score + (1.0 - bm25_weight) * sem_score

        doc_emb = np.array(all_doc_embeddings[i])
        disaster_sim = max_disaster_similarity(doc_emb)

        if disaster_sim >= disaster_threshold:
            hybrid_scores.append(
                (final_score, doc_id, documents[i], metadatas[i], disaster_sim)
            )

    # 排序并截断
    hybrid_scores.sort(key=lambda x: x[0], reverse=True)
    hybrid_scores = hybrid_scores[:top_k]

    # 可选重排
    if rerank and _HAS_RERANKER and len(hybrid_scores) > 1:
        reranker = ClimateReranker()
        hybrid_scores = reranker.rerank(query, hybrid_scores, top_n=top_k)

    return hybrid_scores
