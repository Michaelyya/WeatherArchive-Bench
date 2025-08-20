import os
import numpy as np
import pandas as pd
import faiss
import tqdm
from openai import OpenAI
from run_retrievers.utils import evaluate_retriever_performance, BASE_ADDRESS
from constant.constants import (
    FILE_QUERY_ADDRESS,
    FILE_CONCATENATED_CHUNKS_ADDRESS,
    FILE_CORRECT_PASSAGES_ADDRESS,
)

import dotenv

dotenv.load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def encode_with_openai(texts, model="text-embedding-3-large", batch_size=64):
    """
    使用 OpenAI Embeddings API 编码文本
    """
    embeddings = []
    for i in tqdm.trange(0, len(texts), batch_size, desc=f"Encoding with {model}"):
        batch_texts = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch_texts)
        batch_embeddings = [d.embedding for d in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings


def dense_retrieve_openai(
    df_queries, passages_corpus, model_name="text-embedding-3-large", top_k=100
):
    """
    基于 OpenAI Embeddings 的本地稠密检索
    """
    if "id" not in df_queries.columns:
        df_queries = df_queries.copy()
        df_queries["id"] = df_queries.index.astype(str)
    else:
        df_queries = df_queries.copy()
        df_queries["id"] = df_queries["id"].astype(str)

    passages = [str(p) for p in passages_corpus if pd.notna(p)]
    if not passages:
        raise ValueError("passages_corpus 为空，请检查输入。")

    # 编码 passages
    passage_embeddings = encode_with_openai(passages, model=model_name)

    # 构建 FAISS Index
    dim = len(passage_embeddings[0])
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(passage_embeddings).astype("float32"))

    # 检索
    results = {}
    for _, row in tqdm.tqdm(
        df_queries.iterrows(),
        total=len(df_queries),
        desc=f"Retrieving with {model_name}",
    ):
        qid = str(row["id"])
        query = str(row["query"])

        query_emb = encode_with_openai([query], model=model_name)[0]
        scores, indices = index.search(
            np.array([query_emb]).astype("float32"), min(top_k, len(passages))
        )
        results[qid] = [passages[i] for i in indices[0]]

    return results


def run_and_eval_retrievers():
    # 读取数据
    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)
    df_chunks = pd.read_csv(FILE_CONCATENATED_CHUNKS_ADDRESS)
    df_golds = pd.read_csv(FILE_CORRECT_PASSAGES_ADDRESS)

    if "id" not in df_queries.columns:
        df_queries["id"] = df_queries.index.astype(str)
    else:
        df_queries["id"] = df_queries["id"].astype(str)

    if "correct_passage" in df_golds.columns and len(df_golds) == len(df_queries):
        df_queries["correct_passage"] = df_golds["correct_passage"]

    passages_corpus = df_chunks["Text"].astype(str).tolist()

    # 三个流行的 OpenAI Embedding 模型
    retrievers = [
        (
            "openai-embedding-3-large",
            lambda q, p: dense_retrieve_openai(
                q, p, model_name="text-embedding-3-large"
            ),
            f"{BASE_ADDRESS}/openai-embedding-3-large.csv",
        ),
        (
            "openai-embedding-3-small",
            lambda q, p: dense_retrieve_openai(
                q, p, model_name="text-embedding-3-small"
            ),
            f"{BASE_ADDRESS}/openai-embedding-3-small.csv",
        ),
        (
            "openai-embedding-ada-002",
            lambda q, p: dense_retrieve_openai(
                q, p, model_name="text-embedding-ada-002"
            ),
            f"{BASE_ADDRESS}/openai-embedding-ada-002.csv",
        ),
    ]

    for name, fn, path in retrievers:
        print(f"\nRunning retriever: {name}")
        try:
            results = fn(df_queries, passages_corpus)
            evaluate_retriever_performance(results, path)
        except Exception as e:
            print(f"Retriever '{name}' failed with error:\n{e}")


if __name__ == "__main__":
    run_and_eval_retrievers()
