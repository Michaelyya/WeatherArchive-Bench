import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import tqdm
from run_retrievers.utils import evaluate_retriever_performance
from run_retrievers.utils import BASE_ADDRESS
from constant.constants import FILE_CANDIDATE_POOL_ADDRESS


def dense_retrieve_local(df, model_name):
    """
    通用的本地稠密检索：保持原有逻辑不变
    """
    model = SentenceTransformer(model_name)
    results = {}
    for _, row in tqdm.tqdm(
        df.iterrows(), total=len(df), desc=f"Retrieving with {model_name}"
    ):
        qid = str(row["id"])
        query = row["query"]
        passages = [
            row[f"passage_{i}"]
            for i in range(1, 101)
            if pd.notna(row.get(f"passage_{i}"))
        ]
        if not passages:
            continue

        passage_embeddings = model.encode(
            passages, convert_to_numpy=True, normalize_embeddings=True
        )
        query_embedding = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )

        index = faiss.IndexFlatIP(passage_embeddings.shape[1])
        index.add(passage_embeddings)
        scores, indices = index.search(query_embedding, 10)
        results[qid] = [passages[i] for i in indices[0]]
    return results


def dense_retrieve_local_qwen(df, model_name, query_prompt_name="query"):
    """
    针对 Qwen-Embed 的本地稠密检索。
    若 sentence-transformers 版本支持，将在查询侧使用 prompt_name="query"；
    不支持则回退为普通 encode（与 dense_retrieve_local 等价）。
    """
    model = SentenceTransformer(model_name)
    results = {}
    for _, row in tqdm.tqdm(
        df.iterrows(), total=len(df), desc=f"Retrieving with {model_name}"
    ):
        qid = str(row["id"])
        query = row["query"]
        passages = [
            row[f"passage_{i}"]
            for i in range(1, 101)
            if pd.notna(row.get(f"passage_{i}"))
        ]
        if not passages:
            continue

        passage_embeddings = model.encode(
            passages, convert_to_numpy=True, normalize_embeddings=True
        )

        # 尝试使用 Qwen 的查询 prompt；不支持则回退
        try:
            query_embedding = model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                prompt_name=query_prompt_name,
            )
        except TypeError:
            query_embedding = model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            )

        index = faiss.IndexFlatIP(passage_embeddings.shape[1])
        index.add(passage_embeddings)
        scores, indices = index.search(query_embedding, 10)
        results[qid] = [passages[i] for i in indices[0]]
    return results


# -------- Qwen Embedding 模型封装（可按需增删）--------


def retrieve_with_qwen3_06b(df):
    # 轻量版
    return dense_retrieve_local_qwen(df, "Qwen/Qwen3-Embedding-0.6B")


def retrieve_with_qwen3_4b(df):
    # 效果/成本均衡
    return dense_retrieve_local_qwen(df, "Qwen/Qwen3-Embedding-4B")


def retrieve_with_qwen3_8b(df):
    # 最强但资源占用高
    return dense_retrieve_local_qwen(df, "Qwen/Qwen3-Embedding-8B")


def run_and_eval_retrievers():
    df = pd.read_csv(FILE_CANDIDATE_POOL_ADDRESS)

    retrievers = [
        ("qwen3-0_6b", retrieve_with_qwen3_06b, f"{BASE_ADDRESS}/qwen3-0_6b.csv"),
        ("qwen3-4b", retrieve_with_qwen3_4b, f"{BASE_ADDRESS}/qwen3-4b.csv"),
        ("qwen3-8b", retrieve_with_qwen3_8b, f"{BASE_ADDRESS}/qwen3-8b.csv"),
    ]

    for name, fn, path in retrievers:
        print(f"\nRunning retriever: {name}")
        try:
            results = fn(df)
            evaluate_retriever_performance(results, path)
        except Exception as e:
            print(f"Retriever '{name}' failed with error:\n{e}")
            continue


if __name__ == "__main__":
    run_and_eval_retrievers()
