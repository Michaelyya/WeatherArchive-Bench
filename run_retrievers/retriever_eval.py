import pandas as pd

# from hybrid_retriver import hybrid_retrieve, bm25, collection

FILE_CANDIDATE_POOL_ADDRESS = "Ground-truth/QACandidate_Pool.csv"
import math
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tqdm


def evaluate_retriever_performance(
    retrieval_results: dict[str, list[str]], output_path: str
):
    df = pd.read_csv(f"../{FILE_CANDIDATE_POOL_ADDRESS}")
    results = []

    for _, row in df.iterrows():
        qid = row["id"]
        query = row["query"]
        correct_idx = int(row["correct_passage_index"])
        golden_answer = row[f"passage_{correct_idx}"]

        retrieved_passages = retrieval_results.get(qid, [])

        hit_k = {}  # hit@k flags
        mrr_k = {}  # mrr@k values
        hit_rank = -1  # 真实命中位置

        for k in [1, 5, 10]:
            hit_k[f"recall@{k}"] = 0
            mrr_k[f"mrr@{k}"] = 0.0

        for rank, p in enumerate(retrieved_passages[:10], 1):  # ranks start from 1
            if p.strip() == golden_answer.strip():
                hit_rank = rank
                for k in [1, 5, 10]:
                    if rank <= k:
                        hit_k[f"recall@{k}"] = 1
                        mrr_k[f"mrr@{k}"] = 1.0 / rank
                break  # stop after first match

        results.append(
            {
                "id": qid,
                "query": query,
                "hit_rank": hit_rank,
                **hit_k,
                **mrr_k,
            }
        )

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def dense_retrieve_local(df, model_name):
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


def retrieve_with_deepct(df):
    return dense_retrieve_local(
        df, "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    )


def retrieve_with_docT5(df):
    return dense_retrieve_local(df, "BeIR/query-gen-msmarco-t5-base-v1")


def retrieve_with_colbert_h(df):
    return dense_retrieve_local(df, "colbert-ir/colbertv1.9-msmarco")


def retrieve_with_colbert_v2h(df):
    return dense_retrieve_local(df, "colbert-ir/colbertv2.0")


def retrieve_with_repbert(df):
    return dense_retrieve_local(df, "yjernite/repbert-base-uncased")


def retrieve_with_ance(df):
    return dense_retrieve_local(df, "castorini/ance-msmarco")


def retrieve_with_sbert(df):
    return dense_retrieve_local(
        df, "sentence-transformers/msmarco-distilbert-base-tas-b"
    )


def retrieve_with_colbert(df):
    return dense_retrieve_local(df, "colbert-ir/colbertv1.0")


def retrieve_with_colbert_v2(df):
    return dense_retrieve_local(df, "colbert-ir/colbertv2.0")


def retrieve_with_unicoil(df):
    return dense_retrieve_local(df, "castorini/unicoil-msmarco")


def retrieve_with_splade(df):
    return dense_retrieve_local(df, "naver/splade-cocondenser-ensembledistil")


def run_and_eval_retrievers():
    df = pd.read_csv(f"../{FILE_CANDIDATE_POOL_ADDRESS}")

    retrievers = [
        ("deepct", retrieve_with_deepct, "./retriever_eval/deepct.csv"),
        ("doct5", retrieve_with_docT5, "./retriever_eval/doct5.csv"),
        ("colbert_h", retrieve_with_colbert_h, "./retriever_eval/colbert_h.csv"),
        ("colbert_v2h", retrieve_with_colbert_v2h, "./retriever_eval/colbert_v2h.csv"),
        ("repbert", retrieve_with_repbert, "./retriever_eval/repbert.csv"),
        ("ance", retrieve_with_ance, "./retriever_eval/ance.csv"),
        ("sbert", retrieve_with_sbert, "./retriever_eval/sbert.csv"),
        ("colbert", retrieve_with_colbert, "./retriever_eval/colbert.csv"),
        ("colbert_v2", retrieve_with_colbert_v2, "./retriever_eval/colbert_v2.csv"),
        ("unicoil", retrieve_with_unicoil, "./retriever_eval/unicoil.csv"),
        ("splade", retrieve_with_splade, "./retriever_eval/splade.csv"),
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
