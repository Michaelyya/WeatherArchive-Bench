import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import tqdm
from run_retrievers.utils import evaluate_retriever_performance
from run_retrievers.utils import BASE_ADDRESS
from constant.constants import FILE_CANDIDATE_POOL_ADDRESS


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


def doct5_retrieve_local(df):
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-tas-b")
    results = {}
    for _, row in tqdm.tqdm(
        df.iterrows(), total=len(df), desc="Retrieving with doct5 (tas-b surrogate)"
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


def retrieve_with_ance(df):
    return dense_retrieve_local(df, "castorini/ance-msmarco-passage")


def retrieve_with_colbert(df):
    return dense_retrieve_local(df, "colbert-ir/colbertv1.9")


def retrieve_with_unicoil(df):
    return dense_retrieve_local(df, "castorini/unicoil-msmarco-passage")


def run_and_eval_retrievers():
    df = pd.read_csv(FILE_CANDIDATE_POOL_ADDRESS)

    retrievers = [
        ("doct5", doct5_retrieve_local, f"{BASE_ADDRESS}/doct5.csv"),
        # ("ance", retrieve_with_ance, f"{BASE_ADDRESS}/ance.csv"),
        # ("colbert", retrieve_with_colbert, f"{BASE_ADDRESS}/colbert.csv"),
        # ("unicoil", retrieve_with_unicoil, f"{BASE_ADDRESS}/unicoil.csv"),
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
