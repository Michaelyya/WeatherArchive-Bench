import pandas as pd
from run_retrievers.hybrid_retriver import (
    hybrid_retrieve,
    bm25,
    collection,
)  # This is just a helper file
from run_retrievers.utils import evaluate_retriever_performance
from run_retrievers.utils import BASE_ADDRESS
from constant.constants import FILE_CANDIDATE_POOL_ADDRESS


def run_and_eval_retrievers():
    df = pd.read_csv(FILE_CANDIDATE_POOL_ADDRESS)

    bm25_results = {}
    semantic_results = {}

    for _, row in df.iterrows():
        qid = row["id"]
        query = row["query"]

        bm25_top = hybrid_retrieve(query, bm25, collection, top_k=10, bm25_weight=1.0)
        semantic_top = hybrid_retrieve(
            query, bm25, collection, top_k=10, bm25_weight=0.0
        )

        bm25_results[qid] = [doc for _, _, doc, _, _ in bm25_top]
        semantic_results[qid] = [doc for _, _, doc, _, _ in semantic_top]
        print(f"Processed query {qid}")

    evaluate_retriever_performance(bm25_results, f"{BASE_ADDRESS}/bm25.csv")
    evaluate_retriever_performance(semantic_results, f"{BASE_ADDRESS}/semantic.csv")


if __name__ == "__main__":
    run_and_eval_retrievers()
