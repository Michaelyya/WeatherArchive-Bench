BASE_ADDRESS = "run_retrievers/retriever_eval"
from constant.constants import FILE_CANDIDATE_POOL_ADDRESS
import pandas as pd


def evaluate_retriever_performance(
    retrieval_results: dict[str, list[str]], output_path: str
):
    df = pd.read_csv(FILE_CANDIDATE_POOL_ADDRESS)
    results = []

    for _, row in df.iterrows():
        qid = row["id"]
        query = row["query"]
        correct_idx = int(row["correct_passage_index"])
        golden_answer = row[f"passage_{correct_idx}"]

        retrieved_passages = retrieval_results.get(str(qid), [])

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
