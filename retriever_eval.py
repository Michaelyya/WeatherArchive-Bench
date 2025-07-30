import pandas as pd
from hybrid_retriver import hybrid_retrieve, bm25, collection
from constant.constants import FILE_CANDIDATE_POOL_ADDRESS
import math
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
import tqdm


def init_model(model_name_or_path: str, pooling: str = "mean"):
    return DRES(models.SentenceBERT(model_name_or_path, pooling=pooling))


def run_beir_retrieval(df, model, top_k=10):
    corpus = {}
    queries = {}
    for _, row in df.iterrows():
        qid = str(row["id"])
        queries[qid] = row["query"]
        for i in range(1, 101):  # passage_1 to passage_100
            passage = row.get(f"passage_{i}")
            if passage:
                corpus[f"{qid}_{i}"] = {"title": "", "text": passage}

    results = {}

    for qid, query in tqdm(queries.items(), desc="Retrieving"):
        scores = model.search(query=query, corpus=corpus, top_k=top_k)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results[qid] = [
            corpus[doc_id]["text"]
            for doc_id, _ in sorted_docs
            if doc_id.startswith(f"{qid}_")
        ][:top_k]

    return results


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
    print(f"✅ Results saved to {output_path}")


def retrieve_with_deepct(df):
    model = init_model("sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco")
    return run_beir_retrieval(df, model)


def retrieve_with_docT5(df):
    model = init_model("castorini/doc2query-t5-base-msmarco")
    return run_beir_retrieval(df, model)


def retrieve_with_colbert_h(df):
    model = init_model("colbert-ir/colbertv1.9-msmarco")
    return run_beir_retrieval(df, model)


def retrieve_with_colbert_v2h(df):
    model = init_model("colbert-ir/colbertv2.0")
    return run_beir_retrieval(df, model)


def retrieve_with_repbert(df):
    model = init_model("yjernite/repbert-base-uncased")
    return run_beir_retrieval(df, model)


def retrieve_with_ance(df):
    model = init_model("castorini/ance-msmarco")
    return run_beir_retrieval(df, model)


def retrieve_with_sbert(df):
    model = init_model("sentence-transformers/msmarco-distilbert-base-tas-b")
    return run_beir_retrieval(df, model)


def retrieve_with_colbert(df):
    model = init_model("colbert-ir/colbertv1.0")
    return run_beir_retrieval(df, model)


def retrieve_with_colbert_v2(df):
    model = init_model("colbert-ir/colbertv2.0")
    return run_beir_retrieval(df, model)


def retrieve_with_unicoil(df):
    model = init_model("castorini/unicoil-msmarco")
    return run_beir_retrieval(df, model)


def retrieve_with_splade(df):
    model = init_model("naver/splade-cocondenser-ensembledistil")
    return run_beir_retrieval(df, model)


def run_and_eval_retrievers():
    df = pd.read_csv(FILE_CANDIDATE_POOL_ADDRESS)

    bm25_results = {}
    semantic_results = {}

    for _, row in df.iterrows():
        qid = row["id"]
        query = row["query"]

        bm25_top = hybrid_retrieve(query, bm25, collection, top_k=10, bm25_weight=1.0)
        print("finished bm25 retrieval for query:", qid)
        semantic_top = hybrid_retrieve(
            query, bm25, collection, top_k=10, bm25_weight=0.0
        )
        print("finished semantic retrieval for query:", qid)

        bm25_results[qid] = [doc for _, _, doc, _, _ in bm25_top]
        semantic_results[qid] = [doc for _, _, doc, _, _ in semantic_top]

    evaluate_retriever_performance(bm25_results, "./retriever_eval/bm25.csv")
    evaluate_retriever_performance(semantic_results, "./retriever_eval/semantic.csv")

    # Sparse retrievers
    deepct_results = retrieve_with_deepct(df)
    evaluate_retriever_performance(deepct_results, "./retriever_eval/deepct.csv")

    docT5_results = retrieve_with_docT5(df)
    evaluate_retriever_performance(docT5_results, "./retriever_eval/doct5.csv")

    # Hybrid retrievers
    colbert_results = retrieve_with_colbert_h(df)
    evaluate_retriever_performance(colbert_results, "./retriever_eval/colbert_h.csv")

    colbertv2h_results = retrieve_with_colbert_v2h(df)
    evaluate_retriever_performance(
        colbertv2h_results, "./retriever_eval/colbert_v2h.csv"
    )

    # Dense retrievers
    repbert_results = retrieve_with_repbert(df)
    evaluate_retriever_performance(repbert_results, "./retriever_eval/repbert.csv")

    ance_results = retrieve_with_ance(df)
    evaluate_retriever_performance(ance_results, "./retriever_eval/ance.csv")

    sbert_results = retrieve_with_sbert(df)
    evaluate_retriever_performance(sbert_results, "./retriever_eval/sbert.csv")

    colbert_results = retrieve_with_colbert(df)
    evaluate_retriever_performance(colbert_results, "./retriever_eval/colbert.csv")

    colbertv2_results = retrieve_with_colbert_v2(df)
    evaluate_retriever_performance(colbertv2_results, "./retriever_eval/colbert_v2.csv")

    # Learnt Sparse
    unicoil_results = retrieve_with_unicoil(df)
    evaluate_retriever_performance(unicoil_results, "./retriever_eval/unicoil.csv")

    splade_results = retrieve_with_splade(df)
    evaluate_retriever_performance(splade_results, "./retriever_eval/splade.csv")


if __name__ == "__main__":
    run_and_eval_retrievers()
