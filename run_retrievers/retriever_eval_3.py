import os
import math
import pandas as pd
import numpy as np
import faiss
import tqdm
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from run_retrievers.utils import evaluate_retriever_performance
#from run_retrievers.utils import BASE_ADDRESS
from constant.constants import FILE_CANDIDATE_POOL_ADDRESS


#Full code for BM25
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        corpus: list of list of tokens (e.g. [["climate","change"],["global","warming"]])
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.N
        self.df = defaultdict(int)
        self.doc_tf = []
        
        for doc in corpus:
            tf = Counter(doc)
            self.doc_tf.append(tf)
            for term in tf:
                self.df[term] += 1
        
        # Preprocess IDF
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query, index):
        score = 0.0
        doc_tf = self.doc_tf[index]
        dl = self.doc_len[index]
        for term in query:
            if term not in doc_tf:
                continue
            tf = doc_tf[term]
            idf = self.idf.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (numerator / denominator)
        return score

    def get_scores(self, query):
        return [self.score(query, idx) for idx in range(self.N)]


def dense_retrieve_local(df, model_name):
    model = SentenceTransformer(model_name)
    results = {}
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Retrieving with {model_name}"):
        qid = str(row["id"])
        query = row["query"]
        passages = [row[f"passage_{i}"] for i in range(1, 101) if pd.notna(row.get(f"passage_{i}"))]

        passage_embeddings = model.encode(passages, convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        index = faiss.IndexFlatIP(passage_embeddings.shape[1])
        index.add(passage_embeddings)
        scores, indices = index.search(query_embedding, 10)
        results[qid] = [passages[i] for i in indices[0]]
    return results


def retrieve_with_deepct(df):
    return dense_retrieve_local(df, "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco")

def retrieve_with_colbert_v2h(df):
    return dense_retrieve_local(df, "colbert-ir/colbertv2.0")

def retrieve_with_sbert(df):
    return dense_retrieve_local(df, "sentence-transformers/msmarco-distilbert-base-tas-b")

def retrieve_with_colbert_v2(df):
    return dense_retrieve_local(df, "colbert-ir/colbertv2.0")

def retrieve_with_splade(df):
    return dense_retrieve_local(df, "naver/splade-cocondenser-ensembledistil")


#Hybrid
def hybrid_retrieve_bge_bm25(df, alpha=0.3):
    """
    Dense retriever (bge-small-en) + BM25 
    alpha: BM25 weight (0-1)
    """
    model = SentenceTransformer("C:/Users/14821/.cache/huggingface/hub/models--BAAI--bge-small-en/snapshots/2275a7bdee235e9b4f01fa73aa60d3311983cfea")   # dense retriever
    results = {}

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Hybrid retrieve (bge_small+bm25)"):
        qid = str(row["id"])
        query = row["query"]
        passages = [row[f"passage_{i}"] for i in range(1, 101) if pd.notna(row.get(f"passage_{i}"))]

        #Dense embedding
        passage_embeddings = model.encode(passages, convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        index = faiss.IndexFlatIP(passage_embeddings.shape[1])
        index.add(passage_embeddings)
        dense_scores, _ = index.search(query_embedding, len(passages))
        dense_scores = dense_scores[0]

        #BM25
        tokenized_passages = [p.lower().split() for p in passages]
        bm25 = BM25(tokenized_passages)
        bm25_scores = np.array(bm25.get_scores(query.lower().split()))

        #归一化
        def minmax(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)
        bm25_norm = minmax(bm25_scores)
        dense_norm = minmax(dense_scores)

        # Combine
        final_score = alpha * bm25_norm + (1 - alpha) * dense_norm
        ranked_idx = np.argsort(final_score)[::-1][:10]

        results[qid] = [passages[i] for i in ranked_idx]

    return results


def run_and_eval_retrievers():
    df = pd.read_csv(FILE_CANDIDATE_POOL_ADDRESS)
    
    retrievers = [
        # ("deepct", retrieve_with_deepct, f"{BASE_ADDRESS}/deepct.csv"),
        # ("colbert_v2h", retrieve_with_colbert_v2h, f"{BASE_ADDRESS}/colbert_v2h.csv"),
        # ("sbert", retrieve_with_sbert, f"{BASE_ADDRESS}/sbert.csv"),
        # ("colbert_v2", retrieve_with_colbert_v2, f"{BASE_ADDRESS}/colbert_v2.csv"),
        # ("splade", retrieve_with_splade, f"{BASE_ADDRESS}/splade.csv"),

        ("hybrid_bge_bm25", hybrid_retrieve_bge_bm25, f"C:/Users/14821/PyCharmMiscProject/run_retrievers/retriever_eval/hybrid_bge_bm25.csv"),
    ]

    for name, fn, path in retrievers:
        print(f"\nRunning retriever: {name}")
        try:
            results = fn(df)
            os.makedirs(os.path.dirname(path), exist_ok=True)  
            evaluate_retriever_performance(results, path)
        except Exception as e:
            print(f"Retriever '{name}' failed with error:\n{e}")
            continue


if __name__ == "__main__":
    run_and_eval_retrievers()

