# -*- coding: utf-8 -*-
import os
import time
import math
import json
import pandas as pd
import numpy as np
import faiss
import tqdm

# --- 你的工程中的常量 ---
from constant.constants import (
    FILE_QUERY_ADDRESS,  # e.g., "data/queries.csv"，需要列 'query'
    FILE_CONCATENATED_CHUNKS_ADDRESS,  # e.g., "data/concatenated_chunks.csv"，需要列 'Text'
)
from run_retrievers.utils import BASE_ADDRESS  # e.g., "run_retrievers/retriever_eval"

# --- Google Gemini SDK ---
import google.generativeai as genai

# ============ 工具函数（与原思路一致） ============


def ensure_id_column(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """
    确保 DataFrame 中存在唯一的 id 列；若不存在，用行号生成（字符串）。
    """
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = df.index.astype(str)
    else:
        df[id_col] = df[id_col].astype(str)
    return df


# ============ Gemini 嵌入封装 ============


class GeminiEmbedder:
    """
    统一封装 Google Gemini Embedding 调用。
    - 默认使用 'models/gemini-embedding-001'
    - 如果你的环境只支持新命名，可自动回退到 'models/text-embedding-004'
    - 提供批量 embed_texts，带简单重试与速率控制
    """

    def __init__(
        self,
        api_key_env: str = "GOOGLE_API_KEY",
        model_candidates=("models/gemini-embedding-001", "models/text-embedding-004"),
        requests_per_minute: int = 300,  # 简单节流：可按项目配额调整
        max_retries: int = 5,
        timeout_sec: float = 30.0,
    ):
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise EnvironmentError(
                f"未检测到环境变量 {api_key_env}。请先设置你的 Google AI Studio API Key 到该环境变量。"
            )
        genai.configure(api_key=api_key)

        self.model_name = None
        self._pick_first_available_model(model_candidates)
        self.rpm = max(1, requests_per_minute)
        self.sleep_between = 60.0 / self.rpm
        self.max_retries = max_retries
        self.timeout_sec = timeout_sec

    def _pick_first_available_model(self, candidates):
        # 直接使用首选；若调用失败会在 embed 时自动尝试下一个
        if isinstance(candidates, (list, tuple)) and len(candidates) > 0:
            self.model_name = candidates[0]
            self._fallbacks = list(candidates[1:])
        else:
            self.model_name = "models/gemini-embedding-001"
            self._fallbacks = ["models/text-embedding-004"]

    def _embed_once(self, model_name: str, text: str):
        """
        单条调用；返回 list[float]
        """
        # 注意：google-generativeai 的 embeddings API
        # 文本字段统一放在 input 或者 content 参数上（不同 SDK 版本略有差异）。
        # 这里优先使用 embed_content；部分版本也可用 embed_content(model=..., content=...).
        return genai.embed_content(
            model=model_name,
            content=text,
        )["embedding"]

    def embed_texts(self, texts, batch_size: int = 128, desc: str = "Encoding"):
        """
        批量嵌入，返回 shape = (N, dim) 的 np.ndarray
        - 自动使用当前模型名；如果失败，会对备用模型名重试
        """
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        all_vecs = []
        pbar = tqdm.tqdm(
            range(0, len(texts), batch_size),
            desc=desc,
            total=math.ceil(len(texts) / batch_size),
        )
        cur_model = self.model_name
        fallbacks = list(self._fallbacks)

        for start in pbar:
            end = min(start + batch_size, len(texts))
            batch = texts[start:end]

            # 调用 + 重试
            attempt = 0
            while True:
                try:
                    batch_vecs = []
                    for t in batch:
                        vec = self._embed_once(cur_model, str(t))
                        batch_vecs.append(vec)
                        time.sleep(self.sleep_between)  # 简单节流
                    vecs = np.array(batch_vecs, dtype="float32")
                    all_vecs.append(vecs)
                    break
                except Exception as e:
                    attempt += 1
                    # 如果还有备用模型，换一个再试
                    if ("model not found" in str(e).lower() or "404" in str(e)) and len(
                        fallbacks
                    ) > 0:
                        alt = fallbacks.pop(0)
                        print(
                            f"[GeminiEmbedder] 模型 {cur_model} 不可用，切换到 {alt} 重试…"
                        )
                        cur_model = alt
                        continue
                    if attempt >= self.max_retries:
                        raise RuntimeError(
                            f"Gemini embedding 失败（模型：{cur_model}){e}"
                        )
                    sleep_s = min(2**attempt, 8)  # 指数退避
                    print(
                        f"[GeminiEmbedder] 调用失败，{sleep_s}s 后重试（{attempt}/{self.max_retries}）：{e}"
                    )
                    time.sleep(sleep_s)

        emb = np.vstack(all_vecs)
        # 归一化（与原实现一致，用内积索引 ~ 余弦）
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
        return emb


# ============ 与原实现相同的检索与存储接口 ============


def dense_retrieve_local_ids(
    df_queries: pd.DataFrame,
    df_chunks: pd.DataFrame,
    embedder: GeminiEmbedder,
    top_k: int = 100,
):
    """
    使用 GeminiEmbedder + FAISS 做密集检索，返回每个 query 的 top_k 文档 **id 列表**。
    - df_queries: 需要列 ['id', 'query']
    - df_chunks: 需要列 ['id', 'Text']
    返回：{qid(str): [doc_id(str), ...]}
    """
    passages = df_chunks["Text"].astype(str).tolist()
    ids = df_chunks["id"].astype(str).tolist()
    if len(passages) == 0:
        raise ValueError("语料为空：未在 concatenated_chunks.csv 中找到可用的 'Text'。")

    print("[Gemini] Encoding passages...")
    passage_embeddings = embedder.embed_texts(
        passages, batch_size=128, desc="Encoding passages"
    )
    dim = passage_embeddings.shape[1]

    # 建立 FAISS 内积索引（向量已归一化 -> 内积 ≈ 余弦）
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embeddings)

    results = {}
    print("[Gemini] Retrieving ...")
    for _, row in tqdm.tqdm(
        df_queries.iterrows(),
        total=len(df_queries),
        desc="Retrieving with gemini-embedding-001",
    ):
        qid = str(row["id"])
        query = str(row["query"])
        q_emb = embedder.embed_texts([query], batch_size=1, desc="Encoding queries")
        k = min(top_k, len(passages))
        scores, indices = index.search(q_emb, k)
        top_idx = indices[0]
        results[qid] = [ids[i] for i in top_idx]

    return results


def save_raw_results_as_csv(
    results: dict,
    df_queries: pd.DataFrame,
    output_path: str,
    top_k: int = 100,
):
    """
    将 {qid: [doc_id, ...]} 存为 raw CSV：
    列：query, top_1, ..., top_k
    行对齐 df_queries 的顺序。
    """
    rows = []
    for _, row in df_queries.iterrows():
        qid = str(row["id"])
        query = str(row["query"])
        retrieved_ids = results.get(qid, [])
        retrieved_ids = (retrieved_ids + [""] * top_k)[:top_k]
        rows.append([query] + retrieved_ids)

    columns = ["query"] + [f"top_{i}" for i in range(1, top_k + 1)]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    print(f"[Saved] {output_path}")


def run_retrievers_and_dump_raw():
    # 读取 queries 与 chunks，并确保有 id 列
    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)  # 需要 'query'
    if "query" not in df_queries.columns:
        raise ValueError("queries.csv 需要包含列 'query'。")
    df_queries = ensure_id_column(df_queries, "id")

    df_chunks = pd.read_csv(FILE_CONCATENATED_CHUNKS_ADDRESS)  # 需要 'Text'
    if "Text" not in df_chunks.columns:
        raise ValueError("concatenated_chunks.csv 需要包含列 'Text'。")
    df_chunks = ensure_id_column(df_chunks, "id")

    # 初始化 Gemini 嵌入器（优先 gemini-embedding-001，失败则自动回退 text-embedding-004）
    embedder = GeminiEmbedder(
        api_key_env="GOOGLE_API_KEY",
        model_candidates=("models/gemini-embedding-001", "models/text-embedding-004"),
        requests_per_minute=300,
        max_retries=5,
    )

    # 仅一个“retriever”条目，名称沿用原脚本风格
    retrievers = [
        ("gemini-embedding-001", embedder),
    ]

    for short_name, emb in retrievers:
        print(f"\n=== Running retriever: {short_name} ===")
        try:
            results = dense_retrieve_local_ids(
                df_queries=df_queries,
                df_chunks=df_chunks,
                embedder=emb,
                top_k=100,
            )
            out_path = os.path.join(BASE_ADDRESS, f"raw_model_result_{short_name}.csv")
            save_raw_results_as_csv(results, df_queries, out_path, top_k=100)
        except Exception as e:
            print(f"[Skipped] retriever '{short_name}' failed: {e}")


if __name__ == "__main__":
    run_retrievers_and_dump_raw()
