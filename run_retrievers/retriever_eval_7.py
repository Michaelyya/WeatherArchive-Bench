# -*- coding: utf-8 -*-
import os
import time
import math
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# ============ 全局令牌桶限速器（线程安全） ============


class RateLimiter:
    """
    简单令牌桶：跨线程共享 RPM 配额，避免并发时触发服务端限流。
    rpm: 每分钟允许的请求数
    """

    def __init__(self, rpm: int):
        self.capacity = max(1, int(rpm))
        self.tokens = float(self.capacity)
        self.fill_rate = self.capacity / 60.0  # 每秒补充多少 token
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, tokens: int = 1):
        # 阻塞，直到拿到 tokens
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.timestamp
                if elapsed > 0:
                    self.tokens = min(
                        self.capacity, self.tokens + elapsed * self.fill_rate
                    )
                    self.timestamp = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                # 计算需要等待的时间（不持锁等待）
                needed = tokens - self.tokens
                wait_s = needed / self.fill_rate if self.fill_rate > 0 else 0.1
            time.sleep(wait_s)


# ============ Gemini 嵌入封装（支持多线程） ============


class GeminiEmbedder:
    """
    统一封装 Google Gemini Embedding 调用（多线程友好）。
    - 默认 'models/gemini-embedding-001'，不可用则回退 'models/text-embedding-004'
    - embed_texts 支持 num_threads>1 时并行，每次请求经过全局 RateLimiter
    """

    def __init__(
        self,
        api_key_env: str = "GOOGLE_API_KEY",
        model_candidates=("models/gemini-embedding-001", "models/text-embedding-004"),
        requests_per_minute: int = 300,  # 全局 RPM
        max_retries: int = 5,
    ):
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise EnvironmentError(
                f"未检测到环境变量 {api_key_env}。请先设置你的 Google AI Studio API Key 到该环境变量。"
            )
        genai.configure(api_key=api_key)

        self._candidates = list(model_candidates)
        self.model_name = None
        self._lock = threading.Lock()
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(requests_per_minute)

        # 预先选择可用模型，保证并发时维度一致
        self._ensure_working_model()

    # ---- 内部方法 ----

    def _ensure_working_model(self):
        last_err = None
        for m in self._candidates:
            try:
                _ = self._embed_once(m, "healthcheck")
                self.model_name = m
                return
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"没有可用的 Gemini 嵌入模型：{self._candidates}；最后错误：{last_err}"
        )

    def _embed_once(self, model_name: str, text: str):
        """
        单条调用；返回 list[float]
        """
        # 注意：google-generativeai 的 embeddings API 参数名在不同版本略有差异。
        # 这里采用 embed_content(model=..., content=...)
        resp = genai.embed_content(model=model_name, content=text)
        # SDK 返回一般为 {"embedding": [...]} 或 {"embedding": {"values": [...]}}，两种都兼容
        emb = resp.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            emb = emb["values"]
        return emb

    def _embed_with_retry(self, text: str) -> np.ndarray:
        attempt = 0
        while True:
            # 全局限速
            self.rate_limiter.acquire(1)
            try:
                vec = self._embed_once(self.model_name, str(text))
                return np.asarray(vec, dtype="float32")
            except Exception as e:
                attempt += 1
                # 如果是模型 404 之类，尝试一次回退（极少出现，因为已健康检查）
                if (
                    "model not found" in str(e).lower() or "404" in str(e).lower()
                ) and attempt == 1:
                    with self._lock:
                        # 再扫描候选找一个可用的
                        self._ensure_working_model()
                    continue
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(2**attempt, 8))  # 指数退避

    # ---- 对外方法 ----

    def embed_texts(self, texts, num_threads: int = 1, desc: str = "Encoding"):
        """
        批量嵌入，返回 (N, D) 的 float32 ndarray。支持多线程并发。
        - num_threads=1：顺序执行（兼容原来行为）
        - num_threads>1：使用线程池 + 全局 RateLimiter
        """
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        N = len(texts)
        # 单线程路径
        if num_threads <= 1:
            all_vecs = []
            for t in tqdm.tqdm(texts, desc=desc, total=N):
                all_vecs.append(self._embed_with_retry(t))
            emb = np.vstack(all_vecs)
        else:
            # 多线程路径：按索引收集结果，保证返回顺序与输入一致
            results = [None] * N
            with ThreadPoolExecutor(max_workers=num_threads) as ex:
                futures = {
                    ex.submit(self._embed_with_retry, texts[i]): i for i in range(N)
                }
                pbar = tqdm.tqdm(total=N, desc=f"{desc} (threads={num_threads})")
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as e:
                        # 这里直接抛出，外层会捕获；也可以选择写 None 并继续
                        raise RuntimeError(f"Embedding 失败（index={i}）：{e}")
                    finally:
                        pbar.update(1)
                pbar.close()
            emb = np.vstack(results)

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
    num_threads: int = 15,  # 新增：控制 passages 并发线程数
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

    print("[Gemini] Encoding passages (multi-threading)...")
    passage_embeddings = embedder.embed_texts(
        passages, num_threads=num_threads, desc="Encoding passages"  # <<<<<< 并发！
    )
    dim = passage_embeddings.shape[1]

    # 建立 FAISS 内积索引（向量已归一化 -> 内积 ≈ 余弦）
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embeddings)

    results = {}
    print("[Gemini] Retrieving ...")
    # 查询通常较少，就单线程即可；如果你的 query 也很多，可把 num_threads 也传进来
    for _, row in tqdm.tqdm(
        df_queries.iterrows(),
        total=len(df_queries),
        desc="Retrieving with gemini-embedding-001",
    ):
        qid = str(row["id"])
        query = str(row["query"])
        q_emb = embedder.embed_texts([query], num_threads=1, desc="Encoding queries")
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
        requests_per_minute=300,  # 这里是“全局 RPM”，会被 15 线程共享
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
                num_threads=15,  # <<<<<< 设置 15 个线程
            )
            out_path = os.path.join(BASE_ADDRESS, f"raw_model_result_{short_name}.csv")
            save_raw_results_as_csv(results, df_queries, out_path, top_k=100)
        except Exception as e:
            print(f"[Skipped] retriever '{short_name}' failed: {e}")


if __name__ == "__main__":
    run_retrievers_and_dump_raw()
