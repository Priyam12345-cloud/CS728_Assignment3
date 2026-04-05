import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


QUERY_COL_CANDIDATES = [
    "query",
    "query_text",
    "question",
    "utterance",
    "input",
    "text",
]

QUERY_ID_COL_CANDIDATES = [
    "query_id",
    "id",
    "qid",
]

GOLD_COL_CANDIDATES = [
    "gold_tool_name",
    "gold_tool_id",
    "relevant_tool_id",
    "target_tool_id",
    "tool_id",
    "label",
    "target",
    "answer",
]

TOOL_ID_COL_CANDIDATES = [
    "tool_id",
    "id",
    "name",
    "tool_name",
]

TOOL_NAME_COL_CANDIDATES = [
    "tool_name",
    "name",
    "title",
]

TOOL_DESC_COL_CANDIDATES = [
    "description",
    "tool_description",
    "desc",
    "documentation",
    "details",
    "content",
    "text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 1 retrieval pipeline for tool selection.")
    parser.add_argument("--tools", type=str, required=True, help="Path to tool file (csv/tsv/json/jsonl/parquet).")
    parser.add_argument("--queries", type=str, required=True, help="Path to query file (csv/tsv/json/jsonl/parquet).")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to write rankings and metrics.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k size used for rankings and Recall@k.")

    parser.add_argument("--query-col", type=str, default=None, help="Optional explicit query text column name.")
    parser.add_argument("--query-id-col", type=str, default=None, help="Optional explicit query id column name.")
    parser.add_argument("--gold-col", type=str, default=None, help="Optional explicit gold tool id column name.")

    parser.add_argument("--tool-id-col", type=str, default=None, help="Optional explicit tool id column name.")
    parser.add_argument("--tool-name-col", type=str, default=None, help="Optional explicit tool name column name.")
    parser.add_argument("--tool-desc-col", type=str, default=None, help="Optional explicit tool description column name.")

    parser.add_argument("--device", type=str, default=None, help="Torch device for dense models, e.g., cpu or cuda.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for dense embedding encoding.",
    )
    parser.add_argument(
        "--uae-query-prefix",
        type=str,
        default="",
        help="Optional prefix prepended to queries for UAE model.",
    )
    parser.add_argument(
        "--uae-tool-prefix",
        type=str,
        default="",
        help="Optional prefix prepended to tool text for UAE model.",
    )
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: c.strip() for c in df.columns}
    return df.rename(columns=renamed)


def pick_col(df: pd.DataFrame, explicit: Optional[str], candidates: Sequence[str], required: bool = True) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Column '{explicit}' not found. Available columns: {list(df.columns)}")
        return explicit

    lower_to_actual = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not infer required column from candidates {list(candidates)}. Available columns: {list(df.columns)}"
        )
    return None


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            # Support tool maps like {"tool_name": "tool description", ...}
            # while also handling generic dict payloads predictably.
            if all(not isinstance(v, (dict, list)) for v in payload.values()):
                df = pd.DataFrame(
                    {
                        "tool_name": list(payload.keys()),
                        "description": list(payload.values()),
                    }
                )
            else:
                df = pd.DataFrame(payload)
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            raise ValueError(f"Unsupported JSON structure in {path}: expected object or array.")
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Use one of: csv, tsv, txt, json, jsonl, parquet."
        )

    if df.empty:
        raise ValueError(f"Loaded empty table from {path}")

    return normalize_columns(df)


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text).lower())


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / denom


def cosine_scores(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    query_vecs = l2_normalize(query_vecs)
    doc_vecs = l2_normalize(doc_vecs)
    return query_vecs @ doc_vecs.T


def topk_indices_per_row(score_matrix: np.ndarray, k: int) -> np.ndarray:
    # Argsort once per row; easy to read and reliable for this dataset scale (~100 tools).
    return np.argsort(-score_matrix, axis=1)[:, :k]


def compute_recall_at_k(ranked_tool_ids: List[List[str]], gold_tool_ids: List[str], k: int) -> float:
    hits = 0
    for pred, gold in zip(ranked_tool_ids, gold_tool_ids):
        if gold in pred[:k]:
            hits += 1
    return hits / max(len(gold_tool_ids), 1)


def save_rankings(
    out_path: Path,
    query_ids: List[str],
    queries: List[str],
    gold_tool_ids: List[str],
    ranked_tool_ids: List[List[str]],
    method: str,
) -> None:
    rows = []
    for qid, query, gold, ranking in zip(query_ids, queries, gold_tool_ids, ranked_tool_ids):
        rows.append(
            {
                "method": method,
                "query_id": qid,
                "query": query,
                "gold_tool_id": gold,
                "ranked_tool_ids": ranking,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_tool_texts(tools_df: pd.DataFrame, name_col: Optional[str], desc_col: str) -> List[str]:
    if name_col is None:
        return tools_df[desc_col].fillna("").astype(str).tolist()

    merged = (
        tools_df[name_col].fillna("").astype(str).str.strip()
        + " "
        + tools_df[desc_col].fillna("").astype(str).str.strip()
    )
    return merged.str.strip().tolist()


def run_bm25(queries: List[str], tools_text: List[str], top_k: int) -> List[List[int]]:
    tokenized_tools = [simple_tokenize(x) for x in tools_text]
    bm25 = BM25Okapi(tokenized_tools)

    rankings = []
    for q in queries:
        q_tokens = simple_tokenize(q)
        scores = bm25.get_scores(q_tokens)
        top_idx = np.argsort(-scores)[:top_k]
        rankings.append(top_idx.tolist())
    return rankings


def run_dense(
    model_name: str,
    queries: List[str],
    tools_text: List[str],
    top_k: int,
    batch_size: int,
    device: Optional[str],
) -> List[List[int]]:
    model = SentenceTransformer(model_name, device=device)

    tool_emb = model.encode(
        tools_text,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    query_emb = model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    scores = cosine_scores(query_emb, tool_emb)
    top_idx = topk_indices_per_row(scores, top_k)
    return top_idx.tolist()


def as_string_list(col: Iterable) -> List[str]:
    return [str(x) for x in col]


def rankings_to_tool_ids(rankings: List[List[int]], tool_ids: List[str]) -> List[List[str]]:
    return [[tool_ids[i] for i in row] for row in rankings]


def main() -> None:
    args = parse_args()

    tools_path = Path(args.tools)
    queries_path = Path(args.queries)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tools_df = read_table(tools_path)
    queries_df = read_table(queries_path)

    tool_id_col = pick_col(tools_df, args.tool_id_col, TOOL_ID_COL_CANDIDATES, required=True)
    tool_desc_col = pick_col(tools_df, args.tool_desc_col, TOOL_DESC_COL_CANDIDATES, required=True)
    tool_name_col = pick_col(tools_df, args.tool_name_col, TOOL_NAME_COL_CANDIDATES, required=False)

    query_col = pick_col(queries_df, args.query_col, QUERY_COL_CANDIDATES, required=True)
    gold_col = pick_col(queries_df, args.gold_col, GOLD_COL_CANDIDATES, required=True)
    query_id_col = pick_col(queries_df, args.query_id_col, QUERY_ID_COL_CANDIDATES, required=False)

    if query_id_col is None:
        queries_df = queries_df.copy()
        queries_df["query_id"] = np.arange(len(queries_df)).astype(str)
        query_id_col = "query_id"

    tool_ids = as_string_list(tools_df[tool_id_col])
    tools_text = build_tool_texts(tools_df, tool_name_col, tool_desc_col)

    query_ids = as_string_list(queries_df[query_id_col])
    queries = as_string_list(queries_df[query_col])
    gold_tool_ids = as_string_list(queries_df[gold_col])

    if args.top_k < 5:
        raise ValueError("Use --top-k >= 5 to satisfy the Recall@5 deliverable.")

    methods = []

    # BM25
    t0 = time.time()
    bm25_rank_idx = run_bm25(queries, tools_text, args.top_k)
    bm25_ranked = rankings_to_tool_ids(bm25_rank_idx, tool_ids)
    bm25_r1 = compute_recall_at_k(bm25_ranked, gold_tool_ids, k=1)
    bm25_r5 = compute_recall_at_k(bm25_ranked, gold_tool_ids, k=5)
    bm25_time = time.time() - t0
    save_rankings(out_dir / "rankings_bm25.jsonl", query_ids, queries, gold_tool_ids, bm25_ranked, "bm25")
    methods.append({"method": "BM25", "recall@1": bm25_r1, "recall@5": bm25_r5, "seconds": bm25_time})

    # msmarco-MiniLM
    t0 = time.time()
    msmarco_rank_idx = run_dense(
        model_name="sentence-transformers/msmarco-MiniLM-L6-cos-v5",
        queries=queries,
        tools_text=tools_text,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=args.device,
    )
    msmarco_ranked = rankings_to_tool_ids(msmarco_rank_idx, tool_ids)
    msmarco_r1 = compute_recall_at_k(msmarco_ranked, gold_tool_ids, k=1)
    msmarco_r5 = compute_recall_at_k(msmarco_ranked, gold_tool_ids, k=5)
    msmarco_time = time.time() - t0
    save_rankings(
        out_dir / "rankings_msmarco_minilm.jsonl",
        query_ids,
        queries,
        gold_tool_ids,
        msmarco_ranked,
        "msmarco-MiniLM",
    )
    methods.append(
        {
            "method": "msmarco-MiniLM",
            "recall@1": msmarco_r1,
            "recall@5": msmarco_r5,
            "seconds": msmarco_time,
        }
    )

    # UAE-large-v1
    t0 = time.time()
    uae_queries = [f"{args.uae_query_prefix}{q}" for q in queries]
    uae_tools = [f"{args.uae_tool_prefix}{t}" for t in tools_text]
    uae_rank_idx = run_dense(
        model_name="WhereIsAI/UAE-Large-V1",
        queries=uae_queries,
        tools_text=uae_tools,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=args.device,
    )
    uae_ranked = rankings_to_tool_ids(uae_rank_idx, tool_ids)
    uae_r1 = compute_recall_at_k(uae_ranked, gold_tool_ids, k=1)
    uae_r5 = compute_recall_at_k(uae_ranked, gold_tool_ids, k=5)
    uae_time = time.time() - t0
    save_rankings(out_dir / "rankings_uae_large_v1.jsonl", query_ids, queries, gold_tool_ids, uae_ranked, "UAE-large-v1")
    methods.append({"method": "UAE-large-v1", "recall@1": uae_r1, "recall@5": uae_r5, "seconds": uae_time})

    metrics_df = pd.DataFrame(methods)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    summary = {
        "tools_file": str(tools_path),
        "queries_file": str(queries_path),
        "num_tools": len(tools_df),
        "num_queries": len(queries_df),
        "top_k": args.top_k,
        "columns": {
            "tool_id_col": tool_id_col,
            "tool_name_col": tool_name_col,
            "tool_desc_col": tool_desc_col,
            "query_id_col": query_id_col,
            "query_col": query_col,
            "gold_col": gold_col,
        },
        "metrics": methods,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print("=== Part 1 Retrieval Complete ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
