# CS728 Assignment 3 - Part 1 (Classical Retrieval)

This implementation completes Part 1 deliverables:

- BM25 sparse retrieval baseline
- msmarco-MiniLM dense retrieval
- UAE-large-v1 dense retrieval
- Ranked tool lists for each query
- Recall@1 and Recall@5 for each method

## 1) Expected Inputs

Place two files in this workspace:

1. Tool catalog file (`csv`, `tsv`, `json`, `jsonl`, or `parquet`) with at least:
- Tool ID column (for example: `tool_id`)
- Tool description column (for example: `description`)
- Optional tool name column (for example: `tool_name`)

2. Query file (`csv`, `tsv`, `json`, `jsonl`, or `parquet`) with at least:
- Query text column (for example: `query`)
- Gold/correct tool ID column (for example: `gold_tool_id`)
- Optional query ID column (for example: `query_id`)

The script auto-detects common column names and also supports explicit overrides.

## 2) Install Dependencies

```powershell
c:/Users/Lenovo/Desktop/CS728/Assignment3/.venv/bin/python.exe -m pip install -r requirements.txt
```

## 3) Run Part 1 Pipeline

Example command:

```powershell
c:/Users/Lenovo/Desktop/CS728/Assignment3/.venv/bin/python.exe part1_retrieval.py \
  --tools tools.csv \
  --queries queries.csv \
  --output-dir outputs \
  --top-k 5
```

If your column names are different, pass them explicitly:

```powershell
c:/Users/Lenovo/Desktop/CS728/Assignment3/.venv/bin/python.exe part1_retrieval.py \
  --tools my_tools.jsonl \
  --queries my_queries.jsonl \
  --tool-id-col id \
  --tool-name-col name \
  --tool-desc-col description \
  --query-id-col qid \
  --query-col question \
  --gold-col target_tool_id \
  --output-dir outputs \
  --top-k 5
```

Optional UAE formatting knobs:

- `--uae-query-prefix ""`
- `--uae-tool-prefix ""`

## 4) Outputs

The script writes:

- `outputs/metrics.csv`: Recall@1, Recall@5, runtime per method
- `outputs/summary.json`: config + inferred columns + metrics
- `outputs/rankings_bm25.jsonl`
- `outputs/rankings_msmarco_minilm.jsonl`
- `outputs/rankings_uae_large_v1.jsonl`

Each rankings file contains one JSON line per query with:

- method
- query_id
- query
- gold_tool_id
- ranked_tool_ids

## 5) Deliverable Checklist (Part 1)

- [x] Encode query/tools independently
- [x] Similarity scoring for every query-tool pair
- [x] Top-k retrieval
- [x] Evaluate BM25, msmarco-MiniLM, UAE-large-v1
- [x] Report Recall@1 and Recall@5

## 6) Typical Runtime

For around 100 tools and a moderate number of queries:

- CPU only: usually around 8 to 25 minutes (depends mainly on model downloads and query count)
- After models are cached, reruns are significantly faster
