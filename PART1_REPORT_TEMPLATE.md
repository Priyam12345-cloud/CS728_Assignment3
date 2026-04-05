# Assignment 3 - Part 1 Report

## Setup

- Number of tools: 100
- Number of queries: 5000
- Top-k used: 5

## Methods Evaluated

- BM25 (sparse)
- msmarco-MiniLM (dense)
- UAE-large-v1 (dense)

## Results

| Method | Recall@1 | Recall@5 | Runtime (s) |
|---|---:|---:|---:|
| BM25 | 0.2426 | 0.4278 | 2.43 |
| msmarco-MiniLM | 0.3670 | 0.5958 | 42.39 |
| UAE-large-v1 | 0.6140 | 0.8698 | 887.54 |

## Short Analysis

1. UAE-large-v1 performs best at Recall@1 and Recall@5.
2. UAE-large-v1 also benefits the most in absolute terms when moving from @1 to @5, though BM25 and msmarco-MiniLM both improve substantially as well.

## Reproducibility

Run command used:

```powershell
python part1_retrieval.py --tools CS728_PA3/data/tools.json --queries CS728_PA3/data/test_queries.json --output-dir outputs_test --top-k 5
```

Generated files:

- outputs_test/metrics.csv
- outputs_test/summary.json
- outputs_test/rankings_bm25.jsonl
- outputs_test/rankings_msmarco_minilm.jsonl
- outputs_test/rankings_uae_large_v1.jsonl
