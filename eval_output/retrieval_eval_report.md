# Retrieval Evaluation Report: support-kb-retrieval-eval

- **Dataset version:** 1.0.0
- **Tenant:** eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee
- **Queries:** 6
- **K values:** [3, 5, 10]
- **Methods run:** bm25, dense, hybrid, hybrid_rerank

## Comparison table

| Method | Precision@3 | Recall@3 | MRR | NDCG@3 |
|---|---:|---:|---:|---:|
| bm25 | 0.5556 | 0.7500 | 0.8333 | 0.8167 |
| dense | 0.6667 | 0.9167 | 1.0000 | 0.9862 |
| hybrid | 0.6667 | 0.9167 | 1.0000 | 0.9862 |
| hybrid_rerank | 0.6667 | 0.9167 | 1.0000 | 0.9862 |

## Per-method aggregates

### bm25 (6 queries)
  - Precision@3: 0.5556 | Recall@3: 0.7500 | NDCG@3: 0.8167
  - Precision@5: 0.3333 | Recall@5: 0.7500 | NDCG@5: 0.8167
  - Precision@10: 0.1667 | Recall@10: 0.7500 | NDCG@10: 0.8167
  - MRR: 0.8333

### dense (6 queries)
  - Precision@3: 0.6667 | Recall@3: 0.9167 | NDCG@3: 0.9862
  - Precision@5: 0.4333 | Recall@5: 1.0000 | NDCG@5: 0.9956
  - Precision@10: 0.2167 | Recall@10: 1.0000 | NDCG@10: 0.9956
  - MRR: 1.0000

### hybrid (6 queries)
  - Precision@3: 0.6667 | Recall@3: 0.9167 | NDCG@3: 0.9862
  - Precision@5: 0.4333 | Recall@5: 1.0000 | NDCG@5: 0.9947
  - Precision@10: 0.2167 | Recall@10: 1.0000 | NDCG@10: 0.9947
  - MRR: 1.0000

### hybrid_rerank (6 queries)
  - Precision@3: 0.6667 | Recall@3: 0.9167 | NDCG@3: 0.9862
  - Precision@5: 0.4333 | Recall@5: 1.0000 | NDCG@5: 0.9956
  - Precision@10: 0.2167 | Recall@10: 1.0000 | NDCG@10: 0.9956
  - MRR: 1.0000
