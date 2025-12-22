# ML-1M Rerank Report (K=10)

- Generated at: 2025-12-20 22:43:17

- Best Recall@10: **NGCF-Rerank** = 0.0702
- Best NDCG@10: **NGCF-Rerank** = 0.0356
- Best Coverage@10: **NGCF-Rerank** = 0.4814
- Best LongTailShare@10: **HybridNCF-Rerank** = 0.1198
- Best Novelty@10: **HybridNCF-Rerank** = 10.2087

## Metric Correlations (across models)

|                  |   Recall@10 |   NDCG@10 |   Coverage@10 |   LongTailShare@10 |   Novelty@10 |
|:-----------------|------------:|----------:|--------------:|-------------------:|-------------:|
| Recall@10        |    1        |  0.99993  |      0.925257 |          -0.90466  |    -0.854534 |
| NDCG@10          |    0.99993  |  1        |      0.92872  |          -0.901819 |    -0.850383 |
| Coverage@10      |    0.925257 |  0.92872  |      1        |          -0.691141 |    -0.599087 |
| LongTailShare@10 |   -0.90466  | -0.901819 |     -0.691141 |           1        |     0.988563 |
| Novelty@10       |   -0.854534 | -0.850383 |     -0.599087 |           0.988563 |     1        |

## Tune Source

- Latest tune csv used: `D:\WorkSpace\pycharm\Python学习路线\recommender_project\artifacts\ml1m\tune_v3_rerank_ml1m\20251219_123408\results_v3_ml1m_rerank_tune.csv`
