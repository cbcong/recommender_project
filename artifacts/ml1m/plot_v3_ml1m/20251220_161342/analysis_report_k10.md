# ML-1M Rerank Report (K=10)

- Generated at: 2025-12-20 16:13:47

- Best Recall@10: **NGCF-Rerank** = 0.0661
- Best NDCG@10: **NGCF-Rerank** = 0.0337
- Best Coverage@10: **MultiVAE-Rerank** = 0.6638
- Best LongTailShare@10: **SVDPP-Rerank** = 0.9241
- Best Novelty@10: **DIN-Rerank** = 14.9998

## Metric Correlations (across models)

|                  |   Recall@10 |    NDCG@10 |   Coverage@10 |   LongTailShare@10 |   Novelty@10 |
|:-----------------|------------:|-----------:|--------------:|-------------------:|-------------:|
| Recall@10        |   1         |  0.999847  |     0.0548827 |          -0.940986 |    -0.941865 |
| NDCG@10          |   0.999847  |  1         |     0.0546008 |          -0.940576 |    -0.941213 |
| Coverage@10      |   0.0548827 |  0.0546008 |     1         |           0.280617 |     0.279718 |
| LongTailShare@10 |  -0.940986  | -0.940576  |     0.280617  |           1        |     0.998462 |
| Novelty@10       |  -0.941865  | -0.941213  |     0.279718  |           0.998462 |     1        |

## Tune Source

- Latest tune csv used: `D:\WorkSpace\pycharm\Python学习路线\recommender_project\artifacts\ml1m\tune_v3_rerank_ml1m\20251219_123408\results_v3_ml1m_rerank_tune.csv`
