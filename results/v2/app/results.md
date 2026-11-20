### Scope: corpus (144 questions, app split, chunk size 450)

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR | Right paper in top 5 | Median latency (s) | p95 latency (s) |
|---|---|---|---|---|---|---|---|
| Dense | 14.6% | 26.4% | 34.7% | 0.214 | 59.0% | 0.018 | 0.025 |
| Hybrid | 16.7% | 29.9% | 36.1% | 0.240 | 63.2% | 0.019 | 0.025 |
| Hybrid + rerank | 20.8% | 38.2% | 48.6% | 0.304 | 68.8% | 1.759 | 2.337 |

### Scope: paper (144 questions, app split, chunk size 450)

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR | Median latency (s) | p95 latency (s) |
|---|---|---|---|---|---|---|
| Dense | 31.9% | 59.7% | 72.9% | 0.468 | 0.017 | 0.023 |
| Hybrid | 31.9% | 61.1% | 75.7% | 0.480 | 0.019 | 0.023 |
| Hybrid + rerank | 41.0% | 68.8% | 84.7% | 0.565 | 0.914 | 1.775 |

