### Scope: corpus (109 questions, dev split, chunk size 450)

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR | Right paper in top 5 | Median latency (s) | p95 latency (s) |
|---|---|---|---|---|---|---|---|
| Dense | 13.8% | 27.5% | 33.0% | 0.207 | 44.0% | 0.019 | 0.025 |
| Hybrid | 14.7% | 29.4% | 36.7% | 0.224 | 50.5% | 0.020 | 0.025 |
| Hybrid + rerank | 16.5% | 39.4% | 51.4% | 0.286 | 66.1% | 3.874 | 11.461 |

### Scope: paper (109 questions, dev split, chunk size 450)

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR | Median latency (s) | p95 latency (s) |
|---|---|---|---|---|---|---|
| Dense | 41.3% | 70.6% | 83.5% | 0.569 | 0.020 | 0.025 |
| Hybrid | 41.3% | 67.0% | 82.6% | 0.562 | 0.021 | 0.027 |
| Hybrid + rerank | 47.7% | 69.7% | 89.9% | 0.621 | 2.081 | 3.163 |

