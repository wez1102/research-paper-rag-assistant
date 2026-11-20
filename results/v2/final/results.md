### Scope: corpus (209 questions, final split, chunk size 450)

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR | Right paper in top 5 | Median latency (s) | p95 latency (s) |
|---|---|---|---|---|---|---|---|
| Dense | 17.2% | 28.7% | 31.6% | 0.227 | 49.8% | 0.047 | 0.972 |
| Hybrid | 14.4% | 31.6% | 38.8% | 0.229 | 56.9% | 0.046 | 0.329 |
| Hybrid + rerank | 22.5% | 36.8% | 45.0% | 0.305 | 62.2% | 1.818 | 2.482 |

### Scope: paper (209 questions, final split, chunk size 450)

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR | Median latency (s) | p95 latency (s) |
|---|---|---|---|---|---|---|
| Dense | 38.3% | 67.9% | 76.1% | 0.528 | 0.041 | 0.175 |
| Hybrid | 39.7% | 67.0% | 79.4% | 0.543 | 0.020 | 0.027 |
| Hybrid + rerank | 44.0% | 70.8% | 85.2% | 0.593 | 0.848 | 2.015 |

