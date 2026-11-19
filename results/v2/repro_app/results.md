### Scope: corpus (144 questions, chunk size 450)

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR | Median latency (s) |
|---|---|---|---|---|---|
| Dense | 14.6% | 26.4% | 34.7% | 0.214 | 0.019 |
| Hybrid | 16.7% | 29.9% | 36.1% | 0.238 | 0.020 |
| Hybrid + rerank | 20.8% | 37.5% | 47.9% | 0.305 | 3.611 |

### Scope: paper (144 questions, chunk size 450)

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR | Median latency (s) |
|---|---|---|---|---|---|
| Dense | 31.9% | 59.7% | 72.9% | 0.468 | 0.019 |
| Hybrid | 31.9% | 61.1% | 75.7% | 0.480 | 0.020 |
| Hybrid + rerank | 41.0% | 68.1% | 84.7% | 0.564 | 2.774 |

