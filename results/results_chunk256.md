### Scope: corpus (144 questions, chunk size 256)

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR | Median latency (s) |
|---|---|---|---|---|---|
| Dense | 14.6% | 28.5% | 32.6% | 0.215 | 0.012 |
| Hybrid | 14.6% | 30.6% | 39.6% | 0.233 | 0.012 |
| Hybrid + rerank | 20.8% | 34.7% | 43.1% | 0.285 | 0.574 |

### Scope: paper (144 questions, chunk size 256)

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR | Median latency (s) |
|---|---|---|---|---|---|
| Dense | 31.9% | 55.6% | 68.8% | 0.452 | 0.012 |
| Hybrid | 31.9% | 53.5% | 74.3% | 0.457 | 0.013 |
| Hybrid + rerank | 36.1% | 60.4% | 75.0% | 0.496 | 0.566 |

