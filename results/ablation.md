### dev / corpus

| Setting | Variant | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| Dense (baseline) | `dense` | 13.8% | 27.5% | 33.9% | 0.208 |
| Hybrid (baseline) | `hybrid` | 14.7% | 29.4% | 36.7% | 0.224 |
| Hybrid + rerank (baseline) | `rerank[hybrid,d=20,chunk,prior=0.0]` | 11.9% | 33.9% | 46.8% | 0.236 |
| Best first stage (dev-picked) | `hybrid_maxp_dense_heavy` | 13.8% | 28.4% | 37.6% | 0.219 |
| Best reranker (dev-picked) | `rerank[hybrid,d=30,max,prior=0.5]` | 16.5% | 39.4% | 50.5% | 0.284 |

### dev / paper

| Setting | Variant | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| Dense (baseline) | `dense` | 41.3% | 70.6% | 83.5% | 0.569 |
| Hybrid (baseline) | `hybrid` | 41.3% | 67.0% | 82.6% | 0.562 |
| Hybrid + rerank (baseline) | `rerank[hybrid,d=20,chunk,prior=0.0]` | 39.4% | 70.6% | 87.2% | 0.559 |
| Best first stage (dev-picked) | `hybrid_maxp_dense_heavy` | 43.1% | 67.9% | 84.4% | 0.577 |
| Best reranker (dev-picked) | `rerank[hybrid,d=30,max,prior=0.5]` | 47.7% | 69.7% | 89.9% | 0.621 |

### test / corpus

| Setting | Variant | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| Dense (baseline) | `dense` | 14.6% | 26.4% | 34.7% | 0.214 |
| Hybrid (baseline) | `hybrid` | 16.7% | 29.9% | 36.1% | 0.240 |
| Hybrid + rerank (baseline) | `rerank[hybrid,d=20,chunk,prior=0.0]` | 19.4% | 36.1% | 43.8% | 0.285 |
| Best first stage (dev-picked) | `hybrid_maxp_dense_heavy` | 17.4% | 29.2% | 37.5% | 0.248 |
| Best reranker (dev-picked) | `rerank[hybrid,d=30,max,prior=0.5]` | 20.8% | 36.8% | 47.9% | 0.304 |

### test / paper

| Setting | Variant | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| Dense (baseline) | `dense` | 31.9% | 59.7% | 72.9% | 0.468 |
| Hybrid (baseline) | `hybrid` | 31.9% | 61.1% | 75.7% | 0.480 |
| Hybrid + rerank (baseline) | `rerank[hybrid,d=20,chunk,prior=0.0]` | 38.2% | 67.4% | 80.6% | 0.536 |
| Best first stage (dev-picked) | `hybrid_maxp_dense_heavy` | 34.7% | 63.2% | 80.6% | 0.515 |
| Best reranker (dev-picked) | `rerank[hybrid,d=30,max,prior=0.5]` | 41.0% | 68.1% | 84.7% | 0.564 |

