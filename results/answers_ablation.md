# Answer-quality ablation (qwen3:8b via Ollama)

30 sampled questions per split (seed 42) + 10 off-topic questions, hybrid + rerank retrieval, paper scope.
dev = QASPER validation papers 50-99 (used to pick the prompt); test = papers 0-49 (reported).

| Split | Prompt | Token F1 | Citation accuracy | Refused (answerable) | Refused (off-topic) | Median generation (s) |
|---|---|---|---|---|---|---|
| dev | v1 (original prompt) | 0.254 | 100.0% | 10.0% | 100% | 11.2 |
| dev | v2 (answer-first) | 0.318 | 98.8% | 10.0% | 100% | 9.9 |
| dev | v2 + citation repair | 0.324 | 98.8% | 6.7% | 100% | 10.2 |
| test | v1 (original prompt) | 0.297 | 100.0% | 3.3% | 100% | 10.9 |
| test | v2 + citation repair (chosen on dev) | 0.385 | 100.0% | 0.0% | 100% | 10.1 |
