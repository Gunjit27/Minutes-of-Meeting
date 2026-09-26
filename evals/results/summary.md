| Retrieval | hit@3 | mrr | evidence_recall | faithfulness | answer_relevancy | contextual_relevancy | p50_retrieval_ms |
|---|---|---|---|---|---|---|---|
| vector | 0.79 | 0.68 | 0.33 | 0.58 | 0.64 | 0.33 | 23 |
| hybrid | 0.79 | 0.69 | 0.40 | – | – | – | 19 |
| hybrid_rerank | 0.83 | 0.78 | 0.46 | 0.60 | 0.64 | 0.35 | 317 |

29 QMSum questions over 10 meetings · top-3 chunks · answer LLM: groq · embeddings: hf · judge: llama3.1:8b (Ollama) · – = not judged
