# Evaluation Suite

The evaluation suite currently contains the active runners and the shared golden test data:

- `../golden test/`: trusted evaluation cases and expected outputs
- `eval_retriever.py`: DeepEval retriever evaluation

The retriever evaluation uses the vector baseline by default. This currently performs better on the ExpenseTracker golden set. To test the cross-encoder reranker, fetch 10 vector-search candidates and keep the best 5 after reranking:

```bash
python -m evaluations.eval_retriever --collection-name repomind_e2af1fe6fdf92747 --top-k 5 --fetch-k 10 --rerank
```

Run the vector-only baseline:

```bash
python -m evaluations.eval_retriever --collection-name repomind_e2af1fe6fdf92747 --top-k 5
```

The reranker uses `cross-encoder/ms-marco-MiniLM-L-6-v2` and downloads its weights on first use.
- `evaluate.py`: basic deterministic retrieval evaluation

The shared dataset is `dataset.json`. The current runner evaluates retrieval for one Chroma collection:

```bash
python -m evaluations.evaluate <collection_name>
```

The default retriever sweep uses deterministic file-level precision and recall, so it does not require an LLM judge:

```bash
python -m evaluations.eval_retriever --collection-name repomind_e2af1fe6fdf92747
```

Golden cases are stored in `../golden test/retriever_goldens.json`. The sweep evaluates contextual recall and contextual precision with `top_k=5` by default.
Use `--semantic` to additionally run DeepEval contextual metrics with the local Ollama judge. The judge defaults to `LLM_MODEL` and `OLLAMA_BASE_URL` from `.env`; override it with `EVAL_JUDGE_MODEL` or `--judge-model`.
