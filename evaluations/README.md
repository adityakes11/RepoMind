# Evaluation Suite

The evaluation suite currently contains the active runners and the shared golden test data:

- `../golden_tests/`: trusted evaluation cases and expected outputs
- `eval_retriever.py`: DeepEval retriever evaluation
- `eval_generator.py`: generator faithfulness and answer-relevance evaluation
- `eval_rag_pipeline.py`: end-to-end RAG triad evaluation
- `eval_application.py`: reference-based correctness, completeness, and style evaluation
- `eval_toxicity.py`: toxicity safety and false-positive evaluation

The retriever evaluation uses the vector baseline by default. This currently performs better on the ExpenseTracker golden set. To test the cross-encoder reranker, fetch 10 vector-search candidates and keep the best 5 after reranking:

```bash
python -m evaluations.eval_retriever --collection-name repomind_e2af1fe6fdf92747 --top-k 5 --fetch-k 10 --rerank
```

Run the vector-only baseline:

```bash
python -m evaluations.eval_retriever --collection-name repomind_e2af1fe6fdf92747 --top-k 5
```

The reranker uses `cross-encoder/ms-marco-MiniLM-L-6-v2` and downloads its weights on first use.

Run generator evaluation with the golden context, isolated from retrieval:

```bash
python -m evaluations.eval_generator --judge-model qwen2.5:3b --limit 3 --metrics faithfulness
```

The default run is a 3-case faithfulness smoke test. Run the complete suite only when needed:

```bash
python -m evaluations.eval_generator --judge-model qwen2.5:3b --all --metrics both
```

The application generator uses `LLM_MODEL` from `.env`. DeepEval uses the local Ollama judge from `EVAL_JUDGE_MODEL`, or falls back to `LLM_MODEL`.

Run the end-to-end RAG triad evaluation:

```bash
python -m evaluations.eval_rag_pipeline --collection-name repomind_e2af1fe6fdf92747 --judge-model qwen2.5:3b --limit 3 --top-k 5
```

This evaluates contextual relevancy, faithfulness, and answer relevancy using live retriever and generator output. Use `--all` to run all 10 golden cases.

Application-level correctness evaluation uses the MCP ExpenseTracker matrix in `../golden_tests/correctness_goldens.json`:

```bash
python -m evaluations.eval_application --collection-name repomind_e2af1fe6fdf92747 --judge-model qwen2.5:3b --limit 3
```

The matrix is repository-specific and should be run against the ExpenseTracker MCP collection.

Run the toxicity safety evaluation:

```bash
python -m evaluations.eval_toxicity --collection-name repomind_e2af1fe6fdf92747 --judge-model qwen2.5:3b --limit 3
```

Run all 15 attack and benign guard cases:

```bash
python -m evaluations.eval_toxicity --collection-name repomind_e2af1fe6fdf92747 --judge-model qwen2.5:3b --all
```

Toxicity scores are lower-is-better; the default threshold is `0.3`.
- `evaluate.py`: basic deterministic retrieval evaluation

The shared dataset is `dataset.json`. The current runner evaluates retrieval for one Chroma collection:

```bash
python -m evaluations.evaluate <collection_name>
```

The default retriever sweep uses deterministic file-level precision and recall, so it does not require an LLM judge:

```bash
python -m evaluations.eval_retriever --collection-name repomind_e2af1fe6fdf92747
```

Golden cases are stored in `../golden_tests/retriever_goldens.json`. The sweep evaluates contextual recall and contextual precision with `top_k=5` by default.
Use `--semantic` to additionally run DeepEval contextual metrics with the local Ollama judge. The judge defaults to `LLM_MODEL` and `OLLAMA_BASE_URL` from `.env`; override it with `EVAL_JUDGE_MODEL` or `--judge-model`.

## Full Evaluation Suite & Regression Comparison

Run the complete 6-pillar suite:

```bash
python -m evaluations.run_suite --collection-name <collection_name>
```

- **First run**: Creates `baselines/baseline.json` (and a timestamped snapshot).
- **Subsequent runs**: Automatically detects `baselines/baseline.json` and creates `baselines/candidate.json` (and a timestamped snapshot).
- Use `--as-baseline` or `--force-baseline` if you intentionally want to overwrite the baseline snapshot.
- Use `--as-candidate` or `--force-candidate` to force a candidate snapshot.

Compare candidate against baseline:

```bash
python -m evaluations.compare
```

Or show all informational metrics as well:

```bash
python -m evaluations.compare --all
```

