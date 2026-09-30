# Taste
- Prefers working incrementally and narrowly scoped: wants the agent to focus on one specific part at a time ("let's only work on that right now", "let's focus on that part") rather than spreading effort across the whole codebase. Confidence: 0.6
- Loads API keys from the project `.env` rather than hardcoding them; expects the agent to read credentials (e.g. `OPENROUTER_API_KEY`) from `.env`. Confidence: 0.7
- Prefers OpenRouter as the provider for trying/benchmarking multiple LLM models against a local baseline (Ollama), rather than only a single local model. Confidence: 0.6
- Wants model evaluation to include cross-model comparison visualizations (e.g. average accuracy and cumulative score charts) to compare models against each other. Confidence: 0.6
- Prefers concise answers: asks for explanations "in short" (e.g. root cause of an error) rather than long walkthroughs. Confidence: 0.5
- Wants the agent to actually execute and verify fixes itself (e.g. run the certificate installer inside `.venv`, do a live test call) and report back when it runs successfully, rather than only describing the steps for the user to perform. Confidence: 0.7
- Runs Python tooling through the project's `.venv` interpreter (`.venv/bin/python`) for real tests/verification. Confidence: 0.6
- Prefers consolidating related functionality into a single unified "core" function that handles all cases (single and multiple inputs) rather than keeping separate specialized functions. Confidence: 0.55
- Wants evaluation/analysis outputs persisted to disk as artifacts (both image and JSON per model, e.g. under `Results/<model_name>/`), not just displayed. Confidence: 0.5
