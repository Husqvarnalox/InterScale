# Contributing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest -q
```

- Tests must run on CPU without network or model downloads (use `build_tiny_runner` or the
  scripted runner in `tests/conftest.py`).
- Keep the dependency direction `api → engine → model`; the API layer never calls
  `model(...)`, and the engine never imports FastAPI.
- Be honest in docs: don't describe a feature as implemented unless it is in the code and tested.
- Small, focused PRs with tests. Describe user-visible changes in `CHANGELOG.md`.
