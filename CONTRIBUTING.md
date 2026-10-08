# Contributing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest -q
```

Layout: `src/inferscale/{api,engine,model,metrics}`, `tests/`, `benchmarks/`, `docs/`.

- Dependencies point one way: `api → engine → model`. Handlers never call `model(...)` and the
  engine never imports FastAPI.
- Tests must run on CPU without network access: use `build_tiny_runner` or the scripted runner
  in `tests/conftest.py`.
- Keep docs accurate: describe a feature only if the code does it and a test covers it.
- Small PRs with tests; note user-visible changes in `CHANGELOG.md`.
