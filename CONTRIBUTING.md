# Contributing to AskDB

Thanks for your interest in contributing!

## Quick rules
- Keep PRs small and focused (one feature/fix per PR)
- Add/adjust tests when behavior changes
- Follow existing code style and naming conventions
- Avoid committing secrets or sample credentials

## Dev setup
1. Create and activate venv
2. `pip install -r requirements.txt -r requirements-dev.txt`
3. Create `.env` (see README)
4. Run backend: `python code1.py`
5. Run frontend: `cd frontend && npm install && npm run dev`
6. Run tests: `pytest -q`

## PR checklist
- [ ] Tests pass (`pytest -q`)
- [ ] Frontend build passes (`npm run build`)
- [ ] Updated README/docs if user-facing behavior changed
- [ ] No secrets in commits (.env, tokens, connection strings)
