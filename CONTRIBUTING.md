# Contributing to auricle

Thanks for considering a contribution. This is a small project, so the bar is
simple: focused changes, tests where behaviour changes, and a clean lint.

## Setting up

```bash
git clone https://github.com/naivestud/auricle.git
cd auricle
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

On CPU-only machines install the CPU torch build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Running the checks

```bash
pytest
ruff check .
ruff format --check .
```

All three must pass before a PR can merge. CI runs them across Python
3.10–3.14.

## Making changes

- One concern per pull request. Small diffs are easier to review and revert.
- Add or update tests for any behaviour change. Tests should stay
  deterministic and hermetic — they run on a randomly initialised tiny model,
  so assert properties (shapes, determinism, metric identities) rather than
  accuracy.
- Update `docs/` and the CHANGELOG for user-visible changes.
- Keep the public API surface in `auricle/__init__.py` deliberate; not
  everything needs to be re-exported.

## Commit messages

Prefer Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `tests:`,
`chore:`), but plain, descriptive messages are fine too.

## Releases

Releases are cut from tags. To prepare one:

1. Bump `version` in `pyproject.toml` and `__version__` in
   `src/auricle/__init__.py`.
2. Add a dated entry to `CHANGELOG.md`.
3. Run `scripts/release.sh`, which lints and tests, then prints the tag
   commands.
4. Push the tag; the `release.yml` workflow builds the sdist/wheel and
   attaches them to a GitHub release.
