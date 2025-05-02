#!/usr/bin/env bash
# Prepare a release: run the checks, then print the tag to push.
#
# Usage: scripts/release.sh
#
# The version is read from pyproject.toml, so bump it there (and in
# src/auricle/__init__.py + CHANGELOG.md) before running this.
set -euo pipefail

cd "$(dirname "$0")/.."

VERSION="$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)"

echo "==> Releasing v${VERSION}"
echo "==> Running lint"
ruff check .
ruff format --check .
echo "==> Running tests"
pytest -q
echo
echo "Checks passed. To publish:"
echo "  git add -A && git commit -m \"Release v${VERSION}\""
echo "  git tag v${VERSION}"
echo "  git push origin main --tags"
