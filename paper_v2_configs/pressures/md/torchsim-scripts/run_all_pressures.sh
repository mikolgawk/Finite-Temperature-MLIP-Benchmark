#!/usr/bin/env bash
set -uo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
status=0
for script in "$here"/pressure_*.py; do
    uv run "$script" "$@" || status=1
done
exit "$status"
