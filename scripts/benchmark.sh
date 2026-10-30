#!/usr/bin/env bash

set -euo pipefail

OUTPUT_DIR="${1:-target/criterion-report}"
BENCH_FILTER="${2:-}"

mkdir -p "${OUTPUT_DIR}"

echo "running criterion benchmarks"
if [[ -n "${BENCH_FILTER}" ]]; then
    cargo bench --workspace -- "${BENCH_FILTER}"
else
    cargo bench --workspace
fi

if [[ -d "target/criterion" ]]; then
    cp -R target/criterion "${OUTPUT_DIR}/criterion"
fi

echo "benchmark artifacts written to ${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -type f -name "*.json" | sort
