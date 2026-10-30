
# drift-eval

High-Performance Statistical Validation & Distribution Drift Evaluator Engine written in Rust.

## Overview

`drift-eval` is a multi-crate Rust workspace that detects distribution drift between reference and target datasets using statistical hypothesis tests and distance metrics. It exposes a CLI, a native Rust library, and Python bindings via PyO3.

## Workspace Layout

| Crate | Responsibility |
|-------|----------------|
| `drift-core` | Shared types, configuration, error handling, profiler |
| `drift-math` | eCDF, parallel two-sample KS test, streaming Wasserstein W1, chi-square, delta correlation |
| `drift-parallel` | Rayon executor, memmap2 loader, chunking, zero-copy buffers |
| `drift-report` | Tera HTML reports, JSON/YAML exporters, progress bars |
| `drift-cli` | Clap-based CLI entrypoint and config loader |
| `drift-py` | PyO3 C-extensions packaged with Maturin |

## Quickstart

```bash
cargo build --release --workspace
cargo run --release --bin drift-eval -- run --config configs/default.toml
Python Bindings
bash
pip install maturin
maturin develop --release -m crates/drift-py/Cargo.toml
python -c "import drift_py; print(drift_py.ks_test([1,2,3], [2,3,4]))"
Statistics Implemented
Kolmogorov-Smirnov two-sample test (parallel, O(n log n) sort, O(n) scan)

Wasserstein-1 distance (streaming, chunked merge)

Pearson chi-square test of independence

Delta correlation drift for monotonicity checks

Empirical CDF (eCDF) evaluation with binary search

License
Dual-licensed under MIT OR Apache-2.0.