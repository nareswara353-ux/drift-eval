# drift-py

Python bindings for the `drift-eval` statistical drift detection engine, built with PyO3 and Maturin.

## Functions

- `ks_test(reference, target, alpha=0.05)`
- `wasserstein1(reference, target, chunk_size=4096)`
- `chi_square_test(reference, target, bins=32, alpha=0.05)`
- `delta_correlation(reference, target, alpha=0.05)`
- `run_all(reference, target, alpha=0.05)`

Each function returns a dict with `metric`, `statistic`, `p_value`, `critical`, `verdict`, `sample_size_reference`, and `sample_size_target`.

## License

MIT OR Apache-2.0
