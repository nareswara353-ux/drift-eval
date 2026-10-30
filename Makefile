SHELL := /bin/bash
CARGO := cargo
MATURIN := maturin
PYTHON := python3

.PHONY: all build release test lint fmt clippy bench clean py-develop py-build ci docs

all: fmt lint test

build:
	$(CARGO) build --workspace

release:
	$(CARGO) build --workspace --release

test:
	RUST_BACKTRACE=1 $(CARGO) test --workspace --all-features

lint: fmt clippy

fmt:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery

bench:
	$(CARGO) bench --workspace

clean:
	$(CARGO) clean
	rm -rf crates/drift-py/target
	rm -rf dist

py-develop:
	cd crates/drift-py && $(MATURIN) develop --release

py-build:
	cd crates/drift-py && $(MATURIN) build --release

docs:
	$(CARGO) doc --workspace --no-deps --document-private-items

ci: fmt clippy test release
