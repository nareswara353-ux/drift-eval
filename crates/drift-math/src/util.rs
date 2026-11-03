//! Shared helpers for the exact baseline and the histogram engine.
use drift_core::{DriftError, Result};
use rayon::prelude::*;

/// Rejects NaN and +/-infinity. Baseline routines require finite input.
pub(crate) fn ensure_finite(values: &[f64], op: &str) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(DriftError::numerical(op, "encountered non-finite value"));
    }
    Ok(())
}

/// Sorted copy (O(N) extra memory). Input must be finite; `total_cmp` gives a total order.
pub(crate) fn par_sorted_copy(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    v.par_sort_unstable_by(f64::total_cmp);
    v
}

/// Neumaier compensated sum. Mergeable, so it can be carried in per-worker accumulators.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompSum {
    s: f64,
    c: f64,
}

impl CompSum {
    #[inline]
    pub fn add(&mut self, x: f64) {
        let t = self.s + x;
        if self.s.abs() >= x.abs() {
            self.c += (self.s - t) + x;
        } else {
            self.c += (x - t) + self.s;
        }
        self.s = t;
    }
    pub fn merge(&mut self, other: &CompSum) {
        self.add(other.s);
        self.c += other.c;
    }
    pub fn value(&self) -> f64 {
        self.s + self.c
    }
}

/// Smallest f64 greater than `x` (stable-Rust replacement for `f64::next_up`).
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        x
    } else if x == 0.0 {
        f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

/// Largest f64 smaller than `x`.
pub fn next_down(x: f64) -> f64 {
    -next_up(-x)
}
