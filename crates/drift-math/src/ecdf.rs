use drift_core::{DriftError, Result};

use crate::util::par_sorted_copy;

#[derive(Debug, Clone)]
pub struct EmpiricalCdf {
    sorted: Vec<f64>,
}

impl EmpiricalCdf {
    pub fn from_slice(values: &[f64]) -> Result<Self> {
        if values.is_empty() {
            return Err(DriftError::Invariant(
                "ecdf requires non-empty input".to_string(),
            ));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(DriftError::numerical(
                "ecdf",
                "encountered non-finite value",
            ));
        }
        Ok(Self {
            sorted: par_sorted_copy(values),
        })
    }

    pub fn len(&self) -> usize {
        self.sorted.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }

    pub fn as_sorted(&self) -> &[f64] {
        &self.sorted
    }

    /// Right-continuous eCDF: F(x) = |{v <= x}| / n.
    pub fn evaluate(&self, x: f64) -> f64 {
        if self.sorted.is_empty() {
            return 0.0;
        }
        let idx = self.sorted.partition_point(|value| *value <= x);
        idx as f64 / self.sorted.len() as f64
    }

    pub fn evaluate_batch(&self, queries: &[f64]) -> Vec<f64> {
        queries.iter().map(|q| self.evaluate(*q)).collect()
    }

    /// Exact sup_t |F(t) - G(t)| by a single merge pass (no tolerance-based de-duplication).
    pub fn max_abs_diff(&self, other: &Self) -> f64 {
        ks_statistic_sorted(&self.sorted, &other.sorted)
    }
}

/// Integer numerator of the exact two-sample KS statistic:
/// `n*m*D = max_x | #{a<=x}*m - #{b<=x}*n |`, evaluated at every distinct pooled value.
///
/// Preconditions (not re-checked here): both slices are finite and sorted ascending.
/// The integer form avoids any rounding inside the maximisation.
pub fn ks_numerator_sorted(a: &[f64], b: &[f64]) -> u128 {
    let n = a.len() as u128;
    let m = b.len() as u128;
    let (mut i, mut j) = (0usize, 0usize);
    let mut best: u128 = 0;
    while i < a.len() || j < b.len() {
        let x = match (a.get(i), b.get(j)) {
            (Some(&u), Some(&v)) => u.min(v),
            (Some(&u), None) => u,
            (None, Some(&v)) => v,
            (None, None) => break,
        };
        while i < a.len() && a[i] <= x {
            i += 1;
        }
        while j < b.len() && b[j] <= x {
            j += 1;
        }
        best = best.max((i as u128 * m).abs_diff(j as u128 * n));
    }
    best
}

/// Exact KS statistic D of two finite, ascending slices (0.0 if either is empty).
pub fn ks_statistic_sorted(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    ks_numerator_sorted(a, b) as f64 / (a.len() as f64 * b.len() as f64)
}
