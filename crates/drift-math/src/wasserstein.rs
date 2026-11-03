use drift_core::{DistributionSample, DriftError, DriftMetric, DriftVerdict, Result, TestOutcome};

use crate::util::{ensure_finite, par_sorted_copy, CompSum};

/// Operational placeholder for the scale-free effect threshold tau_W (units of sigma_R).
/// It is NOT a statistical critical value; expose it through the configuration before use.
pub const DEFAULT_TAU_W: f64 = 0.1;

/// Exact W1 between the empirical distributions of two finite, ascending slices:
/// W1 = integral |F - G| dt = sum over distinct pooled points x_i of |F(x_i) - G(x_i)| (x_{i+1} - x_i),
/// because both eCDFs are right-continuous step functions. The |F - G| factor is computed
/// from integers (`|i*m - j*n| / (n*m)`).
pub fn w1_sorted(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let n = a.len() as u128;
    let m = b.len() as u128;
    let nm = a.len() as f64 * b.len() as f64;
    let (mut i, mut j) = (0usize, 0usize);
    let mut acc = CompSum::default();
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
        let next = match (a.get(i), b.get(j)) {
            (Some(&u), Some(&v)) => u.min(v),
            (Some(&u), None) => u,
            (None, Some(&v)) => v,
            (None, None) => break,
        };
        let diff = (i as u128 * m).abs_diff(j as u128 * n) as f64 / nm;
        acc.add(diff * (next - x));
    }
    acc.value()
}

/// Exact W1 on borrowed slices (sorted copies; O(n+m) extra memory).
pub fn wasserstein1_slices(reference: &[f64], target: &[f64]) -> Result<f64> {
    if reference.is_empty() || target.is_empty() {
        return Err(DriftError::Invariant(
            "wasserstein requires non-empty samples".to_string(),
        ));
    }
    ensure_finite(reference, "wasserstein1")?;
    ensure_finite(target, "wasserstein1")?;
    let (r, t) = rayon::join(|| par_sorted_copy(reference), || par_sorted_copy(target));
    let d = w1_sorted(&r, &t);
    if !d.is_finite() {
        return Err(DriftError::numerical(
            "wasserstein1",
            "distance became non-finite",
        ));
    }
    Ok(d)
}

fn sample_std(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let (mut mean, mut m2) = (0.0_f64, 0.0_f64);
    for (k, &x) in values.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (k as f64 + 1.0);
        m2 += delta * (x - mean);
    }
    (m2 / (n as f64 - 1.0)).sqrt()
}

/// Exact baseline. `chunk_size` is validated but has no effect: the exact baseline is a
/// parallel sort followed by one sequential merge. The verdict compares W1 with
/// `DEFAULT_TAU_W * sigma_R` (same units as W1); a constant reference follows the degenerate rule.
pub fn wasserstein1(
    reference: &DistributionSample,
    target: &DistributionSample,
    chunk_size: usize,
) -> Result<TestOutcome> {
    if chunk_size == 0 {
        return Err(DriftError::InvalidChunkSize(0));
    }
    let distance = wasserstein1_slices(&reference.values, &target.values)?;
    let sigma = sample_std(&reference.values);
    let (critical, verdict) = if sigma > 0.0 {
        let critical = DEFAULT_TAU_W * sigma;
        (
            critical,
            if distance > critical {
                DriftVerdict::Drifted
            } else {
                DriftVerdict::Stable
            },
        )
    } else {
        (
            0.0,
            if distance == 0.0 {
                DriftVerdict::Stable
            } else {
                DriftVerdict::Drifted
            },
        )
    };
    TestOutcome::new(
        DriftMetric::Wasserstein1,
        distance,
        None,
        critical,
        verdict,
        reference.len(),
        target.len(),
    )
}
