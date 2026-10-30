use drift_core::{DistributionSample, DriftError, DriftMetric, DriftVerdict, Result, TestOutcome};
use rayon::prelude::*;

const ASYMPTOTIC_THRESHOLD: f64 = 1.0e-3;
const NUMERICAL_EPSILON: f64 = 1.0e-12;

fn sort_parallel(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

fn empirical_cdf_at(sorted: &[f64], x: f64) -> f64 {
    let idx = sorted.partition_point(|value| *value <= x);
    idx as f64 / sorted.len() as f64
}

fn kolmogorov_sf(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0_f64;
    let mut converged = false;
    for k in 1..=1000_i32 {
        let term =
            2.0 * (-1.0_f64).powi(k - 1) * (-2.0 * (k as f64).powi(2) * lambda * lambda).exp();
        sum += term;
        if term.abs() < NUMERICAL_EPSILON {
            converged = true;
            break;
        }
    }
    if converged {
        sum.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

pub fn ks_two_sample(
    reference: &DistributionSample,
    target: &DistributionSample,
    alpha: f64,
) -> Result<TestOutcome> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(DriftError::InvalidAlpha(alpha));
    }
    let n = reference.len();
    let m = target.len();
    if n == 0 || m == 0 {
        return Err(DriftError::Invariant(
            "ks test requires non-empty samples".to_string(),
        ));
    }

    let (sorted_ref, sorted_tgt) = rayon::join(
        || sort_parallel(&reference.values),
        || sort_parallel(&target.values),
    );

    let mut points = Vec::with_capacity(n + m);
    points.extend_from_slice(&sorted_ref);
    points.extend_from_slice(&sorted_tgt);
    points.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let d_stat = points
        .par_iter()
        .map(|x| (empirical_cdf_at(&sorted_ref, *x) - empirical_cdf_at(&sorted_tgt, *x)).abs())
        .reduce(|| 0.0_f64, f64::max);

    if !d_stat.is_finite() {
        return Err(DriftError::numerical(
            "ks_two_sample",
            "statistic became non-finite",
        ));
    }

    let n_f = n as f64;
    let m_f = m as f64;
    let en = (n_f * m_f / (n_f + m_f)).sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d_stat;
    let p_value = kolmogorov_sf(lambda).clamp(0.0, 1.0);

    let critical = (-0.5 * (alpha / 2.0).ln()).sqrt() / en;
    let verdict = if p_value < alpha || d_stat > critical {
        DriftVerdict::Drifted
    } else {
        DriftVerdict::Stable
    };

    let _ = ASYMPTOTIC_THRESHOLD;

    TestOutcome::new(
        DriftMetric::KolmogorovSmirnov,
        d_stat,
        Some(p_value),
        critical,
        verdict,
        n,
        m,
    )
}
