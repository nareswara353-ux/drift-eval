use drift_core::{DistributionSample, DriftError, DriftMetric, DriftVerdict, Result, TestOutcome};

use crate::ecdf::ks_statistic_sorted;
use crate::util::{ensure_finite, par_sorted_copy};

/// Asymptotic Kolmogorov survival function Q(lambda) = 2 * sum_{k>=1} (-1)^{k-1} exp(-2 k^2 lambda^2).
///
/// For lambda < 1 the equivalent theta-function form is used because the alternating series
/// converges slowly there: CDF = sqrt(2 pi)/lambda * sum_{k>=1} exp(-(2k-1)^2 pi^2 / (8 lambda^2)).
/// Q is non-increasing in lambda (relied upon by the p-value interval of the histogram engine).
pub fn kolmogorov_sf(lambda: f64) -> f64 {
    if lambda.is_nan() || lambda <= 0.0 {
        return 1.0;
    }
    if lambda < 1.0 {
        let c = std::f64::consts::PI * std::f64::consts::PI / (8.0 * lambda * lambda);
        let mut sum = 0.0_f64;
        for k in 1..=60_i32 {
            let odd = (2 * k - 1) as f64;
            let term = (-odd * odd * c).exp();
            sum += term;
            if term < 1.0e-18 {
                break;
            }
        }
        let cdf = (2.0 * std::f64::consts::PI).sqrt() / lambda * sum;
        (1.0 - cdf).clamp(0.0, 1.0)
    } else {
        let mut sum = 0.0_f64;
        for k in 1..=200_i32 {
            let kf = k as f64;
            let term = 2.0 * (-2.0 * kf * kf * lambda * lambda).exp();
            if k % 2 == 1 {
                sum += term;
            } else {
                sum -= term;
            }
            if term < 1.0e-18 {
                break;
            }
        }
        sum.clamp(0.0, 1.0)
    }
}

/// Asymptotic p-value p(D) = Q(sqrt(n_e) * D) with n_e = n*m/(n+m) (Eq. (4) of the paper).
pub fn ks_p_value(d: f64, n: usize, m: usize) -> f64 {
    let n_f = n as f64;
    let m_f = m as f64;
    let en = (n_f * m_f / (n_f + m_f)).sqrt();
    kolmogorov_sf(en * d)
}

/// Exact baseline on borrowed slices. Uses sorted copies, i.e. O(n+m) extra memory.
pub fn ks_two_sample_slices(reference: &[f64], target: &[f64], alpha: f64) -> Result<TestOutcome> {
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
    ensure_finite(reference, "ks_two_sample")?;
    ensure_finite(target, "ks_two_sample")?;

    let (sorted_ref, sorted_tgt) = rayon::join(|| par_sorted_copy(reference), || par_sorted_copy(target));
    let d_stat = ks_statistic_sorted(&sorted_ref, &sorted_tgt);
    if !d_stat.is_finite() {
        return Err(DriftError::numerical(
            "ks_two_sample",
            "statistic became non-finite",
        ));
    }

    let en = ((n as f64) * (m as f64) / ((n + m) as f64)).sqrt();
    let p_value = ks_p_value(d_stat, n, m);
    // First-term approximation of the asymptotic critical value at level alpha (reporting only).
    let critical = (-0.5 * (alpha / 2.0).ln()).sqrt() / en;
    let verdict = if p_value < alpha {
        DriftVerdict::Drifted
    } else {
        DriftVerdict::Stable
    };

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

pub fn ks_two_sample(
    reference: &DistributionSample,
    target: &DistributionSample,
    alpha: f64,
) -> Result<TestOutcome> {
    ks_two_sample_slices(&reference.values, &target.values, alpha)
}
