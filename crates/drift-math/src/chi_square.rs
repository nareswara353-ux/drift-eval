use drift_core::{DistributionSample, DriftError, DriftMetric, DriftVerdict, Result, TestOutcome};
use statrs::distribution::{ChiSquared, ContinuousCDF};

fn compute_bounds(reference: &[f64], target: &[f64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in reference.iter().chain(target.iter()) {
        if *v < min {
            min = *v;
        }
        if *v > max {
            max = *v;
        }
    }
    (min, max)
}

fn assign_bin(value: f64, min: f64, width: f64, bins: usize) -> usize {
    if width <= 0.0 {
        return 0;
    }
    let idx = ((value - min) / width).floor();
    if idx.is_finite() && idx >= 0.0 {
        let idx_usize = idx as usize;
        idx_usize.min(bins.saturating_sub(1))
    } else {
        0
    }
}

pub fn chi_square_test(
    reference: &DistributionSample,
    target: &DistributionSample,
    bins: usize,
    alpha: f64,
) -> Result<TestOutcome> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(DriftError::InvalidAlpha(alpha));
    }
    if bins < 2 {
        return Err(DriftError::ConfigInvalid(
            "chi_square bins must be at least 2".to_string(),
        ));
    }
    if reference.is_empty() || target.is_empty() {
        return Err(DriftError::Invariant(
            "chi-square requires non-empty samples".to_string(),
        ));
    }

    let (min, max) = compute_bounds(&reference.values, &target.values);
    let span = max - min;
    if !span.is_finite() || span <= 0.0 {
        return Err(DriftError::numerical(
            "chi_square_test",
            "degenerate value range",
        ));
    }
    let width = span / bins as f64;

    let mut counts_ref = vec![0usize; bins];
    let mut counts_tgt = vec![0usize; bins];

    for v in &reference.values {
        let idx = assign_bin(*v, min, width, bins);
        counts_ref[idx] += 1;
    }
    for v in &target.values {
        let idx = assign_bin(*v, min, width, bins);
        counts_tgt[idx] += 1;
    }

    let total_ref = reference.len() as f64;
    let total_tgt = target.len() as f64;
    let total = total_ref + total_tgt;

    let mut statistic = 0.0_f64;
    let mut valid_cells = 0usize;
    for idx in 0..bins {
        let observed_ref = counts_ref[idx] as f64;
        let observed_tgt = counts_tgt[idx] as f64;
        let cell_total = observed_ref + observed_tgt;
        if cell_total <= 0.0 {
            continue;
        }
        let expected_ref = total_ref * cell_total / total;
        let expected_tgt = total_tgt * cell_total / total;
        if expected_ref > 0.0 {
            statistic += (observed_ref - expected_ref).powi(2) / expected_ref;
        }
        if expected_tgt > 0.0 {
            statistic += (observed_tgt - expected_tgt).powi(2) / expected_tgt;
        }
        valid_cells += 1;
    }

    if valid_cells < 2 {
        return Err(DriftError::numerical(
            "chi_square_test",
            "insufficient occupied bins",
        ));
    }

    let df = (valid_cells - 1) as f64;
    let distribution =
        ChiSquared::new(df).map_err(|e| DriftError::numerical("chi_square_test", e.to_string()))?;
    let p_value = (1.0 - distribution.cdf(statistic)).clamp(0.0, 1.0);
    let critical = distribution.inverse_cdf(1.0 - alpha);

    let verdict = if p_value < alpha || statistic > critical {
        DriftVerdict::Drifted
    } else {
        DriftVerdict::Stable
    };

    TestOutcome::new(
        DriftMetric::ChiSquare,
        statistic,
        Some(p_value),
        critical,
        verdict,
        reference.len(),
        target.len(),
    )
}
