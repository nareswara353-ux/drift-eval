use drift_core::{DistributionSample, DriftError, DriftMetric, DriftVerdict, Result, TestOutcome};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use statrs::function::gamma::gamma_ur;

use crate::util::ensure_finite;

/// Minimum expected count per category before the asymptotic chi-square approximation is used.
pub const MIN_EXPECTED: f64 = 5.0;

#[derive(Debug, Clone)]
pub struct ChiSquareResult {
    pub statistic: f64,
    pub df: usize,
    pub p_value: f64,
    /// Cramer's V for two groups: sqrt(X^2 / N).
    pub cramers_v: f64,
    pub pooled_categories: usize,
}

#[derive(Debug, Clone)]
pub enum ChiSquareOutcome {
    Tested(ChiSquareResult),
    /// Fewer than two categories remain, or two remain and one is still below `MIN_EXPECTED`.
    Insufficient { pooled_categories: usize },
}

/// Two-sample Pearson homogeneity test on exact integer counts.
///
/// Pooling rule (paper, Section II-D): while any category has smallest expected count
/// `min(n,m)(a_k+b_k)/N < MIN_EXPECTED` and more than two categories remain, merge the two
/// categories with the smallest pooled counts a_k+b_k. The rule uses pooled counts only.
/// X^2 = sum_k (a_k m - b_k n)^2 / (n m (a_k + b_k)), df = K' - 1, and the p-value is the
/// regularised upper incomplete gamma Q(df/2, X^2/2), which stays accurate in the far tail.
pub fn chi_square_counts(reference: &[u64], target: &[u64]) -> Result<ChiSquareOutcome> {
    if reference.len() != target.len() {
        return Err(DriftError::DimensionMismatch {
            expected: reference.len(),
            actual: target.len(),
        });
    }
    let n: u64 = reference.iter().sum();
    let m: u64 = target.iter().sum();
    if n == 0 || m == 0 {
        return Err(DriftError::Invariant(
            "chi-square requires non-empty samples".to_string(),
        ));
    }
    let total = (n + m) as f64;
    let min_nm = n.min(m) as f64;

    let mut cats: Vec<(u64, u64)> = reference
        .iter()
        .zip(target.iter())
        .filter(|(a, b)| **a + **b > 0)
        .map(|(a, b)| (*a, *b))
        .collect();

    loop {
        if cats.len() <= 2 {
            break;
        }
        let any_low = cats
            .iter()
            .any(|(a, b)| min_nm * (*a + *b) as f64 / total < MIN_EXPECTED);
        if !any_low {
            break;
        }
        cats.sort_by_key(|(a, b)| *a + *b);
        let (a0, b0) = cats.remove(0);
        cats[0].0 += a0;
        cats[0].1 += b0;
    }

    let still_low = cats
        .iter()
        .any(|(a, b)| min_nm * (*a + *b) as f64 / total < MIN_EXPECTED);
    if cats.len() < 2 || still_low {
        return Ok(ChiSquareOutcome::Insufficient {
            pooled_categories: cats.len(),
        });
    }

    let (nn, mm) = (n as i128, m as i128);
    let mut statistic = 0.0_f64;
    for (a, b) in &cats {
        let num = (*a as i128) * mm - (*b as i128) * nn;
        let numf = num as f64;
        let den = (n as f64) * (m as f64) * ((*a + *b) as f64);
        statistic += numf * numf / den;
    }
    if !statistic.is_finite() {
        return Err(DriftError::numerical(
            "chi_square_counts",
            "statistic became non-finite",
        ));
    }
    let df = cats.len() - 1;
    let p_value = if (df as f64 / 2.0) <= 0.0 || (statistic / 2.0) <= 0.0 { 1.0 } else { gamma_ur(df as f64 / 2.0, statistic / 2.0) }.clamp(0.0, 1.0);
    Ok(ChiSquareOutcome::Tested(ChiSquareResult {
        statistic,
        df,
        p_value,
        cramers_v: (statistic / total).sqrt(),
        pooled_categories: cats.len(),
    }))
}

/// Chi-square on a binned continuous feature. Bins are equal-width on the REFERENCE range
/// (two overflow bins catch target values outside it), so the grid does not depend on the target.
/// For categorical features call `chi_square_counts` directly with exact category counts.
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
    ensure_finite(&reference.values, "chi_square_test")?;
    ensure_finite(&target.values, "chi_square_test")?;

    let (min, max) = reference
        .values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
    let span = max - min;
    if !span.is_finite() || span <= 0.0 {
        return Err(DriftError::numerical(
            "chi_square_test",
            "degenerate reference range",
        ));
    }
    let width = span / bins as f64;
    let index = |v: f64| -> usize {
        if v < min {
            0
        } else if v > max {
            bins + 1
        } else {
            1 + (((v - min) / width).floor() as usize).min(bins - 1)
        }
    };
    let mut counts_ref = vec![0u64; bins + 2];
    let mut counts_tgt = vec![0u64; bins + 2];
    for v in &reference.values {
        counts_ref[index(*v)] += 1;
    }
    for v in &target.values {
        counts_tgt[index(*v)] += 1;
    }

    match chi_square_counts(&counts_ref, &counts_tgt)? {
        ChiSquareOutcome::Tested(r) => {
            let dist = ChiSquared::new(r.df as f64)
                .map_err(|e| DriftError::numerical("chi_square_test", e.to_string()))?;
            let critical = dist.inverse_cdf(1.0 - alpha);
            let verdict = if r.p_value < alpha {
                DriftVerdict::Drifted
            } else {
                DriftVerdict::Stable
            };
            TestOutcome::new(
                DriftMetric::ChiSquare,
                r.statistic,
                Some(r.p_value),
                critical,
                verdict,
                reference.len(),
                target.len(),
            )
        }
        ChiSquareOutcome::Insufficient { .. } => Err(DriftError::numerical(
            "chi_square_test",
            "insufficient expected counts after pooling",
        )),
    }
}
