//! Delta correlation between FEATURES: Delta R = |R_ref - R_eval| (Pearson), listwise deletion.
//!
//! This replaces the earlier single-column routine, which correlated a column with its own ranks
//! and therefore did not measure a change in feature-to-feature correlation.
use drift_core::{DriftError, Result};
use rayon::prelude::*;
use statrs::function::erf::erfc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairStatus {
    Ok,
    /// At least one feature has zero variance in one of the samples; R is undefined.
    ConstantFeature,
    /// Fewer than four complete rows in one sample; the Fisher-z standard error is undefined.
    InsufficientRows,
    /// |r| = 1 in one sample; artanh is infinite and no p-value is produced.
    PerfectCorrelation,
}

#[derive(Debug, Clone)]
pub struct PairResult {
    pub i: usize,
    pub j: usize,
    pub r_ref: Option<f64>,
    pub r_tgt: Option<f64>,
    pub delta: Option<f64>,
    pub p_value: Option<f64>,
    pub status: PairStatus,
}

#[derive(Debug, Clone)]
pub struct CorrelationReport {
    pub n_ref: usize,
    pub n_tgt: usize,
    pub dropped_ref: usize,
    pub dropped_tgt: usize,
    pub pairs: Vec<PairResult>,
    /// Frobenius norm of the full symmetric Delta R matrix (zero diagonal), over defined pairs.
    pub frobenius: f64,
    pub max_delta: f64,
}

struct BlockCorr {
    n: usize,
    dropped: usize,
    /// One entry per requested pair; None when a feature is constant or n < 2.
    r: Vec<Option<f64>>,
}

fn block_correlations(cols: &[&[f64]], pairs: &[(usize, usize)]) -> Result<BlockCorr> {
    let d = cols.len();
    let rows = cols[0].len();
    for c in cols {
        if c.len() != rows {
            return Err(DriftError::DimensionMismatch {
                expected: rows,
                actual: c.len(),
            });
        }
    }
    let mask: Vec<bool> = (0..rows)
        .into_par_iter()
        .map(|r| cols.iter().all(|c| c[r].is_finite()))
        .collect();
    let n = mask.iter().filter(|b| **b).count();
    let dropped = rows - n;
    if n < 2 {
        return Ok(BlockCorr {
            n,
            dropped,
            r: vec![None; pairs.len()],
        });
    }
    let means: Vec<f64> = (0..d)
        .into_par_iter()
        .map(|c| {
            let s: f64 = cols[c]
                .iter()
                .zip(mask.iter())
                .filter(|(_, ok)| **ok)
                .map(|(v, _)| *v)
                .sum();
            s / n as f64
        })
        .collect();
    let ss: Vec<f64> = (0..d)
        .into_par_iter()
        .map(|c| {
            cols[c]
                .iter()
                .zip(mask.iter())
                .filter(|(_, ok)| **ok)
                .map(|(v, _)| (*v - means[c]) * (*v - means[c]))
                .sum()
        })
        .collect();
    let r: Vec<Option<f64>> = pairs
        .par_iter()
        .map(|&(i, j)| {
            if ss[i] == 0.0 || ss[j] == 0.0 {
                return None;
            }
            let mut sxy = 0.0_f64;
            for row in 0..rows {
                if mask[row] {
                    sxy += (cols[i][row] - means[i]) * (cols[j][row] - means[j]);
                }
            }
            Some((sxy / (ss[i] * ss[j]).sqrt()).clamp(-1.0, 1.0))
        })
        .collect();
    Ok(BlockCorr { n, dropped, r })
}

/// Pair p-value via the two-sample Fisher z test, using erfc so the far tail stays accurate:
/// z = (artanh r1 - artanh r2) / sqrt(1/(n-3) + 1/(m-3)), p = erfc(|z| / sqrt(2)).
pub fn fisher_z_p_value(r1: f64, n1: usize, r2: f64, n2: usize) -> f64 {
    let z = (r1.atanh() - r2.atanh()) / (1.0 / (n1 as f64 - 3.0) + 1.0 / (n2 as f64 - 3.0)).sqrt();
    erfc(z.abs() / std::f64::consts::SQRT_2).clamp(0.0, 1.0)
}

/// Exact baseline on borrowed columns (zero-copy). Rows with any non-finite value in any column
/// are dropped from that block (listwise deletion) and counted in `dropped_*`.
pub fn delta_correlation(reference: &[&[f64]], target: &[&[f64]]) -> Result<CorrelationReport> {
    let d = reference.len();
    if d < 2 {
        return Err(DriftError::Invariant(
            "delta correlation requires at least two features".to_string(),
        ));
    }
    if target.len() != d {
        return Err(DriftError::DimensionMismatch {
            expected: d,
            actual: target.len(),
        });
    }
    let pairs: Vec<(usize, usize)> = (0..d)
        .flat_map(|i| ((i + 1)..d).map(move |j| (i, j)))
        .collect();
    let (rb, tb) = (
        block_correlations(reference, &pairs)?,
        block_correlations(target, &pairs)?,
    );

    let mut results = Vec::with_capacity(pairs.len());
    let (mut sq, mut max_delta) = (0.0_f64, 0.0_f64);
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        let (r1, r2) = (rb.r[idx], tb.r[idx]);
        let delta = match (r1, r2) {
            (Some(a), Some(b)) => Some((a - b).abs()),
            _ => None,
        };
        let (status, p_value) = if rb.n <= 3 || tb.n <= 3 {
            (PairStatus::InsufficientRows, None)
        } else if r1.is_none() || r2.is_none() {
            (PairStatus::ConstantFeature, None)
        } else if r1.unwrap().abs() >= 1.0 || r2.unwrap().abs() >= 1.0 {
            (PairStatus::PerfectCorrelation, None)
        } else {
            (
                PairStatus::Ok,
                Some(fisher_z_p_value(r1.unwrap(), rb.n, r2.unwrap(), tb.n)),
            )
        };
        if let Some(dl) = delta {
            sq += 2.0 * dl * dl;
            max_delta = max_delta.max(dl);
        }
        results.push(PairResult {
            i,
            j,
            r_ref: r1,
            r_tgt: r2,
            delta,
            p_value,
            status,
        });
    }
    Ok(CorrelationReport {
        n_ref: rb.n,
        n_tgt: tb.n,
        dropped_ref: rb.dropped,
        dropped_tgt: tb.dropped,
        pairs: results,
        frobenius: sq.sqrt(),
        max_delta,
    })
}
