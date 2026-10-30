use drift_core::{DistributionSample, DriftError, DriftMetric, DriftVerdict, Result, TestOutcome};

fn pearson(xs: &[f64], ys: &[f64]) -> Result<f64> {
    if xs.len() != ys.len() {
        return Err(DriftError::DimensionMismatch {
            expected: xs.len(),
            actual: ys.len(),
        });
    }
    let n = xs.len();
    if n < 2 {
        return Err(DriftError::Invariant(
            "pearson requires at least two points".to_string(),
        ));
    }
    let n_f = n as f64;
    let mean_x = xs.iter().sum::<f64>() / n_f;
    let mean_y = ys.iter().sum::<f64>() / n_f;

    let mut num = 0.0_f64;
    let mut den_x = 0.0_f64;
    let mut den_y = 0.0_f64;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    let denom = (den_x * den_y).sqrt();
    if denom <= f64::EPSILON {
        return Ok(0.0);
    }
    Ok((num / denom).clamp(-1.0, 1.0))
}

fn rank(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; values.len()];
    let mut i = 0usize;
    while i < indexed.len() {
        let mut j = i;
        while j + 1 < indexed.len() && (indexed[j + 1].1 - indexed[i].1).abs() < f64::EPSILON {
            j += 1;
        }
        let avg_rank = ((i + j) as f64 / 2.0) + 1.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    ranks
}

pub fn delta_correlation(
    reference: &DistributionSample,
    target: &DistributionSample,
    alpha: f64,
) -> Result<TestOutcome> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(DriftError::InvalidAlpha(alpha));
    }
    if reference.is_empty() || target.is_empty() {
        return Err(DriftError::Invariant(
            "delta correlation requires non-empty samples".to_string(),
        ));
    }

    let ref_ranks = rank(&reference.values);
    let tgt_ranks = rank(&target.values);
    let rho_ref = pearson(&ref_ranks, &reference.values)?;
    let rho_tgt = pearson(&tgt_ranks, &target.values)?;

    let delta = (rho_ref - rho_tgt).abs();
    if !delta.is_finite() {
        return Err(DriftError::numerical(
            "delta_correlation",
            "delta became non-finite",
        ));
    }

    let n = reference.len().min(target.len()) as f64;
    let fisher_z = 0.5 * ((1.0 + delta) / (1.0 - delta).max(f64::EPSILON)).ln();
    let se = 1.0 / (n - 3.0).max(1.0).sqrt();
    let z_score = fisher_z.abs() / se;

    let critical = 1.959_963_984_540_054_f64;
    let verdict = if z_score > critical {
        DriftVerdict::Drifted
    } else {
        DriftVerdict::Stable
    };

    let p_value = (2.0 * (1.0 - normal_cdf(z_score.abs()))).clamp(0.0, 1.0);
    let _ = alpha;

    TestOutcome::new(
        DriftMetric::DeltaCorrelation,
        delta,
        Some(p_value),
        critical,
        verdict,
        reference.len(),
        target.len(),
    )
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x_abs);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x_abs * x_abs).exp();
    sign * y
}
