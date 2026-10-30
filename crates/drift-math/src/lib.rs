pub mod chi_square;
pub mod delta_correlation;
pub mod ecdf;
pub mod ks_test;
pub mod wasserstein;

use drift_core::{Config, DistributionSample, DriftMetric, Result, TestOutcome};

pub use chi_square::chi_square_test;
pub use delta_correlation::delta_correlation;
pub use ecdf::EmpiricalCdf;
pub use ks_test::ks_two_sample;
pub use wasserstein::wasserstein1;

pub fn run_all(
    reference: &DistributionSample,
    target: &DistributionSample,
    config: &Config,
) -> Result<Vec<TestOutcome>> {
    let mut outcomes = Vec::with_capacity(4);
    let alpha = config.math.alpha;

    if config.math.enable_ks {
        outcomes.push(ks_two_sample(reference, target, alpha)?);
    }
    if config.math.enable_wasserstein {
        outcomes.push(wasserstein1(
            reference,
            target,
            config.math.wasserstein_chunk_size,
        )?);
    }
    if config.math.enable_chi_square {
        outcomes.push(chi_square_test(
            reference,
            target,
            config.math.chi_square_bins,
            alpha,
        )?);
    }
    if config.math.enable_delta_correlation {
        outcomes.push(delta_correlation(reference, target, alpha)?);
    }

    Ok(outcomes)
}

pub fn metric_order(metric: DriftMetric) -> u8 {
    match metric {
        DriftMetric::KolmogorovSmirnov => 0,
        DriftMetric::Wasserstein1 => 1,
        DriftMetric::ChiSquare => 2,
        DriftMetric::DeltaCorrelation => 3,
    }
}
