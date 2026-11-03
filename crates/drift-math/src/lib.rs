pub mod chi_square;
pub mod delta_correlation;
pub mod ecdf;
pub mod histogram_engine;
pub mod ks_test;
pub mod util;
pub mod wasserstein;

use drift_core::{Config, DistributionSample, DriftMetric, Result, TestOutcome};

pub use chi_square::{chi_square_counts, chi_square_test, ChiSquareOutcome, ChiSquareResult};
pub use delta_correlation::{delta_correlation, CorrelationReport, PairResult, PairStatus};
pub use ecdf::EmpiricalCdf;
pub use ks_test::{ks_two_sample, ks_two_sample_slices};
pub use wasserstein::{wasserstein1, wasserstein1_slices};

/// Lenient runner: every enabled single-column metric gets its own `Result`, so one failure
/// (for example a constant reference in the chi-square test) no longer discards the others.
///
/// Delta correlation is a multi-feature statistic and is not run here; call
/// `delta_correlation(&[&[f64]], &[&[f64]])` directly.
pub fn run_all_lenient(
    reference: &DistributionSample,
    target: &DistributionSample,
    config: &Config,
) -> Vec<(DriftMetric, Result<TestOutcome>)> {
    let alpha = config.math.alpha;
    let mut out = Vec::with_capacity(3);
    if config.math.enable_ks {
        out.push((
            DriftMetric::KolmogorovSmirnov,
            ks_two_sample(reference, target, alpha),
        ));
    }
    if config.math.enable_wasserstein {
        out.push((
            DriftMetric::Wasserstein1,
            wasserstein1(reference, target, config.math.wasserstein_chunk_size),
        ));
    }
    if config.math.enable_chi_square {
        out.push((
            DriftMetric::ChiSquare,
            chi_square_test(reference, target, config.math.chi_square_bins, alpha),
        ));
    }
    out
}

/// Strict runner: fails on the first error, as before. Prefer `run_all_lenient`.
pub fn run_all(
    reference: &DistributionSample,
    target: &DistributionSample,
    config: &Config,
) -> Result<Vec<TestOutcome>> {
    run_all_lenient(reference, target, config)
        .into_iter()
        .map(|(_, r)| r)
        .collect()
}

pub fn metric_order(metric: DriftMetric) -> u8 {
    match metric {
        DriftMetric::KolmogorovSmirnov => 0,
        DriftMetric::Wasserstein1 => 1,
        DriftMetric::ChiSquare => 2,
        DriftMetric::DeltaCorrelation => 3,
    }
}
