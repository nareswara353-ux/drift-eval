use drift_core::{ColumnName, Config, DistributionSample, DriftMetric, DriftVerdict};
use drift_math::{
    chi_square_test, delta_correlation, ks_two_sample, run_all, wasserstein1, EmpiricalCdf,
};

fn sample(name: &str, values: Vec<f64>) -> DistributionSample {
    let column = ColumnName::new(name).expect("column");
    DistributionSample::new(column, values).expect("sample")
}

#[test]
fn ecdf_evaluates_monotonic() {
    let ecdf = EmpiricalCdf::from_slice(&[1.0, 2.0, 3.0, 4.0]).expect("ecdf");
    let q1 = ecdf.evaluate(1.0);
    let q2 = ecdf.evaluate(2.5);
    let q3 = ecdf.evaluate(100.0);
    assert!(q1 <= q2 && q2 <= q3);
    assert!((q3 - 1.0).abs() < 1e-12);
}

#[test]
fn ks_detects_identical_samples_as_stable() {
    let reference = sample("ref", (0..200).map(|i| i as f64).collect());
    let target = sample("tgt", (0..200).map(|i| i as f64).collect());
    let outcome = ks_two_sample(&reference, &target, 0.05).expect("ks");
    assert_eq!(outcome.metric, DriftMetric::KolmogorovSmirnov);
    assert_eq!(outcome.verdict, DriftVerdict::Stable);
    assert!(outcome.statistic < 1e-9);
}

#[test]
fn ks_detects_shifted_samples_as_drifted() {
    let reference = sample("ref", (0..200).map(|i| i as f64).collect());
    let target = sample("tgt", (0..200).map(|i| i as f64 + 100.0).collect());
    let outcome = ks_two_sample(&reference, &target, 0.05).expect("ks");
    assert_eq!(outcome.verdict, DriftVerdict::Drifted);
}

#[test]
fn wasserstein_zero_for_identical() {
    let reference = sample("ref", vec![1.0, 2.0, 3.0, 4.0]);
    let target = sample("tgt", vec![1.0, 2.0, 3.0, 4.0]);
    let outcome = wasserstein1(&reference, &target, 16).expect("w1");
    assert!(outcome.statistic < 1e-9);
}

#[test]
fn wasserstein_positive_for_shifted() {
    let reference = sample("ref", vec![1.0, 2.0, 3.0, 4.0]);
    let target = sample("tgt", vec![11.0, 12.0, 13.0, 14.0]);
    let outcome = wasserstein1(&reference, &target, 16).expect("w1");
    assert!(outcome.statistic > 0.0);
}

#[test]
fn chi_square_stable_for_identical() {
    let values: Vec<f64> = (0..500).map(|i| (i as f64).sin()).collect();
    let reference = sample("ref", values.clone());
    let target = sample("tgt", values);
    let outcome = chi_square_test(&reference, &target, 8, 0.05).expect("chi");
    assert_eq!(outcome.verdict, DriftVerdict::Stable);
}

#[test]
fn delta_correlation_returns_outcome() {
    let reference = sample("ref", (0..100).map(|i| i as f64).collect());
    let target = sample("tgt", (0..100).map(|i| (i as f64).powi(2)).collect());
    let outcome = delta_correlation(&reference, &target, 0.05).expect("delta");
    assert!(outcome.statistic >= 0.0);
}

#[test]
fn run_all_respects_config() {
    let mut config = Config::default();
    config.math.enable_chi_square = false;
    config.math.enable_delta_correlation = false;

    let reference = sample("ref", (0..200).map(|i| i as f64).collect());
    let target = sample("tgt", (0..200).map(|i| i as f64 + 0.5).collect());
    let outcomes = run_all(&reference, &target, &config).expect("run_all");

    assert_eq!(outcomes.len(), 2);
    assert!(outcomes
        .iter()
        .any(|o| o.metric == DriftMetric::KolmogorovSmirnov));
    assert!(outcomes
        .iter()
        .any(|o| o.metric == DriftMetric::Wasserstein1));
}
