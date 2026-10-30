use drift_core::{
    ColumnName, Config, DistributionSample, DriftMetric, DriftVerdict, Profiler, Threshold,
};

#[test]
fn config_default_is_valid() {
    let config = Config::default();
    assert!(config.validate().is_ok());
}

#[test]
fn config_rejects_invalid_alpha() {
    let mut config = Config::default();
    config.math.alpha = 1.5;
    assert!(config.validate().is_err());
}

#[test]
fn config_roundtrip_toml() {
    let config = Config::default();
    let serialized = config.to_toml_string().expect("serialize");
    let parsed = Config::from_toml_str(&serialized).expect("parse");
    assert_eq!(config, parsed);
}

#[test]
fn config_roundtrip_yaml() {
    let config = Config::default();
    let serialized = config.to_yaml_string().expect("serialize");
    let parsed = Config::from_yaml_str(&serialized).expect("parse");
    assert_eq!(config, parsed);
}

#[test]
fn column_name_rejects_empty() {
    assert!(ColumnName::new("").is_err());
    assert!(ColumnName::new("   ").is_err());
}

#[test]
fn distribution_sample_rejects_non_finite() {
    let column = ColumnName::new("x").expect("column");
    assert!(DistributionSample::new(column, vec![1.0, f64::NAN]).is_err());
}

#[test]
fn threshold_evaluates_verdict() {
    let threshold = Threshold::new(DriftMetric::KolmogorovSmirnov, 0.1).expect("threshold");
    assert_eq!(threshold.evaluate(0.05), DriftVerdict::Stable);
    assert_eq!(threshold.evaluate(0.5), DriftVerdict::Drifted);
}

#[test]
fn profiler_measures_phase() {
    let mut profiler = Profiler::new();
    profiler.start("phase").expect("start");
    std::thread::sleep(std::time::Duration::from_millis(5));
    profiler.stop().expect("stop");
    let snapshot = profiler.snapshot();
    assert!(snapshot.phase_millis("phase").unwrap_or(0.0) >= 0.0);
    assert!(snapshot.total_millis() >= 0.0);
}
