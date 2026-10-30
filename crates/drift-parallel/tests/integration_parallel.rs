use std::path::PathBuf;

use drift_core::{ColumnName, Config, DistributionSample, Profiler};
use drift_parallel::{ChunkPlan, Executor, ExecutorConfig, ZeroCopySlice};
use drift_report::{export_json, export_yaml, ReportBundle};

fn fixture_config(output_dir: PathBuf) -> Config {
    let mut config = Config::default();
    config.report.output_dir = output_dir.display().to_string();
    config.report.emit_html = false;
    config.report.emit_json = true;
    config.report.emit_yaml = true;
    config.report.show_progress = false;
    config
}

fn fixture_sample(name: &str, offset: f64) -> DistributionSample {
    let column = ColumnName::new(name).expect("column");
    let values: Vec<f64> = (0..128).map(|i| (i as f64) * 0.5 + offset).collect();
    DistributionSample::new(column, values).expect("sample")
}

#[test]
fn executor_runs_pipeline_and_emits_reports() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = fixture_config(dir.path().to_path_buf());
    let reference = fixture_sample("reference", 0.0);
    let target = fixture_sample("target", 0.25);

    let executor = Executor::new(ExecutorConfig { threads: Some(2) }).expect("executor");
    let outcomes = executor
        .run(|| drift_math::run_all(&reference, &target, &config))
        .expect("run_all");

    let mut profiler = Profiler::new();
    profiler.start("compute").expect("start");
    profiler.stop().expect("stop");

    let bundle = ReportBundle::from_parts(&config, outcomes, profiler.snapshot());
    let json_path = dir.path().join("report.json");
    let yaml_path = dir.path().join("report.yaml");
    export_json(&bundle, &json_path).expect("json");
    export_yaml(&bundle, &yaml_path).expect("yaml");

    assert!(json_path.exists());
    assert!(yaml_path.exists());
}

#[test]
fn chunk_plan_partitions_correctly() {
    let plan = ChunkPlan::new(100, 30).expect("plan");
    assert_eq!(plan.chunk_count, 4);
    let ranges: Vec<_> = plan.ranges();
    assert_eq!(ranges.first().map(|r| r.start), Some(0));
    assert_eq!(ranges.last().map(|r| r.end), Some(100));
}

#[test]
fn zero_copy_slice_matches_source() {
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let view = ZeroCopySlice::new(&data);
    let sub = view.subview(10, 20).expect("subview");
    assert_eq!(sub.as_slice().first().copied(), Some(10.0));
    assert_eq!(sub.as_slice().last().copied(), Some(29.0));
}

#[test]
fn config_validation_rejects_zero_chunk() {
    let mut config = Config::default();
    config.parallel.chunk_size = 0;
    assert!(config.validate().is_err());
}
