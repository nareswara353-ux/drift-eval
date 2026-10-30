mod cli;
mod config_loader;

use drift_core::{ColumnName, Config, DistributionSample, Profiler, Result};
use drift_parallel::Executor;
use drift_report::{persist_all, ProgressReporter, ReportBundle};
use tracing::info;

fn run_pipeline(config: Config) -> Result<()> {
    let mut profiler = Profiler::new();

    profiler.start("setup")?;
    let executor = Executor::new(drift_parallel::ExecutorConfig {
        threads: config.parallel.threads,
    })?;
    let progress = ProgressReporter::new(config.report.show_progress, 5, "drift-eval");
    profiler.stop()?;

    profiler.start("synthesize")?;
    let reference = synthesize_reference(&config)?;
    let target = synthesize_target(&config)?;
    progress.handle().step("samples synthesized");
    profiler.stop()?;

    profiler.start("compute")?;
    let outcomes = executor.run(|| drift_math::run_all(&reference, &target, &config))?;
    progress.handle().step("metrics computed");
    profiler.stop()?;

    profiler.start("report")?;
    let bundle = ReportBundle::from_parts(&config, outcomes, profiler.snapshot());
    let written = persist_all(&bundle, &config)?;
    progress.handle().step("reports written");
    profiler.stop()?;

    progress.handle().set(5, "done");
    progress.handle().finish("drift-eval complete");

    info!(summary = %bundle.summary(), files = ?written, "pipeline finished");
    Ok(())
}

fn synthesize_reference(config: &Config) -> Result<DistributionSample> {
    let values: Vec<f64> = (0..512).map(|i| (i as f64).sin() + 0.5).collect();
    DistributionSample::new(ColumnName::new(config.reference_column.clone())?, values)
}

fn synthesize_target(config: &Config) -> Result<DistributionSample> {
    let values: Vec<f64> = (0..512).map(|i| (i as f64).cos() + 0.7).collect();
    DistributionSample::new(ColumnName::new(config.target_column.clone())?, values)
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let parsed = cli::Cli::parse_args();
    let config = config_loader::resolve(&parsed)?;
    run_pipeline(config)?;
    Ok(())
}
