use drift_core::{ColumnName, Config, DistributionSample};
use drift_math::run_all;

fn build(name: &str, shift: f64) -> DistributionSample {
    let column = ColumnName::new(name).expect("column");
    let values: Vec<f64> = (0..256)
        .map(|i| (i as f64 * 0.1).sin() + shift)
        .collect();
    DistributionSample::new(column, values).expect("sample")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::default();
    let reference = build("reference", 0.0);
    let target = build("target", 0.3);

    let outcomes = run_all(&reference, &target, &config)?;

    println!("reference={} target={}", reference.column, target.column);
    for outcome in &outcomes {
        println!(
            "{:<22} stat={:.6} p={:?} critical={:.6} verdict={:?}",
            outcome.metric.label(),
            outcome.statistic,
            outcome.p_value,
            outcome.critical,
            outcome.verdict,
        );
    }

    let drifted = outcomes.iter().filter(|o| o.is_significant()).count();
    println!("drifted={}/{}", drifted, outcomes.len());
    Ok(())
}
