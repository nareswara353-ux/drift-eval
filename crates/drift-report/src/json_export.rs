use std::path::Path;

use drift_core::{DriftError, DriftVerdict, Result, TestOutcome};
use serde::{Deserialize, Serialize};

use crate::ReportBundle;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeJson {
    pub metric: String,
    pub statistic: f64,
    pub p_value: Option<f64>,
    pub critical: f64,
    pub verdict: String,
    pub sample_size_reference: usize,
    pub sample_size_target: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportJson {
    pub reference_column: String,
    pub target_column: String,
    pub alpha: f64,
    pub drifted_count: usize,
    pub total_metrics: usize,
    pub total_millis: f64,
    pub phases_nanos: std::collections::BTreeMap<String, u64>,
    pub outcomes: Vec<OutcomeJson>,
}

fn verdict_label(verdict: DriftVerdict) -> &'static str {
    match verdict {
        DriftVerdict::Stable => "stable",
        DriftVerdict::Drifted => "drifted",
        DriftVerdict::Inconclusive => "inconclusive",
    }
}

fn to_outcome_json(outcome: &TestOutcome) -> OutcomeJson {
    OutcomeJson {
        metric: outcome.metric.label().to_string(),
        statistic: outcome.statistic,
        p_value: outcome.p_value,
        critical: outcome.critical,
        verdict: verdict_label(outcome.verdict).to_string(),
        sample_size_reference: outcome.sample_size_reference,
        sample_size_target: outcome.sample_size_target,
    }
}

impl ReportJson {
    pub fn from_bundle(bundle: &ReportBundle) -> Self {
        Self {
            reference_column: bundle.reference_column.clone(),
            target_column: bundle.target_column.clone(),
            alpha: bundle.alpha,
            drifted_count: bundle.drifted_count(),
            total_metrics: bundle.outcomes.len(),
            total_millis: bundle.profiler.total_millis(),
            phases_nanos: bundle.profiler.phases.clone(),
            outcomes: bundle.outcomes.iter().map(to_outcome_json).collect(),
        }
    }
}

pub fn export_json(bundle: &ReportBundle, output: &Path) -> Result<()> {
    let report = ReportJson::from_bundle(bundle);
    let serialized = serde_json::to_string_pretty(&report)
        .map_err(|e| DriftError::Serialize(format!("json: {e}")))?;
    std::fs::write(output, serialized)
        .map_err(|e| DriftError::io(output.display().to_string(), e))?;
    Ok(())
}

pub fn export_json_string(bundle: &ReportBundle) -> Result<String> {
    let report = ReportJson::from_bundle(bundle);
    serde_json::to_string_pretty(&report).map_err(|e| DriftError::Serialize(format!("json: {e}")))
}
