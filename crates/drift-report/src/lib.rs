pub mod html;
pub mod json_export;
pub mod progress;
pub mod yaml_export;

use drift_core::{Config, ProfilerSnapshot, Result, TestOutcome};
use serde::{Deserialize, Serialize};

pub use html::render_html_report;
pub use json_export::{export_json, ReportJson};
pub use progress::{ProgressHandle, ProgressReporter};
pub use yaml_export::export_yaml;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportBundle {
    pub reference_column: String,
    pub target_column: String,
    pub outcomes: Vec<TestOutcome>,
    pub profiler: ProfilerSnapshot,
    pub alpha: f64,
}

impl ReportBundle {
    pub fn from_parts(
        config: &Config,
        outcomes: Vec<TestOutcome>,
        profiler: ProfilerSnapshot,
    ) -> Self {
        Self {
            reference_column: config.reference_column.clone(),
            target_column: config.target_column.clone(),
            outcomes,
            profiler,
            alpha: config.math.alpha,
        }
    }

    pub fn drifted_count(&self) -> usize {
        self.outcomes.iter().filter(|o| o.is_significant()).count()
    }

    pub fn summary(&self) -> String {
        format!(
            "{} of {} metrics indicated drift between '{}' and '{}'",
            self.drifted_count(),
            self.outcomes.len(),
            self.reference_column,
            self.target_column
        )
    }
}

pub fn persist_all(bundle: &ReportBundle, config: &Config) -> Result<Vec<String>> {
    let mut written = Vec::new();
    let dir = std::path::Path::new(&config.report.output_dir);
    std::fs::create_dir_all(dir)
        .map_err(|e| drift_core::DriftError::io(dir.display().to_string(), e))?;

    if config.report.emit_html {
        let path = dir.join("drift_report.html");
        render_html_report(bundle, &path, config.report.template_path.as_deref())?;
        written.push(path.display().to_string());
    }
    if config.report.emit_json {
        let path = dir.join("drift_report.json");
        export_json(bundle, &path)?;
        written.push(path.display().to_string());
    }
    if config.report.emit_yaml {
        let path = dir.join("drift_report.yaml");
        export_yaml(bundle, &path)?;
        written.push(path.display().to_string());
    }

    Ok(written)
}
