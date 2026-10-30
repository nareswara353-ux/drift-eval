use std::path::Path;

use drift_core::{DriftError, Result};
use serde::Serialize;
use tera::{Context, Tera};

use crate::ReportBundle;

const FALLBACK_TEMPLATE: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>drift-eval report</title>
<style>
body{font-family:system-ui,sans-serif;margin:2rem;color:#111}
h1{font-size:1.5rem}
table{border-collapse:collapse;width:100%;margin-top:1rem}
th,td{border:1px solid #ddd;padding:.5rem;text-align:left;font-size:.9rem}
th{background:#f4f4f4}
.drifted{color:#b00020;font-weight:600}
.stable{color:#0a7c3a;font-weight:600}
.meta{color:#555;font-size:.85rem}
</style>
</head>
<body>
<h1>Distribution Drift Report</h1>
<p class="meta">Reference: {{ reference_column }} | Target: {{ target_column }} | Alpha: {{ alpha }}</p>
<p class="meta">{{ drifted_count }} of {{ total }} metrics drifted | Total time: {{ total_millis }} ms</p>
<table>
<thead><tr><th>Metric</th><th>Statistic</th><th>P-Value</th><th>Critical</th><th>Verdict</th><th>N(ref/tgt)</th></tr></thead>
<tbody>
{% for outcome in outcomes %}
<tr>
<td>{{ outcome.metric }}</td>
<td>{{ outcome.statistic }}</td>
<td>{% if outcome.p_value %}{{ outcome.p_value }}{% else %}-{% endif %}</td>
<td>{{ outcome.critical }}</td>
<td class="{% if outcome.verdict == 'drifted' %}drifted{% else %}stable{% endif %}">{{ outcome.verdict }}</td>
<td>{{ outcome.sample_size_reference }} / {{ outcome.sample_size_target }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</body>
</html>
"#;

#[derive(Serialize)]
struct OutcomeView {
    metric: String,
    statistic: f64,
    p_value: Option<f64>,
    critical: f64,
    verdict: String,
    sample_size_reference: usize,
    sample_size_target: usize,
}

#[derive(Serialize)]
struct TemplateContext {
    reference_column: String,
    target_column: String,
    alpha: f64,
    drifted_count: usize,
    total: usize,
    total_millis: f64,
    outcomes: Vec<OutcomeView>,
}

fn build_context(bundle: &ReportBundle) -> TemplateContext {
    TemplateContext {
        reference_column: bundle.reference_column.clone(),
        target_column: bundle.target_column.clone(),
        alpha: bundle.alpha,
        drifted_count: bundle.drifted_count(),
        total: bundle.outcomes.len(),
        total_millis: bundle.profiler.total_millis(),
        outcomes: bundle
            .outcomes
            .iter()
            .map(|o| OutcomeView {
                metric: o.metric.label().to_string(),
                statistic: o.statistic,
                p_value: o.p_value,
                critical: o.critical,
                verdict: match o.verdict {
                    drift_core::DriftVerdict::Drifted => "drifted".to_string(),
                    drift_core::DriftVerdict::Stable => "stable".to_string(),
                    drift_core::DriftVerdict::Inconclusive => "inconclusive".to_string(),
                },
                sample_size_reference: o.sample_size_reference,
                sample_size_target: o.sample_size_target,
            })
            .collect(),
    }
}

pub fn render_html_report(
    bundle: &ReportBundle,
    output: &Path,
    template_path: Option<&str>,
) -> Result<()> {
    let mut tera = Tera::default();
    match template_path {
        Some(path) => {
            tera.add_template_file(path, Some("report"))
                .map_err(|e| DriftError::Template(e.to_string()))?;
        }
        None => {
            tera.add_raw_template("report", FALLBACK_TEMPLATE)
                .map_err(|e| DriftError::Template(e.to_string()))?;
        }
    }

    let context = Context::from_serialize(build_context(bundle))
        .map_err(|e| DriftError::Template(e.to_string()))?;
    let rendered = tera
        .render("report", &context)
        .map_err(|e| DriftError::Template(e.to_string()))?;

    std::fs::write(output, rendered)
        .map_err(|e| DriftError::io(output.display().to_string(), e))?;
    Ok(())
}
