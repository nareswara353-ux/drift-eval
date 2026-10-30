use std::path::Path;

use drift_core::{DriftError, Result};

use crate::json_export::ReportJson;
use crate::ReportBundle;

pub fn export_yaml(bundle: &ReportBundle, output: &Path) -> Result<()> {
    let report = ReportJson::from_bundle(bundle);
    let serialized =
        serde_yaml::to_string(&report).map_err(|e| DriftError::Serialize(format!("yaml: {e}")))?;
    std::fs::write(output, serialized)
        .map_err(|e| DriftError::io(output.display().to_string(), e))?;
    Ok(())
}

pub fn export_yaml_string(bundle: &ReportBundle) -> Result<String> {
    let report = ReportJson::from_bundle(bundle);
    serde_yaml::to_string(&report).map_err(|e| DriftError::Serialize(format!("yaml: {e}")))
}
