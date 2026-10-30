use serde::{Deserialize, Serialize};

use crate::error::{DriftError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub math: MathConfig,
    pub parallel: ParallelConfig,
    pub report: ReportConfig,
    pub reference_column: String,
    pub target_column: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            math: MathConfig::default(),
            parallel: ParallelConfig::default(),
            report: ReportConfig::default(),
            reference_column: "reference".to_string(),
            target_column: "target".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MathConfig {
    pub alpha: f64,
    pub enable_ks: bool,
    pub enable_wasserstein: bool,
    pub enable_chi_square: bool,
    pub enable_delta_correlation: bool,
    pub chi_square_bins: usize,
    pub wasserstein_chunk_size: usize,
}

impl Default for MathConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            enable_ks: true,
            enable_wasserstein: true,
            enable_chi_square: true,
            enable_delta_correlation: true,
            chi_square_bins: 32,
            wasserstein_chunk_size: 4096,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParallelConfig {
    pub threads: Option<usize>,
    pub chunk_size: usize,
    pub use_memory_map: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            threads: None,
            chunk_size: 16384,
            use_memory_map: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReportConfig {
    pub output_dir: String,
    pub emit_html: bool,
    pub emit_json: bool,
    pub emit_yaml: bool,
    pub show_progress: bool,
    pub template_path: Option<String>,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            output_dir: "reports".to_string(),
            emit_html: true,
            emit_json: true,
            emit_yaml: false,
            show_progress: true,
            template_path: None,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if !(self.math.alpha > 0.0 && self.math.alpha < 1.0) {
            return Err(DriftError::InvalidAlpha(self.math.alpha));
        }
        if self.math.chi_square_bins < 2 {
            return Err(DriftError::ConfigInvalid(
                "chi_square_bins must be at least 2".to_string(),
            ));
        }
        if self.math.wasserstein_chunk_size == 0 {
            return Err(DriftError::InvalidChunkSize(0));
        }
        if self.parallel.chunk_size == 0 {
            return Err(DriftError::InvalidChunkSize(0));
        }
        if self.reference_column.trim().is_empty() {
            return Err(DriftError::ConfigInvalid(
                "reference_column cannot be empty".to_string(),
            ));
        }
        if self.target_column.trim().is_empty() {
            return Err(DriftError::ConfigInvalid(
                "target_column cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    pub fn from_toml_str(input: &str) -> Result<Self> {
        let config: Self =
            toml::from_str(input).map_err(|e| DriftError::ConfigParse(format!("toml: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_yaml_str(input: &str) -> Result<Self> {
        let config: Self = serde_yaml::from_str(input)
            .map_err(|e| DriftError::ConfigParse(format!("yaml: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn to_toml_string(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|e| DriftError::Serialize(format!("toml: {e}")))
    }

    pub fn to_yaml_string(&self) -> Result<String> {
        serde_yaml::to_string(self).map_err(|e| DriftError::Serialize(format!("yaml: {e}")))
    }
}
