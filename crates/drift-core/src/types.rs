use serde::{Deserialize, Serialize};

use crate::error::{DriftError, Result};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColumnName(String);

impl ColumnName {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(DriftError::Invariant(
                "column name cannot be empty".to_string(),
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl std::fmt::Display for ColumnName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DistributionSample {
    pub column: ColumnName,
    pub values: Vec<f64>,
}

impl DistributionSample {
    pub fn new(column: ColumnName, values: Vec<f64>) -> Result<Self> {
        if values.is_empty() {
            return Err(DriftError::EmptySample(column.as_str().to_string()));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(DriftError::NonFiniteValue(column.as_str().to_string()));
        }
        Ok(Self { column, values })
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftMetric {
    KolmogorovSmirnov,
    Wasserstein1,
    ChiSquare,
    DeltaCorrelation,
}

impl DriftMetric {
    pub fn label(&self) -> &'static str {
        match self {
            Self::KolmogorovSmirnov => "kolmogorov-smirnov",
            Self::Wasserstein1 => "wasserstein-1",
            Self::ChiSquare => "chi-square",
            Self::DeltaCorrelation => "delta-correlation",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftVerdict {
    Stable,
    Drifted,
    Inconclusive,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Threshold {
    pub metric: DriftMetric,
    pub critical: f64,
}

impl Threshold {
    pub fn new(metric: DriftMetric, critical: f64) -> Result<Self> {
        if !critical.is_finite() || critical < 0.0 {
            return Err(DriftError::Invariant(format!(
                "threshold for {} must be finite and non-negative",
                metric.label()
            )));
        }
        Ok(Self { metric, critical })
    }

    pub fn evaluate(&self, statistic: f64) -> DriftVerdict {
        if statistic > self.critical {
            DriftVerdict::Drifted
        } else {
            DriftVerdict::Stable
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestOutcome {
    pub metric: DriftMetric,
    pub statistic: f64,
    pub p_value: Option<f64>,
    pub critical: f64,
    pub verdict: DriftVerdict,
    pub sample_size_reference: usize,
    pub sample_size_target: usize,
}

impl TestOutcome {
    pub fn new(
        metric: DriftMetric,
        statistic: f64,
        p_value: Option<f64>,
        critical: f64,
        verdict: DriftVerdict,
        sample_size_reference: usize,
        sample_size_target: usize,
    ) -> Result<Self> {
        if !statistic.is_finite() {
            return Err(DriftError::numerical(
                metric.label(),
                "statistic is non-finite",
            ));
        }
        Ok(Self {
            metric,
            statistic,
            p_value,
            critical,
            verdict,
            sample_size_reference,
            sample_size_target,
        })
    }

    pub fn is_significant(&self) -> bool {
        self.verdict == DriftVerdict::Drifted
    }
}
