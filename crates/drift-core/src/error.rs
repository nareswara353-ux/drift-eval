use thiserror::Error;

pub type Result<T> = std::result::Result<T, DriftError>;

#[derive(Debug, Error)]
pub enum DriftError {
    #[error("io failure at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("configuration parse failure: {0}")]
    ConfigParse(String),

    #[error("configuration validation failure: {0}")]
    ConfigInvalid(String),

    #[error("serialization failure: {0}")]
    Serialize(String),

    #[error("deserialization failure: {0}")]
    Deserialize(String),

    #[error("empty sample for column '{0}'")]
    EmptySample(String),

    #[error("non-finite value encountered in column '{0}'")]
    NonFiniteValue(String),

    #[error("sample size mismatch: reference={reference}, target={target}")]
    SampleSizeMismatch { reference: usize, target: usize },

    #[error("numerical failure in {context}: {detail}")]
    Numerical { context: String, detail: String },

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("invalid alpha {0}: must be in (0, 1)")]
    InvalidAlpha(f64),

    #[error("invalid chunk size {0}: must be greater than zero")]
    InvalidChunkSize(usize),

    #[error("report template failure: {0}")]
    Template(String),

    #[error("memory mapping failure for {path}: {source}")]
    MemoryMap {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("unsupported metric: {0}")]
    UnsupportedMetric(String),

    #[error("profiler not initialized")]
    ProfilerUninitialized,

    #[error("python binding failure: {0}")]
    PythonBinding(String),

    #[error("internal invariant violation: {0}")]
    Invariant(String),
}

impl DriftError {
    pub fn io(path: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub fn numerical(context: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Numerical {
            context: context.into(),
            detail: detail.into(),
        }
    }
}
