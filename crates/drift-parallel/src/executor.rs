use drift_core::{DriftError, Result};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

#[derive(Debug, Clone, Copy, Default)]
pub struct ExecutorConfig {
    pub threads: Option<usize>,
}

pub struct Executor {
    pool: ThreadPool,
}

impl Executor {
    pub fn new(config: ExecutorConfig) -> Result<Self> {
        let mut builder = ThreadPoolBuilder::new();
        if let Some(threads) = config.threads {
            if threads == 0 {
                return Err(DriftError::Invariant(
                    "executor thread count must be greater than zero".to_string(),
                ));
            }
            builder = builder.num_threads(threads);
        }
        let pool = builder
            .build()
            .map_err(|e| DriftError::Invariant(format!("rayon pool failure: {e}")))?;
        Ok(Self { pool })
    }

    pub fn default_pool() -> Result<Self> {
        Self::new(ExecutorConfig::default())
    }

    pub fn thread_count(&self) -> usize {
        self.pool.current_num_threads()
    }

    pub fn run<T, F>(&self, f: F) -> Result<T>
    where
        T: Send,
        F: FnOnce() -> Result<T> + Send,
    {
        self.pool.install(f)
    }

    pub fn map<T, U, F>(&self, items: &[T], f: F) -> Vec<U>
    where
        T: Sync,
        U: Send,
        F: Fn(&T) -> U + Sync + Send,
    {
        self.pool.install(|| items.par_iter().map(f).collect())
    }

    pub fn try_map<T, U, F>(&self, items: &[T], f: F) -> Result<Vec<U>>
    where
        T: Sync,
        U: Send,
        F: Fn(&T) -> Result<U> + Sync + Send,
    {
        self.pool.install(|| items.par_iter().map(f).collect())
    }
}
