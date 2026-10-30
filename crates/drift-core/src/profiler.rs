use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{DriftError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProfilerSnapshot {
    pub phases: BTreeMap<String, u64>,
    pub total_nanos: u64,
}

impl ProfilerSnapshot {
    pub fn phase_millis(&self, name: &str) -> Option<f64> {
        self.phases
            .get(name)
            .map(|nanos| *nanos as f64 / 1_000_000.0)
    }

    pub fn total_millis(&self) -> f64 {
        self.total_nanos as f64 / 1_000_000.0
    }
}

#[derive(Debug)]
pub struct Profiler {
    started_at: Instant,
    current: Option<(String, Instant)>,
    phases: BTreeMap<String, Duration>,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            started_at: Instant::now(),
            current: None,
            phases: BTreeMap::new(),
        }
    }

    pub fn start(&mut self, phase: impl Into<String>) -> Result<()> {
        if self.current.is_some() {
            return Err(DriftError::Invariant(
                "a phase is already running".to_string(),
            ));
        }
        self.current = Some((phase.into(), Instant::now()));
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        let (name, start) = self
            .current
            .take()
            .ok_or_else(|| DriftError::Invariant("no phase is running".to_string()))?;
        let elapsed = start.elapsed();
        self.phases
            .entry(name)
            .and_modify(|existing| *existing += elapsed)
            .or_insert(elapsed);
        Ok(())
    }

    pub fn measure<T, F>(&mut self, phase: impl Into<String>, f: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        self.start(phase)?;
        let outcome = f();
        self.stop()?;
        outcome
    }

    pub fn snapshot(&self) -> ProfilerSnapshot {
        let phases = self
            .phases
            .iter()
            .map(|(k, v)| (k.clone(), v.as_nanos() as u64))
            .collect();
        ProfilerSnapshot {
            phases,
            total_nanos: self.started_at.elapsed().as_nanos() as u64,
        }
    }
}
