use drift_core::{DriftError, Result};

#[derive(Debug, Clone)]
pub struct EmpiricalCdf {
    sorted: Vec<f64>,
}

impl EmpiricalCdf {
    pub fn from_slice(values: &[f64]) -> Result<Self> {
        if values.is_empty() {
            return Err(DriftError::Invariant(
                "ecdf requires non-empty input".to_string(),
            ));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(DriftError::numerical(
                "ecdf",
                "encountered non-finite value",
            ));
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(Self { sorted })
    }

    pub fn len(&self) -> usize {
        self.sorted.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }

    pub fn as_sorted(&self) -> &[f64] {
        &self.sorted
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        if self.sorted.is_empty() {
            return 0.0;
        }
        let idx = self.sorted.partition_point(|value| *value <= x);
        idx as f64 / self.sorted.len() as f64
    }

    pub fn evaluate_batch(&self, queries: &[f64]) -> Vec<f64> {
        queries.iter().map(|q| self.evaluate(*q)).collect()
    }

    pub fn max_abs_diff(&self, other: &Self) -> f64 {
        let mut points = Vec::with_capacity(self.sorted.len() + other.sorted.len());
        points.extend_from_slice(&self.sorted);
        points.extend_from_slice(&other.sorted);
        points.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        points.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

        points
            .iter()
            .map(|x| (self.evaluate(*x) - other.evaluate(*x)).abs())
            .fold(0.0_f64, f64::max)
    }
}
