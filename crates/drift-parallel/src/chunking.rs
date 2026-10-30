use drift_core::{DriftError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPlan {
    pub total: usize,
    pub chunk_size: usize,
    pub chunk_count: usize,
}

impl ChunkPlan {
    pub fn new(total: usize, chunk_size: usize) -> Result<Self> {
        if chunk_size == 0 {
            return Err(DriftError::InvalidChunkSize(0));
        }
        let chunk_count = if total == 0 {
            0
        } else {
            total.div_ceil(chunk_size)
        };
        Ok(Self {
            total,
            chunk_size,
            chunk_count,
        })
    }

    pub fn from_threads(total: usize, threads: usize) -> Result<Self> {
        if threads == 0 {
            return Err(DriftError::Invariant(
                "threads must be greater than zero".to_string(),
            ));
        }
        if total == 0 {
            return Self::new(0, 1);
        }
        let chunk_size = total.div_ceil(threads);
        Self::new(total, chunk_size.max(1))
    }

    pub fn ranges(&self) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::with_capacity(self.chunk_count);
        let mut start = 0usize;
        while start < self.total {
            let end = (start + self.chunk_size).min(self.total);
            ranges.push(start..end);
            start = end;
        }
        ranges
    }

    pub fn last_range(&self) -> Option<std::ops::Range<usize>> {
        self.ranges().into_iter().last()
    }
}

pub fn chunk_indices(total: usize, chunk_size: usize) -> Result<Vec<std::ops::Range<usize>>> {
    let plan = ChunkPlan::new(total, chunk_size)?;
    Ok(plan.ranges())
}

pub fn slice_chunks<'a, T>(values: &'a [T], plan: &ChunkPlan) -> Vec<&'a [T]> {
    plan.ranges()
        .into_iter()
        .filter_map(|range| values.get(range))
        .collect()
}
