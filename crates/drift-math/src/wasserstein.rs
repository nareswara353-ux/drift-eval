use drift_core::{DistributionSample, DriftError, DriftMetric, DriftVerdict, Result, TestOutcome};

struct SortedPool<'a> {
    reference: &'a [f64],
    target: &'a [f64],
    n: f64,
    m: f64,
    i: usize,
    j: usize,
    prev_x: f64,
    prev_cdf_ref: f64,
    prev_cdf_tgt: f64,
}

impl<'a> SortedPool<'a> {
    fn new(reference: &'a [f64], target: &'a [f64]) -> Self {
        Self {
            reference,
            target,
            n: reference.len() as f64,
            m: target.len() as f64,
            i: 0,
            j: 0,
            prev_x: f64::NEG_INFINITY,
            prev_cdf_ref: 0.0,
            prev_cdf_tgt: 0.0,
        }
    }

    fn next_point(&mut self) -> Option<(f64, f64, f64)> {
        if self.i >= self.reference.len() && self.j >= self.target.len() {
            return None;
        }
        let (x, advance_ref, advance_tgt) =
            match (self.reference.get(self.i), self.target.get(self.j)) {
                (Some(a), Some(b)) => {
                    if a < b {
                        (*a, true, false)
                    } else if b < a {
                        (*b, false, true)
                    } else {
                        (*a, true, true)
                    }
                }
                (Some(a), None) => (*a, true, false),
                (None, Some(b)) => (*b, false, true),
                (None, None) => return None,
            };
        if advance_ref {
            self.i += 1;
        }
        if advance_tgt {
            self.j += 1;
        }
        Some((x, self.i as f64 / self.n, self.j as f64 / self.m))
    }
}

pub fn wasserstein1(
    reference: &DistributionSample,
    target: &DistributionSample,
    chunk_size: usize,
) -> Result<TestOutcome> {
    if chunk_size == 0 {
        return Err(DriftError::InvalidChunkSize(0));
    }
    if reference.is_empty() || target.is_empty() {
        return Err(DriftError::Invariant(
            "wasserstein requires non-empty samples".to_string(),
        ));
    }

    let mut ref_sorted = reference.values.clone();
    let mut tgt_sorted = target.values.clone();
    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    tgt_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut pool = SortedPool::new(&ref_sorted, &tgt_sorted);
    let mut distance = 0.0_f64;
    let mut processed = 0usize;

    while let Some((x, cdf_ref, cdf_tgt)) = pool.next_point() {
        if pool.prev_x.is_finite() && x.is_finite() {
            let dx = x - pool.prev_x;
            let mid_ref = 0.5 * (pool.prev_cdf_ref + cdf_ref);
            let mid_tgt = 0.5 * (pool.prev_cdf_tgt + cdf_tgt);
            distance += dx * (mid_ref - mid_tgt).abs();
        }
        pool.prev_x = x;
        pool.prev_cdf_ref = cdf_ref;
        pool.prev_cdf_tgt = cdf_tgt;
        processed += 1;
        if processed % chunk_size == 0 {
            std::hint::spin_loop();
        }
    }

    if !distance.is_finite() {
        return Err(DriftError::numerical(
            "wasserstein1",
            "distance became non-finite",
        ));
    }

    let n = reference.len();
    let m = target.len();
    let en = (n as f64 * m as f64 / (n as f64 + m as f64)).sqrt();
    let critical = 1.0 / en;
    let verdict = if distance > critical {
        DriftVerdict::Drifted
    } else {
        DriftVerdict::Stable
    };

    TestOutcome::new(
        DriftMetric::Wasserstein1,
        distance,
        None,
        critical,
        verdict,
        n,
        m,
    )
}
