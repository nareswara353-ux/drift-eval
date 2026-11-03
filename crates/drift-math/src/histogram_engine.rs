//! Histogram engine: Algorithms 1-3 of the paper.
//!
//! * Algorithm 1 - [`parallel_histogram`]: chunked parallel eCDF builder over borrowed slices.
//! * Algorithm 2 - [`StreamingEvaluator`] / [`certify`]: streaming KS and W1 with interval certificates.
//! * Algorithm 3 - [`parallel_comoments`] and [`drift_report`]: delta correlation and drift decisions.
//!
//! Guarantees hold for real arithmetic. The KS interval is formed from exact integers and rounded
//! outward; the W1 interval uses floating-point sums and is validated empirically, not proved,
//! against rounding (see the "Numerical validity" remark in the paper).
use drift_core::{DriftError, Result};
use rayon::prelude::*;
use statrs::function::erf::erfc;

use crate::chi_square::{chi_square_counts, ChiSquareOutcome};
use crate::ks_test::kolmogorov_sf;
use crate::util::{next_down, next_up, par_sorted_copy, CompSum};

// ---------------------------------------------------------------------------------------------
// Edges
// ---------------------------------------------------------------------------------------------

/// Strictly increasing edges t_0 < ... < t_B defining bins
/// I_0 = (-inf, t_0], I_k = (t_{k-1}, t_k] (1 <= k <= B), I_{B+1} = (t_B, +inf).
#[derive(Debug, Clone)]
pub struct Edges {
    t: Vec<f64>,
}

impl Edges {
    pub fn new(t: Vec<f64>) -> Result<Self> {
        if t.len() < 2 {
            return Err(DriftError::Invariant("edges need at least two points".into()));
        }
        if t.iter().any(|v| !v.is_finite()) || t.windows(2).any(|w| !(w[0] < w[1])) {
            return Err(DriftError::Invariant(
                "edges must be finite and strictly increasing".into(),
            ));
        }
        Ok(Self { t })
    }

    /// B uniform bins on [min, max].
    pub fn uniform(min: f64, max: f64, bins: usize) -> Result<Self> {
        if bins < 1 || !(min < max) {
            return Err(DriftError::Invariant("uniform edges need bins >= 1 and min < max".into()));
        }
        let mut t: Vec<f64> = (0..=bins)
            .map(|k| min + (max - min) * (k as f64) / (bins as f64))
            .collect();
        t[0] = min;
        t[bins] = max;
        Self::new(t)
    }

    /// Reference-quantile edges. One-time reference preparation: copies and sorts the finite
    /// reference values (O(n) memory, O(n log n) time), amortised over all later evaluations.
    /// t_0 = min and t_B = max of the reference; duplicate quantiles are removed, so the
    /// resulting number of bins may be smaller than `bins` when the data contain ties.
    pub fn from_reference_quantiles(reference: &[f64], bins: usize) -> Result<Self> {
        if bins < 1 {
            return Err(DriftError::Invariant("bins must be >= 1".into()));
        }
        let finite: Vec<f64> = reference.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return Err(DriftError::Invariant("reference has no finite values".into()));
        }
        let sorted = par_sorted_copy(&finite);
        let n = sorted.len();
        let mut t = Vec::with_capacity(bins + 1);
        t.push(sorted[0]);
        for k in 1..=bins {
            let idx = ((k * n + bins - 1) / bins).saturating_sub(1).min(n - 1);
            t.push(sorted[idx]);
        }
        t.dedup();
        if t.len() < 2 {
            return Err(DriftError::Invariant("reference is constant".into()));
        }
        Self::new(t)
    }

    /// Number of interior bins B (the histogram has B + 2 counters).
    pub fn bins(&self) -> usize {
        self.t.len() - 1
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.t
    }

    /// Number of edges strictly smaller than `v`; equals the bin index.
    #[inline]
    pub fn bin_index(&self, v: f64) -> usize {
        self.t.partition_point(|e| *e < v)
    }
}

// ---------------------------------------------------------------------------------------------
// Algorithm 1
// ---------------------------------------------------------------------------------------------

/// Mergeable per-worker accumulator: B+2 integer counts, two compensated tail sums, and the
/// number of non-finite (missing) values. Memory is independent of the number of rows.
#[derive(Debug, Clone)]
pub struct Histogram {
    pub counts: Vec<u64>,
    pub tail_lo: CompSum,
    pub tail_hi: CompSum,
    pub missing: u64,
}

impl Histogram {
    pub fn new(bins: usize) -> Self {
        Self {
            counts: vec![0; bins + 2],
            tail_lo: CompSum::default(),
            tail_hi: CompSum::default(),
            missing: 0,
        }
    }

    /// Adds the values of one borrowed slice. Non-finite values (NaN, +/-inf) count as missing.
    pub fn absorb(&mut self, edges: &Edges, chunk: &[f64]) {
        let b = edges.bins();
        let (t0, tb) = (edges.t[0], edges.t[b]);
        for &v in chunk {
            if !v.is_finite() {
                self.missing += 1;
                continue;
            }
            let k = edges.bin_index(v);
            self.counts[k] += 1;
            if k == 0 {
                self.tail_lo.add(t0 - v);
            } else if k == b + 1 {
                self.tail_hi.add(v - tb);
            }
        }
    }

    pub fn merge(&mut self, other: &Histogram) -> Result<()> {
        if self.counts.len() != other.counts.len() {
            return Err(DriftError::DimensionMismatch {
                expected: self.counts.len(),
                actual: other.counts.len(),
            });
        }
        for (a, b) in self.counts.iter_mut().zip(other.counts.iter()) {
            *a += *b;
        }
        self.tail_lo.merge(&other.tail_lo);
        self.tail_hi.merge(&other.tail_hi);
        self.missing += other.missing;
        Ok(())
    }

    pub fn nonmissing(&self) -> u64 {
        self.counts.iter().sum()
    }

    pub fn total(&self) -> u64 {
        self.nonmissing() + self.missing
    }
}

fn tree_merge(mut parts: Vec<Histogram>) -> Result<Histogram> {
    while parts.len() > 1 {
        let mut next = Vec::with_capacity((parts.len() + 1) / 2);
        let mut it = parts.into_iter();
        while let Some(mut a) = it.next() {
            if let Some(b) = it.next() {
                a.merge(&b)?;
            }
            next.push(a);
        }
        parts = next;
    }
    parts
        .pop()
        .ok_or_else(|| DriftError::Invariant("no partial histograms".into()))
}

/// Algorithm 1. `workers` logical workers (one coarse Rayon task each, not pinned threads);
/// chunk p of `chunk_size` rows is assigned to worker `p % workers` and processed in increasing
/// order, each worker owns exactly one `Histogram`, and the `workers` results are combined with a
/// fixed pairwise tree. Counts are identical for every (chunk_size, workers); the tail sums are
/// reproducible for fixed (chunk_size, workers) on the same floating-point semantics.
/// `values` is borrowed (for example from a memory map); nothing is copied.
pub fn parallel_histogram(
    values: &[f64],
    edges: &Edges,
    chunk_size: usize,
    workers: usize,
) -> Result<Histogram> {
    if chunk_size == 0 {
        return Err(DriftError::InvalidChunkSize(0));
    }
    if workers == 0 {
        return Err(DriftError::ConfigInvalid("workers must be >= 1".into()));
    }
    let len = values.len();
    let p = (len + chunk_size - 1) / chunk_size;
    let bins = edges.bins();
    let parts: Vec<Histogram> = (0..workers)
        .into_par_iter()
        .map(|w| {
            let mut h = Histogram::new(bins);
            let mut idx = w;
            while idx < p {
                let s = idx * chunk_size;
                let e = (s + chunk_size).min(len);
                h.absorb(edges, &values[s..e]);
                idx += workers;
            }
            h
        })
        .collect();
    tree_merge(parts)
}

// ---------------------------------------------------------------------------------------------
// Moments (Algorithm 3, Chan merge). Listwise deletion only; pairwise deletion is NOT implemented.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CoMoments {
    pub d: usize,
    pub n: u64,
    pub dropped: u64,
    pub mean: Vec<f64>,
    /// Row-major d x d co-moment matrix sum (x - mean)(x - mean)^T.
    pub m2: Vec<f64>,
}

impl CoMoments {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            n: 0,
            dropped: 0,
            mean: vec![0.0; d],
            m2: vec![0.0; d * d],
        }
    }

    pub fn update_row(&mut self, x: &[f64]) {
        let d = self.d;
        self.n += 1;
        let nf = self.n as f64;
        let delta: Vec<f64> = (0..d).map(|i| x[i] - self.mean[i]).collect();
        for i in 0..d {
            self.mean[i] += delta[i] / nf;
        }
        for i in 0..d {
            let d2i = x[i] - self.mean[i];
            for j in 0..d {
                self.m2[j * d + i] += delta[j] * d2i;
            }
        }
    }

    /// Eq. (merge) of the paper.
    pub fn merge(&mut self, other: &CoMoments) {
        self.dropped += other.dropped;
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            self.n = other.n;
            self.mean = other.mean.clone();
            self.m2 = other.m2.clone();
            return;
        }
        let d = self.d;
        let (na, nb) = (self.n as f64, other.n as f64);
        let n = na + nb;
        let delta: Vec<f64> = (0..d).map(|i| other.mean[i] - self.mean[i]).collect();
        for i in 0..d {
            for j in 0..d {
                self.m2[i * d + j] += other.m2[i * d + j] + delta[i] * delta[j] * na * nb / n;
            }
        }
        for i in 0..d {
            self.mean[i] += delta[i] * nb / n;
        }
        self.n += other.n;
    }

    /// Sample standard deviation of feature i (0 if n < 2).
    pub fn std(&self, i: usize) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            (self.m2[i * self.d + i] / (self.n as f64 - 1.0)).sqrt()
        }
    }

    /// Pearson correlation; None if n < 2 or either variance is exactly zero.
    pub fn correlation(&self, i: usize, j: usize) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let (mii, mjj) = (self.m2[i * self.d + i], self.m2[j * self.d + j]);
        if mii <= 0.0 || mjj <= 0.0 {
            return None;
        }
        Some((self.m2[i * self.d + j] / (mii * mjj).sqrt()).clamp(-1.0, 1.0))
    }
}

/// One pass over borrowed columns with the same static worker assignment as Algorithm 1.
/// Rows with any non-finite value are dropped (listwise deletion) and counted in `dropped`.
pub fn parallel_comoments(
    columns: &[&[f64]],
    chunk_size: usize,
    workers: usize,
) -> Result<CoMoments> {
    if chunk_size == 0 {
        return Err(DriftError::InvalidChunkSize(0));
    }
    if workers == 0 || columns.is_empty() {
        return Err(DriftError::ConfigInvalid("workers and columns must be >= 1".into()));
    }
    let d = columns.len();
    let rows = columns[0].len();
    for c in columns {
        if c.len() != rows {
            return Err(DriftError::DimensionMismatch {
                expected: rows,
                actual: c.len(),
            });
        }
    }
    let p = (rows + chunk_size - 1) / chunk_size;
    let parts: Vec<CoMoments> = (0..workers)
        .into_par_iter()
        .map(|w| {
            let mut acc = CoMoments::new(d);
            let mut buf = vec![0.0_f64; d];
            let mut idx = w;
            while idx < p {
                let s = idx * chunk_size;
                let e = (s + chunk_size).min(rows);
                for r in s..e {
                    let mut ok = true;
                    for c in 0..d {
                        buf[c] = columns[c][r];
                        ok &= buf[c].is_finite();
                    }
                    if ok {
                        acc.update_row(&buf);
                    } else {
                        acc.dropped += 1;
                    }
                }
                idx += workers;
            }
            acc
        })
        .collect();
    let mut parts = parts.into_iter();
    let mut total = parts.next().expect("workers >= 1");
    // Fixed left-to-right merge over worker results (deterministic for fixed chunk_size, workers).
    for part in parts {
        total.merge(&part);
    }
    Ok(total)
}

// ---------------------------------------------------------------------------------------------
// Algorithm 2
// ---------------------------------------------------------------------------------------------

/// Reference state, built once and reused for every evaluation batch.
#[derive(Debug, Clone)]
pub struct ReferenceSummary {
    pub edges: Edges,
    pub hist: Histogram,
    pub n: u64,
    pub mean: f64,
    pub std: f64,
}

impl ReferenceSummary {
    /// Requires that every non-missing reference value lies in [t_0, t_B]; this is what makes the
    /// tail terms of the W1 certificate exact. Edges from `Edges::from_reference_quantiles`
    /// satisfy it by construction.
    pub fn from_slice(values: &[f64], edges: Edges, chunk_size: usize, workers: usize) -> Result<Self> {
        let hist = parallel_histogram(values, &edges, chunk_size, workers)?;
        let n = hist.nonmissing();
        if n == 0 {
            return Err(DriftError::Invariant("reference has no finite values".into()));
        }
        if hist.counts[edges.bins() + 1] != 0 || hist.tail_lo.value() != 0.0 {
            return Err(DriftError::Invariant(
                "edges must cover the reference range [min, max]".into(),
            ));
        }
        let mom = parallel_comoments(&[values], chunk_size, workers)?;
        Ok(Self {
            edges,
            hist,
            n,
            mean: mom.mean[0],
            std: mom.std(0),
        })
    }
}

/// Interval certificate. Intervals contain the exact sort-based values in exact arithmetic.
#[derive(Debug, Clone)]
pub struct Certificate {
    pub n: u64,
    pub m: u64,
    pub bins: usize,
    pub ks_lo: f64,
    pub ks_hi: f64,
    pub delta_b: f64,
    pub p_lo: f64,
    pub p_hi: f64,
    pub w1_hat: f64,
    pub w1_eps: f64,
    pub w1_lo: f64,
    pub w1_hi: f64,
}

/// Absolute slack added to the p-value interval to cover the truncation of the series for Q.
const P_SLACK: f64 = 1.0e-12;

/// Computes D_B, delta_B, W1_hat, eps_W and the p-value interval from two histograms.
pub fn certify(reference: &ReferenceSummary, eval: &Histogram) -> Result<Certificate> {
    let edges = &reference.edges;
    let b = edges.bins();
    if eval.counts.len() != b + 2 {
        return Err(DriftError::DimensionMismatch {
            expected: b + 2,
            actual: eval.counts.len(),
        });
    }
    let n = reference.n;
    let m = eval.nonmissing();
    if n == 0 || m == 0 {
        return Err(DriftError::Invariant("certificate requires non-empty samples".into()));
    }
    let (nn, mm) = (n as u128, m as u128);
    let nm_f = n as f64 * m as f64;

    // Integer numerators of Delta_k * n * m, k = 0..=B.
    let mut cr: u128 = 0;
    let mut ce: u128 = 0;
    let mut delta_num: Vec<i128> = Vec::with_capacity(b + 1);
    let mut dmax: u128 = 0;
    for k in 0..=b {
        cr += reference.hist.counts[k] as u128;
        ce += eval.counts[k] as u128;
        let v = (cr * mm) as i128 - (ce * nn) as i128;
        dmax = dmax.max(v.unsigned_abs());
        delta_num.push(v);
    }
    // delta_B numerator: max_k min(p_k, q_k) * n * m, k = 0..=B+1.
    let mut dm: u128 = 0;
    for k in 0..=(b + 1) {
        let pk = reference.hist.counts[k] as u128 * mm;
        let qk = eval.counts[k] as u128 * nn;
        dm = dm.max(pk.min(qk));
    }

    let down = |x: u128| next_down(next_down(x as f64 / nm_f)).max(0.0);
    let up = |x: u128| next_up(next_up(x as f64 / nm_f)).min(1.0);
    let ks_lo = down(dmax);
    let ks_hi = up(dmax + dm);
    let delta_b = dm as f64 / nm_f;

    let n_e = (n as f64 * m as f64 / (n as f64 + m as f64)).sqrt();
    let p_hi = (kolmogorov_sf(n_e * ks_lo) + P_SLACK).min(1.0);
    let p_lo = (kolmogorov_sf(n_e * ks_hi) - P_SLACK).max(0.0);

    // W1 certificate (floating point).
    let t = edges.as_slice();
    let mut interior = CompSum::default();
    let mut eps = CompSum::default();
    for k in 1..=b {
        let w = t[k] - t[k - 1];
        let a0 = (delta_num[k - 1].unsigned_abs() as f64) / nm_f;
        let a1 = (delta_num[k].unsigned_abs() as f64) / nm_f;
        interior.add(w * 0.5 * (a0 + a1));
        let pk = reference.hist.counts[k] as u128 * mm;
        let qk = eval.counts[k] as u128 * nn;
        eps.add(w * (pk.max(qk) as f64 / nm_f));
    }
    let w1_hat = interior.value() + (eval.tail_lo.value() + eval.tail_hi.value()) / m as f64;
    let w1_eps = eps.value();

    Ok(Certificate {
        n,
        m,
        bins: b,
        ks_lo,
        ks_hi,
        delta_b,
        p_lo,
        p_hi,
        w1_hat,
        w1_eps,
        w1_lo: (w1_hat - w1_eps).max(0.0),
        w1_hi: w1_hat + w1_eps,
    })
}

/// Streaming KS / W1 engine: absorb evaluation chunks as they arrive, certify at any time.
pub struct StreamingEvaluator<'a> {
    reference: &'a ReferenceSummary,
    hist: Histogram,
}

impl<'a> StreamingEvaluator<'a> {
    pub fn new(reference: &'a ReferenceSummary) -> Self {
        Self {
            reference,
            hist: Histogram::new(reference.edges.bins()),
        }
    }

    /// Sequential absorb of one chunk.
    pub fn update(&mut self, chunk: &[f64]) {
        self.hist.absorb(&self.reference.edges, chunk);
    }

    /// Parallel absorb of a large borrowed slice (Algorithm 1 on the slice, then merge).
    pub fn update_parallel(&mut self, values: &[f64], chunk_size: usize, workers: usize) -> Result<()> {
        let h = parallel_histogram(values, &self.reference.edges, chunk_size, workers)?;
        self.hist.merge(&h)
    }

    pub fn histogram(&self) -> &Histogram {
        &self.hist
    }

    pub fn certificate(&self) -> Result<Certificate> {
        certify(self.reference, &self.hist)
    }
}

// ---------------------------------------------------------------------------------------------
// Algorithm 3: decisions
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    Ks,
    W1Normalized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Label {
    Drift,
    NoActionableDrift,
    Indeterminate,
    Invalid,
}

#[derive(Debug, Clone)]
pub struct EngineParams {
    pub alpha_m: f64,
    pub alpha_r: f64,
    pub gate: Gate,
    pub tau_g: f64,
    pub tau_r: f64,
    pub tau_v: f64,
    pub tau_pi: f64,
    pub bins: usize,
    pub chunk_size: usize,
    pub workers: usize,
    pub correlation: bool,
}

impl EngineParams {
    pub fn validate(&self) -> Result<()> {
        for a in [self.alpha_m, self.alpha_r] {
            if !(a > 0.0 && a < 1.0) {
                return Err(DriftError::InvalidAlpha(a));
            }
        }
        if [self.tau_g, self.tau_r, self.tau_v, self.tau_pi].iter().any(|t| !(*t >= 0.0)) {
            return Err(DriftError::ConfigInvalid("thresholds must be >= 0".into()));
        }
        if self.bins < 2 || self.workers == 0 {
            return Err(DriftError::ConfigInvalid("bins >= 2 and workers >= 1 required".into()));
        }
        if self.chunk_size == 0 {
            return Err(DriftError::InvalidChunkSize(0));
        }
        Ok(())
    }
}

pub struct ContinuousInput<'a> {
    pub name: &'a str,
    pub reference: &'a [f64],
    pub target: &'a [f64],
}

pub struct CategoricalInput<'a> {
    pub name: &'a str,
    pub reference_counts: &'a [u64],
    pub target_counts: &'a [u64],
}

#[derive(Debug, Clone)]
pub struct FeatureReport {
    pub name: String,
    pub label: Label,
    /// True when the label comes from the constant-feature rules rather than from a test.
    pub degenerate: bool,
    pub certificate: Option<Certificate>,
    pub effect_interval: Option<(f64, f64)>,
    pub cohens_d: Option<f64>,
    pub missing_rate_ref: f64,
    pub missing_rate_tgt: f64,
    pub missingness_p: Option<f64>,
    pub missingness_drift: bool,
}

#[derive(Debug, Clone)]
pub struct CategoricalReport {
    pub name: String,
    pub label: Label,
    pub p_value: Option<f64>,
    pub cramers_v: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct PairReport {
    pub i: usize,
    pub j: usize,
    pub r_ref: Option<f64>,
    pub r_tgt: Option<f64>,
    pub delta: Option<f64>,
    pub p_value: Option<f64>,
    pub label: Label,
}

#[derive(Debug, Clone)]
pub struct DriftReport {
    pub features: Vec<FeatureReport>,
    pub categorical: Vec<CategoricalReport>,
    pub pairs: Vec<PairReport>,
    pub dropped_ref: u64,
    pub dropped_tgt: u64,
}

/// Holm step-down: returns the rejection flags at familywise level `alpha`.
/// The rejection set is monotone in the p-values, which is what lets the certified p-value
/// interval [p_lo, p_hi] be pushed through the procedure.
pub fn holm_reject(p: &[f64], alpha: f64) -> Vec<bool> {
    let m = p.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut rejected = vec![false; m];
    for (rank, &i) in order.iter().enumerate() {
        if p[i] <= alpha / (m - rank) as f64 {
            rejected[i] = true;
        } else {
            break;
        }
    }
    rejected
}

/// Two-proportion z-test p-value for missing rates x1/n1 versus x2/n2 (pooled variance).
pub fn two_proportion_p(x1: u64, n1: u64, x2: u64, n2: u64) -> f64 {
    if n1 == 0 || n2 == 0 {
        return 1.0;
    }
    let pooled = (x1 + x2) as f64 / (n1 + n2) as f64;
    if pooled <= 0.0 || pooled >= 1.0 {
        return 1.0;
    }
    let se = (pooled * (1.0 - pooled) * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
    let z = (x1 as f64 / n1 as f64 - x2 as f64 / n2 as f64) / se;
    erfc(z.abs() / std::f64::consts::SQRT_2).clamp(0.0, 1.0)
}

fn finite_min_max(values: &[f64]) -> Option<(f64, f64)> {
    values
        .par_iter()
        .filter(|v| v.is_finite())
        .map(|v| (*v, *v))
        .reduce_with(|a, b| (a.0.min(b.0), a.1.max(b.1)))
}

struct Prepared {
    report: FeatureReport,
    /// (p_hi, p_lo) in the marginal family, when a certificate exists.
    ks_p: Option<(f64, f64)>,
    effect: Option<(f64, f64)>,
    missing_p: Option<f64>,
}

fn prepare_feature(f: &ContinuousInput<'_>, params: &EngineParams) -> Result<Prepared> {
    let total_r = f.reference.len() as u64;
    let total_t = f.target.len() as u64;
    let mut rep = FeatureReport {
        name: f.name.to_string(),
        label: Label::Invalid,
        degenerate: false,
        certificate: None,
        effect_interval: None,
        cohens_d: None,
        missing_rate_ref: 0.0,
        missing_rate_tgt: 0.0,
        missingness_p: None,
        missingness_drift: false,
    };
    let rng_r = finite_min_max(f.reference);
    let rng_t = finite_min_max(f.target);
    let n_ok = f.reference.par_iter().filter(|v| v.is_finite()).count() as u64;
    let m_ok = f.target.par_iter().filter(|v| v.is_finite()).count() as u64;
    if total_r > 0 {
        rep.missing_rate_ref = (total_r - n_ok) as f64 / total_r as f64;
    }
    if total_t > 0 {
        rep.missing_rate_tgt = (total_t - m_ok) as f64 / total_t as f64;
    }
    let missing_p = if total_r > 0 && total_t > 0 {
        Some(two_proportion_p(total_r - n_ok, total_r, total_t - m_ok, total_t))
    } else {
        None
    };
    rep.missingness_p = missing_p;

    let (Some((rmin, rmax)), Some((tmin, tmax))) = (rng_r, rng_t) else {
        // Empty sample after excluding missing values: status Invalid, no statistic.
        return Ok(Prepared { report: rep, ks_p: None, effect: None, missing_p });
    };

    if rmin == rmax {
        // Constant reference: degenerate rules, no test, no sigma_R normalisation.
        rep.degenerate = true;
        rep.label = if tmin == tmax && tmin == rmin {
            Label::NoActionableDrift
        } else {
            Label::Drift
        };
        return Ok(Prepared { report: rep, ks_p: None, effect: None, missing_p });
    }

    let edges = Edges::from_reference_quantiles(f.reference, params.bins)?;
    let rs = ReferenceSummary::from_slice(f.reference, edges, params.chunk_size, params.workers)?;
    let mut ev = StreamingEvaluator::new(&rs);
    ev.update_parallel(f.target, params.chunk_size, params.workers)?;
    let cert = ev.certificate()?;

    let effect = match params.gate {
        Gate::Ks => (cert.ks_lo, cert.ks_hi),
        Gate::W1Normalized => (cert.w1_lo / rs.std, cert.w1_hi / rs.std),
    };
    let tm = parallel_comoments(&[f.target], params.chunk_size, params.workers)?;
    if rs.n >= 2 && tm.n >= 2 {
        let (n, m) = (rs.n as f64, tm.n as f64);
        let sp2 = ((n - 1.0) * rs.std * rs.std + (m - 1.0) * tm.std(0) * tm.std(0)) / (n + m - 2.0);
        if sp2 > 0.0 {
            rep.cohens_d = Some((tm.mean[0] - rs.mean) / sp2.sqrt());
        }
    }
    rep.certificate = Some(cert.clone());
    rep.effect_interval = Some(effect);
    Ok(Prepared {
        report: rep,
        ks_p: Some((cert.p_hi, cert.p_lo)),
        effect: Some(effect),
        missing_p,
    })
}

/// Algorithm 3 over a set of continuous features (and optional categorical count pairs).
///
/// Marginal family: KS p-value intervals of continuous features, chi-square p-values of
/// categorical features, and missing-rate p-values. Correlation family: Fisher-z p-values of all
/// admissible feature pairs. Holm runs within each family at alpha_m / alpha_r, so the overall
/// familywise error rate is at most alpha_m + alpha_r (asymptotically). Holm is applied to the
/// upper bounds p_hi (certain rejections) and to the lower bounds p_lo (possible rejections).
pub fn drift_report(
    continuous: &[ContinuousInput<'_>],
    categorical: &[CategoricalInput<'_>],
    params: &EngineParams,
) -> Result<DriftReport> {
    params.validate()?;
    let prepared: Vec<Prepared> = continuous
        .iter()
        .map(|f| prepare_feature(f, params))
        .collect::<Result<_>>()?;

    // Chi-square for categorical features.
    let cat_tests: Vec<(Option<f64>, Option<f64>)> = categorical
        .iter()
        .map(|c| match chi_square_counts(c.reference_counts, c.target_counts) {
            Ok(ChiSquareOutcome::Tested(r)) => Ok((Some(r.p_value), Some(r.cramers_v))),
            Ok(ChiSquareOutcome::Insufficient { .. }) => Ok((None, None)),
            Err(e) => Err(e),
        })
        .collect::<Result<_>>()?;

    // Marginal family bookkeeping: (p_hi, p_lo) entries and where they came from.
    enum Slot {
        Ks(usize),
        Missing(usize),
        Cat(usize),
    }
    let mut slots: Vec<Slot> = Vec::new();
    let mut p_hi: Vec<f64> = Vec::new();
    let mut p_lo: Vec<f64> = Vec::new();
    for (i, pr) in prepared.iter().enumerate() {
        if let Some((hi, lo)) = pr.ks_p {
            slots.push(Slot::Ks(i));
            p_hi.push(hi);
            p_lo.push(lo);
        }
        if let Some(p) = pr.missing_p {
            slots.push(Slot::Missing(i));
            p_hi.push(p);
            p_lo.push(p);
        }
    }
    for (i, (p, _)) in cat_tests.iter().enumerate() {
        if let Some(p) = p {
            slots.push(Slot::Cat(i));
            p_hi.push(*p);
            p_lo.push(*p);
        }
    }
    let rej_hi = holm_reject(&p_hi, params.alpha_m);
    let rej_lo = holm_reject(&p_lo, params.alpha_m);

    let mut features: Vec<FeatureReport> = prepared.iter().map(|p| p.report.clone()).collect();
    let mut categorical_out: Vec<CategoricalReport> = categorical
        .iter()
        .zip(cat_tests.iter())
        .map(|(c, (p, v))| CategoricalReport {
            name: c.name.to_string(),
            label: Label::Invalid,
            p_value: *p,
            cramers_v: *v,
        })
        .collect();

    for (s, slot) in slots.iter().enumerate() {
        match slot {
            Slot::Ks(i) => {
                let (e_lo, e_hi) = prepared[*i].effect.expect("effect set with ks_p");
                features[*i].label = if rej_hi[s] && e_lo >= params.tau_g {
                    Label::Drift
                } else if !rej_lo[s] || e_hi < params.tau_g {
                    Label::NoActionableDrift
                } else {
                    Label::Indeterminate
                };
            }
            Slot::Missing(i) => {
                let gap = (features[*i].missing_rate_tgt - features[*i].missing_rate_ref).abs();
                features[*i].missingness_drift = rej_hi[s] && gap >= params.tau_pi;
            }
            Slot::Cat(i) => {
                let v = categorical_out[*i].cramers_v.unwrap_or(0.0);
                categorical_out[*i].label = if rej_hi[s] && v >= params.tau_v {
                    Label::Drift
                } else {
                    Label::NoActionableDrift
                };
            }
        }
    }

    // Correlation family.
    let mut pairs_out: Vec<PairReport> = Vec::new();
    let (mut dropped_ref, mut dropped_tgt) = (0u64, 0u64);
    if params.correlation && continuous.len() >= 2 {
        let rcols: Vec<&[f64]> = continuous.iter().map(|f| f.reference).collect();
        let tcols: Vec<&[f64]> = continuous.iter().map(|f| f.target).collect();
        let mr = parallel_comoments(&rcols, params.chunk_size, params.workers)?;
        let mt = parallel_comoments(&tcols, params.chunk_size, params.workers)?;
        dropped_ref = mr.dropped;
        dropped_tgt = mt.dropped;
        let d = continuous.len();
        let mut raw: Vec<PairReport> = Vec::new();
        for i in 0..d {
            for j in (i + 1)..d {
                let (r1, r2) = (mr.correlation(i, j), mt.correlation(i, j));
                let delta = match (r1, r2) {
                    (Some(a), Some(b)) => Some((a - b).abs()),
                    _ => None,
                };
                let p = match (r1, r2) {
                    (Some(a), Some(b)) if mr.n > 3 && mt.n > 3 && a.abs() < 1.0 && b.abs() < 1.0 => {
                        Some(crate::delta_correlation::fisher_z_p_value(
                            a,
                            mr.n as usize,
                            b,
                            mt.n as usize,
                        ))
                    }
                    _ => None,
                };
                raw.push(PairReport {
                    i,
                    j,
                    r_ref: r1,
                    r_tgt: r2,
                    delta,
                    p_value: p,
                    label: Label::Invalid,
                });
            }
        }
        let testable: Vec<usize> = raw
            .iter()
            .enumerate()
            .filter(|(_, p)| p.p_value.is_some())
            .map(|(k, _)| k)
            .collect();
        let ps: Vec<f64> = testable.iter().map(|k| raw[*k].p_value.unwrap()).collect();
        let rej = holm_reject(&ps, params.alpha_r);
        for (pos, k) in testable.iter().enumerate() {
            let dl = raw[*k].delta.unwrap_or(0.0);
            raw[*k].label = if rej[pos] && dl >= params.tau_r {
                Label::Drift
            } else {
                Label::NoActionableDrift
            };
        }
        pairs_out = raw;
    }

    Ok(DriftReport {
        features,
        categorical: categorical_out,
        pairs: pairs_out,
        dropped_ref,
        dropped_tgt,
    })
}
