use std::time::Instant;
use drift_core::{ColumnName, DistributionSample};
use drift_math::ks_two_sample;

fn main() {
    println!("=== REAL RANDOM DATA CACHED BASELINE BENCHMARK (N=10M) ===");
    let n = 10_000_000;
    
    // Xorshift64 PRNG to generate true unsorted random f64 data
    let mut rng_ref: u64 = 88172645463325252;
    let mut next_ref = || {
        rng_ref ^= rng_ref << 13;
        rng_ref ^= rng_ref >> 7;
        rng_ref ^= rng_ref << 17;
        (rng_ref as f64) / (u64::MAX as f64) * 10.0 - 5.0
    };

    let mut rng_tgt: u64 = 12345678901234567;
    let mut next_tgt = || {
        rng_tgt ^= rng_tgt << 13;
        rng_tgt ^= rng_tgt >> 7;
        rng_tgt ^= rng_tgt << 17;
        (rng_tgt as f64) / (u64::MAX as f64) * 10.0 - 4.9 // shifted by 0.1
    };

    let ref_norm: Vec<f64> = (0..n).map(|_| next_ref()).collect();
    let tgt_norm: Vec<f64> = (0..n).map(|_| next_tgt()).collect();

    // 1. Reference Sorting Prep (Cold Cached)
    let t_prep = Instant::now();
    let mut ref_sorted = ref_norm.clone();
    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let prep_ms = t_prep.elapsed().as_secs_f64() * 1000.0;

    // 2. Evaluation Batch Process (Sort Y + Merge)
    let col_ref = ColumnName::new("ref_col".to_string()).unwrap();
    let col_tgt = ColumnName::new("tgt_col".to_string()).unwrap();
    let s_ref = DistributionSample::new(col_ref, ref_sorted).unwrap();
    let s_tgt = DistributionSample::new(col_tgt, tgt_norm).unwrap();

    let t_eval = Instant::now();
    let _ks = ks_two_sample(&s_ref, &s_tgt, 0.05).unwrap();
    let eval_ms = t_eval.elapsed().as_secs_f64() * 1000.0;

    println!("Real Unsorted Data Results:");
    println!("  - Exact Cached Prep (T_prep,X) : {:8.2} ms", prep_ms);
    println!("  - Exact Cached Eval (T_eval,Y) : {:8.2} ms", eval_ms);
}
