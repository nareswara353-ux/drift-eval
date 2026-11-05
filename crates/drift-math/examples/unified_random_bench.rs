use std::time::Instant;
use drift_core::{ColumnName, DistributionSample};
use drift_math::ks_two_sample;
use drift_math::histogram_engine::*;

fn main() {
    println!("=== UNIFIED RANDOM DATA BENCHMARK (N=10M) ===");
    let n = 10_000_000;
    let chunk_size = 65_536;
    let b = 1024;
    
    // True unsorted random generator
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
        (rng_tgt as f64) / (u64::MAX as f64) * 10.0 - 4.9
    };

    let ref_norm: Vec<f64> = (0..n).map(|_| next_ref()).collect();
    let tgt_norm: Vec<f64> = (0..n).map(|_| next_tgt()).collect();

    // 1. Exact Baseline Cached
    let t_prep = Instant::now();
    let mut ref_sorted = ref_norm.clone();
    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let exact_prep_ms = t_prep.elapsed().as_secs_f64() * 1000.0;

    let col_ref = ColumnName::new("ref_col".to_string()).unwrap();
    let col_tgt = ColumnName::new("tgt_col".to_string()).unwrap();
    let s_ref = DistributionSample::new(col_ref, ref_sorted).unwrap();
    let s_tgt = DistributionSample::new(col_tgt, tgt_norm.clone()).unwrap();

    let t_eval = Instant::now();
    let _ks = ks_two_sample(&s_ref, &s_tgt, 0.05).unwrap();
    let exact_eval_ms = t_eval.elapsed().as_secs_f64() * 1000.0;

    println!("[Exact Baseline Cached (Random Data)]");
    println!("  - Prep T_prep,X : {:8.2} ms", exact_prep_ms);
    println!("  - Eval T_eval,Y : {:8.2} ms", exact_eval_ms);
    println!();

    // 2. Proposed Engine (P=1 vs P=16 on Same Random Data)
    for &workers in &[1, 16] {
        let edges_ref = Edges::uniform(-5.0, 5.0, b).unwrap();
        let edges_tgt = Edges::uniform(-5.0, 5.0, b).unwrap();

        let t0 = Instant::now();
        let ref_summary = ReferenceSummary::from_slice(&ref_norm, edges_ref, chunk_size, workers).unwrap();
        let eng_prep_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let hist_tgt = parallel_histogram(&tgt_norm, &edges_tgt, chunk_size, workers).unwrap();
        let _cert = certify(&ref_summary, &hist_tgt).unwrap();
        let eng_eval_ms = t1.elapsed().as_secs_f64() * 1000.0;

        println!("[Proposed Engine (Workers P = {:2}, Random Data)]", workers);
        println!("  - Cold Prep (T_prep) : {:8.2} ms", eng_prep_ms);
        println!("  - Warm Eval (T_eval) : {:8.2} ms", eng_eval_ms);
        println!("  - Cold Total (T_cold): {:8.2} ms", eng_prep_ms + eng_eval_ms);
        println!();
    }
}
