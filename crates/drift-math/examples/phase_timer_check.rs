use std::time::Instant;
use rayon::prelude::*;
use drift_core::{ColumnName, DistributionSample};
use drift_math::ks_two_sample;
use drift_math::histogram_engine::*;

fn main() {
    println!("=== PHASE-ISOLATED & EXACT P=16 BENCHMARK (N=10M) ===");
    let n = 10_000_000;
    let b = 1024;
    let chunk_size = 65_536;

    // True unsorted random generator (Xorshift64)
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

    // ---------------------------------------------------------
    // 1. ISOLATED EXACT BASELINE PHASE TIMING (P=1)
    // ---------------------------------------------------------
    let t_alloc = Instant::now();
    let mut ref_sorted = ref_norm.clone();
    let alloc_ms = t_alloc.elapsed().as_secs_f64() * 1000.0;

    let t_sort_ref = Instant::now();
    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sort_ref_ms = t_sort_ref.elapsed().as_secs_f64() * 1000.0;

    let mut tgt_sorted = tgt_norm.clone();
    let t_sort_tgt = Instant::now();
    tgt_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sort_tgt_ms = t_sort_tgt.elapsed().as_secs_f64() * 1000.0;

    let col_ref = ColumnName::new("ref_col".to_string()).unwrap();
    let col_tgt = ColumnName::new("tgt_col".to_string()).unwrap();
    let s_ref = DistributionSample::new(col_ref.clone(), ref_sorted).unwrap();
    let s_tgt = DistributionSample::new(col_tgt.clone(), tgt_sorted).unwrap();

    let t_merge = Instant::now();
    let _ks = ks_two_sample(&s_ref, &s_tgt, 0.05).unwrap();
    let merge_ms = t_merge.elapsed().as_secs_f64() * 1000.0;

    println!("[Exact Baseline P=1 Phase Breakdown]");
    println!("  - Memory Allocation (.clone) : {:8.2} ms", alloc_ms);
    println!("  - Pure Sort Ref X             : {:8.2} ms", sort_ref_ms);
    println!("  - Pure Sort Batch Y           : {:8.2} ms", sort_tgt_ms);
    println!("  - Pure KS Merge Pass          : {:8.2} ms", merge_ms);
    println!("  - Exact Total Eval (Sort Y + Merge): {:8.2} ms", sort_tgt_ms + merge_ms);
    println!();

    // ---------------------------------------------------------
    // 2. EXACT BASELINE P=16 (PARALLEL SORTING)
    // ---------------------------------------------------------
    let mut ref_sorted_p16 = ref_norm.clone();
    let t_p16_prep = Instant::now();
    ref_sorted_p16.par_sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p16_prep_ms = t_p16_prep.elapsed().as_secs_f64() * 1000.0;

    let mut tgt_sorted_p16 = tgt_norm.clone();
    let t_p16_eval = Instant::now();
    tgt_sorted_p16.par_sort_by(|a, b| a.partial_cmp(b).unwrap());
    let s_ref_p16 = DistributionSample::new(col_ref, ref_sorted_p16).unwrap();
    let s_tgt_p16 = DistributionSample::new(col_tgt, tgt_sorted_p16).unwrap();
    let _ks_p16 = ks_two_sample(&s_ref_p16, &s_tgt_p16, 0.05).unwrap();
    let p16_eval_ms = t_p16_eval.elapsed().as_secs_f64() * 1000.0;

    println!("[Exact Baseline Cached (Multi-Thread P=16)]");
    println!("  - Parallel Sort Ref X (T_prep): {:8.2} ms", p16_prep_ms);
    println!("  - Parallel Sort Y + Merge (T_eval): {:8.2} ms", p16_eval_ms);
    println!("  - Cold Total (K=1)            : {:8.2} ms", p16_prep_ms + p16_eval_ms);
    println!();
}
