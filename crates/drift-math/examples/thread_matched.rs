use std::time::Instant;
use drift_core::{ColumnName, DistributionSample};
use drift_math::{ks_two_sample, wasserstein1_slices};
use drift_math::histogram_engine::*;

fn main() {
    println!("==========================================================================");
    println!("  THREAD-MATCHED & COMBINED BASELINE EXPERIMENT (N = 10,000,000)");
    println!("==========================================================================");

    let n = 10_000_000;
    let chunk_size = 65_536;
    let b = 1024;

    let ref_norm: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 10.0 - 5.0).collect();
    let tgt_norm: Vec<f64> = ref_norm.iter().map(|x| x + 0.1).collect();

    let col_ref = ColumnName::new("ref_col".to_string()).expect("valid col");
    let col_tgt = ColumnName::new("tgt_col".to_string()).expect("valid col");
    let s_ref = DistributionSample::new(col_ref, ref_norm.clone()).expect("valid sample");
    let s_tgt = DistributionSample::new(col_tgt, tgt_norm.clone()).expect("valid sample");

    // 1. Exact Combined Baseline (Sequential P=1)
    let t0 = Instant::now();
    let _ks = ks_two_sample(&s_ref, &s_tgt, 0.05).expect("ks ok");
    let _w1 = wasserstein1_slices(&ref_norm, &tgt_norm).expect("w1 ok");
    let exact_p1_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("[Exact Baseline (KS + W1 Combined)]");
    println!("  - Single-Thread (P=1) Combined : {:8.2} ms", exact_p1_ms);
    println!();

    // 2. Proposed Engine (P=1 and P=16 Breakdown)
    for &workers in &[1, 16] {
        let edges_ref = Edges::uniform(-5.0, 5.0, b).expect("valid edges ref");
        let edges_tgt = Edges::uniform(-5.0, 5.0, b).expect("valid edges tgt");

        let t_prep = Instant::now();
        let ref_summary = ReferenceSummary::from_slice(&ref_norm, edges_ref, chunk_size, workers).expect("valid ref summary");
        let prep_ms = t_prep.elapsed().as_secs_f64() * 1000.0;

        let t_eval = Instant::now();
        let hist_tgt = parallel_histogram(&tgt_norm, &edges_tgt, chunk_size, workers).expect("valid tgt histogram");
        let _cert = certify(&ref_summary, &hist_tgt).expect("valid certification");
        let eval_ms = t_eval.elapsed().as_secs_f64() * 1000.0;

        println!("[Proposed Engine (Workers P = {:2})]", workers);
        println!("  - Cold Prep (T_prep)          : {:8.2} ms", prep_ms);
        println!("  - Warm Eval (T_eval)          : {:8.2} ms", eval_ms);
        println!("  - Cold Total (T_cold)         : {:8.2} ms", prep_ms + eval_ms);
        println!();
    }
}
