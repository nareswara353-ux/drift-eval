use std::time::Instant;
use drift_core::{ColumnName, DistributionSample};
use drift_math::{ks_two_sample, wasserstein1_slices};
use drift_math::histogram_engine::*;

fn main() {
    println!("==========================================================================");
    println!("  UNIFIED COMPREHENSIVE EXPERIMENT: DRIFT-MATH SYSTEM EVALUATION");
    println!("==========================================================================");
    println!("Hardware: AMD Ryzen 7 PRO 4750U (8 Cores / 16 Threads)");
    println!();

    let chunk_size = 65_536;
    let b = 1024;

    for &n in &[100_000, 1_000_000, 10_000_000] {
        println!("--------------------------------------------------------------------------");
        println!("DATASET SIZE: N = {:10} rows per sample", n);
        println!("--------------------------------------------------------------------------");

        let ref_norm: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 10.0 - 5.0).collect();
        let tgt_norm: Vec<f64> = ref_norm.iter().map(|x| x + 0.1).collect();

        // 1. Exact Rust Baseline
        let col_ref = ColumnName::new("ref_col".to_string()).expect("valid col");
        let col_tgt = ColumnName::new("tgt_col".to_string()).expect("valid col");
        let s_ref = DistributionSample::new(col_ref, ref_norm.clone()).expect("valid sample");
        let s_tgt = DistributionSample::new(col_tgt, tgt_norm.clone()).expect("valid sample");

        let t0 = Instant::now();
        let _ks_res = ks_two_sample(&s_ref, &s_tgt, 0.05).expect("ks ok");
        let exact_ks_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let _w1_res = wasserstein1_slices(&ref_norm, &tgt_norm).expect("w1 ok");
        let exact_w1_ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!("  [Exact Baseline]");
        println!("    - Exact KS Test          : {:8.2} ms", exact_ks_ms);
        println!("    - Exact W1 Slices        : {:8.2} ms", exact_w1_ms);
        println!("    - Exact Combined Total   : {:8.2} ms", exact_ks_ms + exact_w1_ms);

        // 2. Histogram Engine Evaluation (Single-threaded P=1 vs Multi-threaded P=16)
        for &workers in &[1, 16] {
            let edges_ref = Edges::uniform(-5.0, 5.0, b).expect("valid edges ref");
            let edges_tgt = Edges::uniform(-5.0, 5.0, b).expect("valid edges tgt");

            // Cold Prep: Reference Summary preparation
            let t_prep = Instant::now();
            let ref_summary = ReferenceSummary::from_slice(&ref_norm, edges_ref, chunk_size, workers).expect("valid ref summary");
            let prep_ms = t_prep.elapsed().as_secs_f64() * 1000.0;

            // Warm Eval: Streaming Evaluation + Certification
            let t_eval = Instant::now();
            let hist_tgt = parallel_histogram(&tgt_norm, &edges_tgt, chunk_size, workers).expect("valid tgt histogram");
            let _cert = certify(&ref_summary, &hist_tgt).expect("valid certification");
            let eval_ms = t_eval.elapsed().as_secs_f64() * 1000.0;

            let total_cold_ms = prep_ms + eval_ms;

            println!("  [Histogram Engine (Workers = {:2})]", workers);
            println!("    - Cold Prep (Ref Quantile): {:8.2} ms", prep_ms);
            println!("    - Warm Streaming Eval     : {:8.2} ms", eval_ms);
            println!("    - Cold-Start Total        : {:8.2} ms", total_cold_ms);
        }
        println!();
    }
}
