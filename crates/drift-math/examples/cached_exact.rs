use std::time::Instant;
use drift_math::ks_two_sample;
use drift_core::{ColumnName, DistributionSample};

fn main() {
    println!("=== EXACT BASELINE WITH CACHED REFERENCE BENCHMARK (N=10M) ===");
    let n = 10_000_000;
    
    let ref_norm: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 10.0 - 5.0).collect();
    let tgt_norm: Vec<f64> = ref_norm.iter().map(|x| x + 0.1).collect();
    
    // 1. Reference Sorting Prep (Cold Cached once)
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
    
    println!("Exact Cached Prep (T_prep,X) : {:8.2} ms", prep_ms);
    println!("Exact Cached Eval (T_eval,Y) : {:8.2} ms", eval_ms);
}
