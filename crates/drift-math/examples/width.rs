use drift_math::histogram_engine::*;

fn main() {
    println!("=== DRIFT-MATH CERTIFICATE WIDTH EVALUATION ===");
    let n = 100_000;
    let chunk_size = 10_000;
    let workers = 4;
    
    for b in [256, 1024, 4096] {
        let ref_norm: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 10.0 - 5.0).collect();
        let tgt_norm: Vec<f64> = ref_norm.iter().map(|x| x + 0.1).collect();
        
        let edges_ref = Edges::uniform(-5.0, 5.0, b).expect("valid edges ref");
        let edges_tgt = Edges::uniform(-5.0, 5.0, b).expect("valid edges tgt");
        
        let ref_summary = ReferenceSummary::from_slice(&ref_norm, edges_ref, chunk_size, workers).expect("valid ref summary");
        let hist_tgt = parallel_histogram(&tgt_norm, &edges_tgt, chunk_size, workers).expect("valid tgt histogram");
        
        if let Ok(cert) = certify(&ref_summary, &hist_tgt) {
            println!("B = {:4} | Certificate Generated Successfully", b);
            println!("        -> KS Interval Bounds  : [{:.6}, {:.6}]", cert.ks_lo, cert.ks_hi);
            println!("        -> KS Delta_B (Width)  : {:.6}", cert.delta_b);
            println!("        -> W1 Estimate (W1_hat): {:.6}", cert.w1_hat);
            println!("        -> W1 Epsilon Error    : {:.6}", cert.w1_eps);
        }
    }
}
