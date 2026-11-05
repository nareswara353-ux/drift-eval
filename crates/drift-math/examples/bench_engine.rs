use std::time::Instant;
use drift_math::histogram_engine::*;

fn main() {
    println!("=== PARALLEL HISTOGRAM ENGINE BENCHMARK ===");
    let chunk_size = 65_536;
    let workers = 16;
    let b = 1024;

    for &n in &[100_000, 1_000_000, 10_000_000] {
        let ref_norm: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 10.0 - 5.0).collect();
        let tgt_norm: Vec<f64> = ref_norm.iter().map(|x| x + 0.1).collect();

        let t0 = Instant::now();
        let edges_ref = Edges::uniform(-5.0, 5.0, b).expect("valid edges ref");
        let edges_tgt = Edges::uniform(-5.0, 5.0, b).expect("valid edges tgt");

        let ref_summary = ReferenceSummary::from_slice(&ref_norm, edges_ref, chunk_size, workers).expect("valid ref summary");
        let hist_tgt = parallel_histogram(&tgt_norm, &edges_tgt, chunk_size, workers).expect("valid tgt histogram");

        let _cert = certify(&ref_summary, &hist_tgt).expect("valid certification");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!("N = {:10} | Parallel Histogram Engine: {:8.2} ms", n, elapsed_ms);
    }
}
