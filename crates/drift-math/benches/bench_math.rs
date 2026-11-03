use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use drift_core::DistributionSample;
use drift_core::types::ColumnName;
use drift_math::{ks_two_sample, wasserstein1_slices};

fn bench_drift_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("drift_math_1m");
    
    let n = 1_000_000;
    let ref_data: Vec<f64> = (0..n).map(|x| x as f64).collect();
    let tgt_data: Vec<f64> = (0..n).map(|x| (x as f64) + 0.1).collect();
    
    let col_ref = ColumnName::new("ref_col".to_string()).expect("valid ref col");
    let col_tgt = ColumnName::new("tgt_col".to_string()).expect("valid tgt col");
    
    let s_ref = DistributionSample::new(col_ref, ref_data.clone()).expect("valid ref sample");
    let s_tgt = DistributionSample::new(col_tgt, tgt_data.clone()).expect("valid tgt sample");

    group.bench_function(BenchmarkId::new("ks_test", n), |b| {
        b.iter(|| ks_two_sample(&s_ref, &s_tgt, 0.05))
    });

    group.bench_function(BenchmarkId::new("wasserstein1_slices", n), |b| {
        b.iter(|| wasserstein1_slices(&ref_data, &tgt_data))
    });

    group.finish();
}

criterion_group!(benches, bench_drift_math);
criterion_main!(benches);
