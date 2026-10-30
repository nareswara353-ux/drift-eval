use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use drift_core::{ColumnName, DistributionSample};
use drift_math::{ks_two_sample, wasserstein1};
use std::hint::black_box;

fn make_sample(name: &str, n: usize, offset: f64) -> DistributionSample {
    let column = ColumnName::new(name).expect("column");
    let values: Vec<f64> = (0..n).map(|i| (i as f64).sin() + offset).collect();
    DistributionSample::new(column, values).expect("sample")
}

fn bench_ks(c: &mut Criterion) {
    let mut group = c.benchmark_group("ks_two_sample");
    for size in [1_000_usize, 10_000, 100_000] {
        let reference = make_sample("reference", size, 0.0);
        let target = make_sample("target", size, 0.1);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let outcome = ks_two_sample(black_box(&reference), black_box(&target), 0.05)
                    .expect("ks");
                black_box(outcome);
            });
        });
    }
    group.finish();
}

fn bench_wasserstein(c: &mut Criterion) {
    let mut group = c.benchmark_group("wasserstein1");
    for size in [1_000_usize, 10_000, 100_000] {
        let reference = make_sample("reference", size, 0.0);
        let target = make_sample("target", size, 0.2);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let outcome = wasserstein1(black_box(&reference), black_box(&target), 4096)
                    .expect("w1");
                black_box(outcome);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ks, bench_wasserstein);
criterion_main!(benches);
