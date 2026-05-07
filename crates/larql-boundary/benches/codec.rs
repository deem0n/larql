use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use larql_boundary::codec::{bf16, int8};
use larql_boundary::metadata;

// Gemma 3 4B dimensions
const D: usize = 2560;
const VOCAB: usize = 262_145;

fn residual() -> Vec<f32> {
    (0..D).map(|i| (i as f32 * 0.01).sin() * 80.0).collect()
}

fn logits() -> Vec<f32> {
    let mut l = vec![0.01f32; VOCAB];
    l[42] = 10.0;
    l[17] = 3.0;
    l
}

fn bench_bf16(c: &mut Criterion) {
    let r = residual();
    let enc = bf16::encode(&r);

    c.bench_function("bf16_encode_d2560", |b| {
        b.iter(|| bf16::encode(black_box(&r)))
    });
    c.bench_function("bf16_decode_d2560", |b| {
        b.iter(|| bf16::decode(black_box(&enc)))
    });
}

fn bench_int8(c: &mut Criterion) {
    let r = residual();
    let payload = int8::encode(&r);
    let bytes = payload.to_bytes();

    c.bench_function("int8_encode_d2560", |b| {
        b.iter(|| int8::encode(black_box(&r)))
    });
    c.bench_function("int8_decode_d2560", |b| {
        b.iter(|| {
            let p = int8::Payload::from_bytes(black_box(&bytes));
            int8::decode(black_box(&p))
        })
    });
}

fn bench_metadata(c: &mut Criterion) {
    let raw = logits();
    let hat = {
        let mut h = raw.clone();
        h[42] = 9.5;
        h
    };

    c.bench_function("metadata_compute_no_hat", |b| {
        b.iter(|| metadata::compute(black_box(&raw), None))
    });
    c.bench_function("metadata_compute_with_hat", |b| {
        b.iter(|| metadata::compute(black_box(&raw), Some(black_box(&hat))))
    });
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");
    let r = residual();
    for d in [256usize, 1024, 2560] {
        let rv: Vec<f32> = r[..d].to_vec();
        group.bench_with_input(BenchmarkId::new("bf16", d), &rv, |b, v| {
            b.iter(|| bf16::decode(&bf16::encode(black_box(v))))
        });
        group.bench_with_input(BenchmarkId::new("int8", d), &rv, |b, v| {
            b.iter(|| int8::decode(&int8::encode(black_box(v))))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bf16,
    bench_int8,
    bench_metadata,
    bench_roundtrip
);
criterion_main!(benches);
