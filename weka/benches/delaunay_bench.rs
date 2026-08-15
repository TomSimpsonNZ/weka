//! Benchmark: weka's pure-Rust divide-and-conquer Delaunay vs the original C
//! library (called over FFI) on identical uniform-random point sets.
//!
//! Both compute the same triangulation; this measures wall-clock per call so we
//! can report the Rust/C ratio (target: <= 1.0, i.e. as fast or faster).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

fn random_points(n: usize, seed: u64) -> Vec<[f64; 2]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| [rng.gen::<f64>(), rng.gen::<f64>()])
        .collect()
}

fn c_input(points: &[[f64; 2]]) -> ctriangle_sys::Input {
    let mut pointlist = Vec::with_capacity(points.len() * 2);
    for p in points {
        pointlist.push(p[0]);
        pointlist.push(p[1]);
    }
    ctriangle_sys::Input {
        pointlist,
        ..Default::default()
    }
}

fn bench_delaunay(c: &mut Criterion) {
    let mut group = c.benchmark_group("delaunay_uniform");
    for &n in &[1_000usize, 10_000, 100_000] {
        let pts = random_points(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("weka", n), &pts, |b, pts| {
            b.iter(|| {
                let t = weka::delaunay_points(black_box(pts), false);
                black_box(t.triangles.len())
            });
        });

        group.bench_with_input(BenchmarkId::new("c_triangle", n), &pts, |b, pts| {
            let input = c_input(pts);
            b.iter(|| {
                let out = ctriangle_sys::triangulate_safe("zQ", black_box(&input));
                black_box(out.numberoftriangles)
            });
        });
    }
    group.finish();
}

/// A square boundary PSLG with `n` uniform-random interior points.
fn square_with_interior(n: usize, seed: u64) -> (Vec<[f64; 2]>, Vec<[usize; 2]>) {
    let mut pts = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..n {
        pts.push([
            0.02 + rng.gen::<f64>() * 0.96,
            0.02 + rng.gen::<f64>() * 0.96,
        ]);
    }
    (pts, vec![[0, 1], [1, 2], [2, 3], [3, 0]])
}

fn c_pslg_input(points: &[[f64; 2]], segs: &[[usize; 2]]) -> ctriangle_sys::Input {
    let mut pointlist = Vec::with_capacity(points.len() * 2);
    for p in points {
        pointlist.push(p[0]);
        pointlist.push(p[1]);
    }
    let mut segmentlist = Vec::with_capacity(segs.len() * 2);
    for s in segs {
        segmentlist.push(s[0] as i32);
        segmentlist.push(s[1] as i32);
    }
    ctriangle_sys::Input {
        pointlist,
        segmentlist,
        ..Default::default()
    }
}

/// Constrained Delaunay (the FEA domain-meshing path) — weka vs C.
fn bench_cdt(c: &mut Criterion) {
    let mut group = c.benchmark_group("cdt");
    for &n in &[1_000usize, 10_000, 100_000] {
        let (pts, segs) = square_with_interior(n, 7);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("weka", n),
            &(pts.clone(), segs.clone()),
            |b, (p, s)| {
                b.iter(|| {
                    black_box(
                        weka::cdt_pslg(black_box(p), None, black_box(s), None, false)
                            .triangles
                            .len(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("c_triangle", n),
            &(pts, segs),
            |b, (p, s)| {
                let input = c_pslg_input(p, s);
                b.iter(|| {
                    black_box(
                        ctriangle_sys::triangulate_safe("pzQ", black_box(&input)).numberoftriangles,
                    )
                });
            },
        );
    }
    group.finish();
}

/// Quality meshing (`-q -a`, the FEA refinement path) — weka vs C, on a fixed
/// square domain at a few area targets.
fn bench_quality(c: &mut Criterion) {
    use weka::Triangulator;
    let mut group = c.benchmark_group("quality_q20");
    let (pts, segs) = square_with_interior(50, 3);
    for &area in &[1e-3, 1e-4, 1e-5] {
        let pslg = weka::Pslg {
            points: pts.clone(),
            segments: segs.clone(),
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::new("weka", area), &pslg, |b, pslg| {
            b.iter(|| {
                let t = Triangulator::new()
                    .min_angle(20.0)
                    .max_area(area)
                    .triangulate_pslg(black_box(pslg))
                    .unwrap();
                black_box(t.triangles.len())
            });
        });
        let input = c_pslg_input(&pts, &segs);
        let sw = format!("pq20a{area}zQ");
        group.bench_with_input(BenchmarkId::new("c_triangle", area), &input, |b, input| {
            b.iter(|| {
                black_box(ctriangle_sys::triangulate_safe(&sw, black_box(input)).numberoftriangles)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_delaunay, bench_cdt, bench_quality);
criterion_main!(benches);
