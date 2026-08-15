//! P4 differential tests: weka constrained Delaunay vs the C library.
//!
//! The domain is a unit square (its 4 edges are segments, so the convex hull is
//! fully segment-bounded and C's hole-carving removes nothing) plus interior
//! points, with one long internal diagonal segment that forces edge flips during
//! recovery. We assert the triangle set, triangle count, and segment recovery
//! match C run with `pznQ`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use weka::predicates::orient2d;
use weka::Triangulation;

fn tri_key(p: [[f64; 2]; 3]) -> [[u64; 2]; 3] {
    let mut k = [
        [p[0][0].to_bits(), p[0][1].to_bits()],
        [p[1][0].to_bits(), p[1][1].to_bits()],
        [p[2][0].to_bits(), p[2][1].to_bits()],
    ];
    k.sort_unstable();
    k
}

fn seg_key(a: [f64; 2], b: [f64; 2]) -> [[u64; 2]; 2] {
    let mut k = [
        [a[0].to_bits(), a[1].to_bits()],
        [b[0].to_bits(), b[1].to_bits()],
    ];
    k.sort_unstable();
    k
}

fn weka_tris(t: &Triangulation) -> BTreeSet<[[u64; 2]; 3]> {
    t.triangles
        .iter()
        .map(|&[a, b, c]| tri_key([t.points[a], t.points[b], t.points[c]]))
        .collect()
}

fn c_tris(out: &ctriangle_sys::Output) -> BTreeSet<[[u64; 2]; 3]> {
    let pt = |i: i32| {
        let i = i as usize;
        [out.pointlist[2 * i], out.pointlist[2 * i + 1]]
    };
    out.trianglelist
        .chunks_exact(3)
        .map(|c| tri_key([pt(c[0]), pt(c[1]), pt(c[2])]))
        .collect()
}

fn build_case(n_interior: usize, seed: u64) -> (Vec<[f64; 2]>, Vec<[usize; 2]>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut pts = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    for _ in 0..n_interior {
        pts.push([0.05 + rng.gen::<f64>() * 0.9, 0.05 + rng.gen::<f64>() * 0.9]);
    }
    let mut segs = vec![[0, 1], [1, 2], [2, 3], [3, 0]];
    if n_interior >= 2 {
        // One internal diagonal between the first and last interior vertices.
        segs.push([4, 4 + n_interior - 1]);
    }
    (pts, segs)
}

fn run_c(points: &[[f64; 2]], segs: &[[usize; 2]]) -> ctriangle_sys::Output {
    let mut pointlist = Vec::new();
    for p in points {
        pointlist.push(p[0]);
        pointlist.push(p[1]);
    }
    let mut segmentlist = Vec::new();
    for s in segs {
        segmentlist.push(s[0] as i32);
        segmentlist.push(s[1] as i32);
    }
    let input = ctriangle_sys::Input {
        pointlist,
        segmentlist,
        ..Default::default()
    };
    ctriangle_sys::triangulate_safe("pznQ", &input)
}

#[test]
fn cdt_matches_c() {
    for &n in &[0usize, 1, 2, 10, 50, 200, 1000] {
        for seed in 0..3 {
            let (pts, segs) = build_case(n, seed * 31 + n as u64 + 1);
            let t = weka::cdt_pslg(&pts, None, &segs, None, false);
            let c = run_c(&pts, &segs);

            assert_eq!(
                t.triangles.len(),
                c.numberoftriangles,
                "triangle count (n={n}, seed={seed})"
            );
            assert_eq!(
                weka_tris(&t),
                c_tris(&c),
                "triangle set (n={n}, seed={seed})"
            );

            // Every triangle counterclockwise.
            for &[a, b, c2] in &t.triangles {
                assert!(orient2d(t.points[a], t.points[b], t.points[c2], false) > 0.0);
            }

            // Every input segment recovered as a mesh subsegment.
            let recovered: BTreeSet<_> = t
                .segments
                .iter()
                .map(|&[a, b]| seg_key(t.points[a], t.points[b]))
                .collect();
            for &[a, b] in &segs {
                assert!(
                    recovered.contains(&seg_key(pts[a], pts[b])),
                    "segment {a}->{b} not recovered (n={n}, seed={seed})"
                );
            }
        }
    }
}
