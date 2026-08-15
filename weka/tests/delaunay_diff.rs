//! P3 differential tests: weka's divide-and-conquer Delaunay vs the C library.
//!
//! For general-position inputs the Delaunay triangulation is unique, so we assert
//! the *canonical set of coordinate-triples* matches C exactly. For degenerate
//! (cocircular) grids the triangulation is not unique, so we assert matching
//! counts plus intrinsic validity (CCW + empty-circumcircle).

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use weka::predicates::{incircle, orient2d};
use weka::Triangulation;

/// A coordinate triple, canonicalized (sorted) so triangle identity is
/// independent of corner order and vertex numbering.
fn tri_key(p: [[f64; 2]; 3]) -> [[u64; 2]; 3] {
    let mut k = [
        [p[0][0].to_bits(), p[0][1].to_bits()],
        [p[1][0].to_bits(), p[1][1].to_bits()],
        [p[2][0].to_bits(), p[2][1].to_bits()],
    ];
    k.sort_unstable();
    k
}

fn weka_set(t: &Triangulation) -> BTreeSet<[[u64; 2]; 3]> {
    t.triangles
        .iter()
        .map(|&[a, b, c]| tri_key([t.points[a], t.points[b], t.points[c]]))
        .collect()
}

fn c_set(out: &ctriangle_sys::Output) -> BTreeSet<[[u64; 2]; 3]> {
    let pt = |i: i32| {
        let i = i as usize;
        [out.pointlist[2 * i], out.pointlist[2 * i + 1]]
    };
    out.trianglelist
        .chunks_exact(3)
        .map(|c| tri_key([pt(c[0]), pt(c[1]), pt(c[2])]))
        .collect()
}

fn run_c(points: &[[f64; 2]]) -> ctriangle_sys::Output {
    let mut pointlist = Vec::with_capacity(points.len() * 2);
    for p in points {
        pointlist.push(p[0]);
        pointlist.push(p[1]);
    }
    let input = ctriangle_sys::Input {
        pointlist,
        ..Default::default()
    };
    // z = zero-based, Q = quiet, n = neighbors (forces hull bookkeeping too).
    ctriangle_sys::triangulate_safe("zQn", &input)
}

/// Assert every weka triangle is CCW and locally Delaunay (empty circumcircle).
fn assert_delaunay_valid(t: &Triangulation) {
    let nb = t.neighbors.as_ref().expect("neighbors requested");
    for (i, &[a, b, c]) in t.triangles.iter().enumerate() {
        let (pa, pb, pc) = (t.points[a], t.points[b], t.points[c]);
        assert!(
            orient2d(pa, pb, pc, false) > 0.0,
            "triangle {i} is not counterclockwise"
        );
        for &n in &nb[i] {
            if n < 0 {
                continue;
            }
            let opp = opposite_vertex(t, n as usize, [a, b, c]);
            if let Some(o) = opp {
                assert!(
                    incircle(pa, pb, pc, t.points[o], false) <= 0.0,
                    "triangle {i} fails empty-circumcircle vs neighbor {n}"
                );
            }
        }
    }
}

/// The corner of triangle `j` that is not shared with `shared` (by index).
fn opposite_vertex(t: &Triangulation, j: usize, shared: [usize; 3]) -> Option<usize> {
    t.triangles[j].iter().copied().find(|v| !shared.contains(v))
}

fn random_points(n: usize, seed: u64) -> Vec<[f64; 2]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| [rng.gen::<f64>(), rng.gen::<f64>()])
        .collect()
}

#[test]
fn matches_c_on_random_pointsets() {
    for &n in &[3usize, 4, 5, 10, 50, 100, 500, 1000, 5000] {
        for seed in 0..3 {
            let pts = random_points(n, seed * 100 + n as u64);
            let t = weka::delaunay_points(&pts, true);
            let c = run_c(&pts);
            assert_eq!(
                t.triangles.len(),
                c.numberoftriangles,
                "triangle count mismatch (n={n}, seed={seed})"
            );
            assert_eq!(
                weka_set(&t),
                c_set(&c),
                "canonical triangle set mismatch (n={n}, seed={seed})"
            );
            assert_delaunay_valid(&t);
        }
    }
}

#[test]
fn counts_and_validity_on_grids() {
    for &side in &[2usize, 3, 5, 8, 12] {
        let mut pts = Vec::new();
        for i in 0..side {
            for j in 0..side {
                pts.push([i as f64, j as f64]);
            }
        }
        let t = weka::delaunay_points(&pts, true);
        let c = run_c(&pts);
        // Counts are invariant across valid Delaunay triangulations of the same
        // point set, even when the set of triangles differs (cocircular grids).
        assert_eq!(
            t.triangles.len(),
            c.numberoftriangles,
            "grid {side}x{side} triangle count mismatch"
        );
        assert_eq!(t.points.len(), c.numberofpoints);
        assert_delaunay_valid(&t);
    }
}

#[test]
fn hull_size_matches_c() {
    let pts = random_points(200, 12345);
    let t = weka::delaunay_points(&pts, true);
    let c = run_c(&pts);
    // With no PSLG, Triangle reports the hull edge count as numberofsegments.
    assert_eq!(t.hull_size, c.numberofsegments, "hull size mismatch");
}

#[test]
fn matches_c_at_scale() {
    for &n in &[20_000usize, 100_000] {
        let pts = random_points(n, 99);
        let t = weka::delaunay_points(&pts, false);
        let c = run_c(&pts);
        assert_eq!(
            t.triangles.len(),
            c.numberoftriangles,
            "count mismatch n={n}"
        );
        assert_eq!(weka_set(&t), c_set(&c), "set mismatch n={n}");
    }
}
