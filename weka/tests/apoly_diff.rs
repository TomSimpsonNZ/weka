//! P5 differential test: the bundled `A.poly` PSLG (the letter "A" with a
//! triangular hole) meshed by weka vs the C library (`pzQ`). Exercises segment
//! recovery, concavity carving (`infecthull`), and hole carving.

use std::collections::BTreeSet;
use weka::predicates::orient2d;

const A_POLY: &str = include_str!("../../triangle/A.poly");

struct Pslg {
    points: Vec<[f64; 2]>,
    attrs: Vec<f64>,
    nattr: usize,
    segments: Vec<[usize; 2]>,
    holes: Vec<[f64; 2]>,
}

/// Parse a `.poly` file (1-based indices) into 0-based arrays.
fn parse_poly(s: &str) -> Pslg {
    let mut toks = s.split_whitespace().map(|t| t.to_string());
    let mut next = || toks.next().unwrap();
    let nverts: usize = next().parse().unwrap();
    let _dim: usize = next().parse().unwrap();
    let nattr: usize = next().parse().unwrap();
    let nmark: usize = next().parse().unwrap();

    let mut points = Vec::with_capacity(nverts);
    let mut attrs = Vec::with_capacity(nverts * nattr);
    for _ in 0..nverts {
        let _idx: i64 = next().parse().unwrap();
        let x: f64 = next().parse().unwrap();
        let y: f64 = next().parse().unwrap();
        points.push([x, y]);
        for _ in 0..nattr {
            attrs.push(next().parse::<f64>().unwrap());
        }
        for _ in 0..nmark {
            let _m: i64 = next().parse().unwrap();
        }
    }

    let nseg: usize = next().parse().unwrap();
    let segmark: usize = next().parse().unwrap();
    let mut segments = Vec::with_capacity(nseg);
    for _ in 0..nseg {
        let _idx: i64 = next().parse().unwrap();
        let e1: usize = next().parse().unwrap();
        let e2: usize = next().parse().unwrap();
        segments.push([e1 - 1, e2 - 1]); // 1-based -> 0-based
        for _ in 0..segmark {
            let _m: i64 = next().parse().unwrap();
        }
    }

    let nholes: usize = next().parse().unwrap_or(0);
    let mut holes = Vec::with_capacity(nholes);
    for _ in 0..nholes {
        let _idx: i64 = next().parse().unwrap();
        let hx: f64 = next().parse().unwrap();
        let hy: f64 = next().parse().unwrap();
        holes.push([hx, hy]);
    }

    Pslg {
        points,
        attrs,
        nattr,
        segments,
        holes,
    }
}

fn tri_key(p: [[f64; 2]; 3]) -> [[u64; 2]; 3] {
    let mut k = [
        [p[0][0].to_bits(), p[0][1].to_bits()],
        [p[1][0].to_bits(), p[1][1].to_bits()],
        [p[2][0].to_bits(), p[2][1].to_bits()],
    ];
    k.sort_unstable();
    k
}

#[test]
fn apoly_matches_c() {
    let p = parse_poly(A_POLY);

    // weka
    let t = weka::mesh_pslg(
        &p.points,
        &p.attrs,
        p.nattr,
        None,
        &p.segments,
        None,
        &p.holes,
        &[],
        false,
        false,
    );

    // C library
    let mut pointlist = Vec::new();
    for q in &p.points {
        pointlist.push(q[0]);
        pointlist.push(q[1]);
    }
    let mut segmentlist = Vec::new();
    for s in &p.segments {
        segmentlist.push(s[0] as i32);
        segmentlist.push(s[1] as i32);
    }
    let mut holelist = Vec::new();
    for h in &p.holes {
        holelist.push(h[0]);
        holelist.push(h[1]);
    }
    let input = ctriangle_sys::Input {
        pointlist,
        pointattributelist: p.attrs.clone(),
        numberofpointattributes: p.nattr as i32,
        segmentlist,
        holelist,
        ..Default::default()
    };
    let c = ctriangle_sys::triangulate_safe("pzQ", &input);

    // Triangle count must match exactly after carving.
    assert_eq!(
        t.triangles.len(),
        c.numberoftriangles,
        "A.poly triangle count: weka {} vs C {}",
        t.triangles.len(),
        c.numberoftriangles
    );

    // Canonical triangle set must match (CDT is unique in general position).
    let weka_set: BTreeSet<_> = t
        .triangles
        .iter()
        .map(|&[a, b, c]| tri_key([t.points[a], t.points[b], t.points[c]]))
        .collect();
    let pt = |i: i32| {
        let i = i as usize;
        [c.pointlist[2 * i], c.pointlist[2 * i + 1]]
    };
    let c_set: BTreeSet<_> = c
        .trianglelist
        .chunks_exact(3)
        .map(|w| tri_key([pt(w[0]), pt(w[1]), pt(w[2])]))
        .collect();
    assert_eq!(weka_set, c_set, "A.poly triangle set mismatch");

    // All triangles counterclockwise.
    for &[a, b, cc] in &t.triangles {
        assert!(orient2d(t.points[a], t.points[b], t.points[cc], false) > 0.0);
    }

    // Segment count matches (boundary recovered, no Steiner points added).
    assert_eq!(
        t.segments.len(),
        c.numberofsegments,
        "A.poly segment count mismatch"
    );
}

#[test]
fn apoly_quality_matches_c_scale() {
    // Quality-mesh A.poly (min angle 20°) and compare element count + validity
    // to C. A.poly has small input angles, so some triangles legitimately stay
    // below 20° (Triangle's MPW rule leaves them); we therefore check the count
    // is within tolerance of C and that the mesh is valid, rather than a strict
    // angle bound on every element.
    let p = parse_poly(A_POLY);
    let t = weka::mesh_pslg_quality(
        &p.points,
        &p.attrs,
        p.nattr,
        None,
        &p.segments,
        None,
        &p.holes,
        &[],
        false,
        false,
        20.0,
        None,
    );

    let mut pointlist = Vec::new();
    for q in &p.points {
        pointlist.push(q[0]);
        pointlist.push(q[1]);
    }
    let mut segmentlist = Vec::new();
    for s in &p.segments {
        segmentlist.push(s[0] as i32);
        segmentlist.push(s[1] as i32);
    }
    let mut holelist = Vec::new();
    for h in &p.holes {
        holelist.push(h[0]);
        holelist.push(h[1]);
    }
    let input = ctriangle_sys::Input {
        pointlist,
        pointattributelist: p.attrs.clone(),
        numberofpointattributes: p.nattr as i32,
        segmentlist,
        holelist,
        ..Default::default()
    };
    let c = ctriangle_sys::triangulate_safe("pq20zQ", &input);

    // All output triangles counterclockwise (valid mesh).
    for &[a, b, cc] in &t.triangles {
        assert!(orient2d(t.points[a], t.points[b], t.points[cc], false) > 0.0);
    }
    // Element count within tolerance of C (refinement strategies differ slightly).
    let ratio = t.triangles.len() as f64 / c.numberoftriangles as f64;
    assert!(
        (0.7..=1.4).contains(&ratio),
        "A.poly q20: weka {} vs C {} triangles (ratio {ratio:.3})",
        t.triangles.len(),
        c.numberoftriangles
    );
}
