//! P6 quality-meshing tests. The hard FEA acceptance criteria are intrinsic
//! (every triangle meets the minimum angle and maximum area); we also sanity-
//! check the element count against the C library (allowing a tolerance, since
//! weka omits Chew's free-vertex deletion — see quality.rs).

use weka::predicates::orient2d;
use weka::{RegionSpec, Triangulation};

/// Minimum interior angle of a triangle, in degrees.
fn min_angle_deg(p: [[f64; 2]; 3]) -> f64 {
    let ang = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        let v1 = [b[0] - a[0], b[1] - a[1]];
        let v2 = [c[0] - a[0], c[1] - a[1]];
        let dot = v1[0] * v2[0] + v1[1] * v2[1];
        let cross = (v1[0] * v2[1] - v1[1] * v2[0]).abs();
        cross.atan2(dot).to_degrees()
    };
    ang(p[0], p[1], p[2])
        .min(ang(p[1], p[0], p[2]))
        .min(ang(p[2], p[0], p[1]))
}

fn area(p: [[f64; 2]; 3]) -> f64 {
    0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
        .abs()
}

fn tri_pts(t: &Triangulation, i: usize) -> [[f64; 2]; 3] {
    let [a, b, c] = t.triangles[i];
    [t.points[a], t.points[b], t.points[c]]
}

fn c_count(switches: &str, points: &[[f64; 2]], seg: &[[usize; 2]], holes: &[[f64; 2]]) -> usize {
    let mut pointlist = Vec::new();
    for p in points {
        pointlist.push(p[0]);
        pointlist.push(p[1]);
    }
    let mut segmentlist = Vec::new();
    for s in seg {
        segmentlist.push(s[0] as i32);
        segmentlist.push(s[1] as i32);
    }
    let mut holelist = Vec::new();
    for h in holes {
        holelist.push(h[0]);
        holelist.push(h[1]);
    }
    let input = ctriangle_sys::Input {
        pointlist,
        segmentlist,
        holelist,
        ..Default::default()
    };
    ctriangle_sys::triangulate_safe(switches, &input).numberoftriangles
}

#[test]
fn square_quality_meets_bounds() {
    // Unit square boundary; refine to min angle 20°, max area 0.01.
    let points = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let seg = vec![[0, 1], [1, 2], [2, 3], [3, 0]];
    let min_angle = 20.0;
    let max_area = 0.01;

    let t = weka::mesh_pslg_quality(
        &points,
        &[],
        0,
        None,
        &seg,
        None,
        &[],
        &[],
        false,
        false,
        min_angle,
        Some(max_area),
    );

    assert!(
        t.triangles.len() > 50,
        "expected a refined mesh, got {}",
        t.triangles.len()
    );
    for i in 0..t.triangles.len() {
        let p = tri_pts(&t, i);
        assert!(orient2d(p[0], p[1], p[2], false) > 0.0, "tri {i} not CCW");
        assert!(
            min_angle_deg(p) >= min_angle - 1e-4,
            "tri {i} angle {} < {min_angle}",
            min_angle_deg(p)
        );
        assert!(
            area(p) <= max_area * (1.0 + 1e-9),
            "tri {i} area {} > {max_area}",
            area(p)
        );
    }

    // Sanity vs C (counts won't match exactly; bound the difference).
    let c = c_count("pq20a0.01zQ", &points, &seg, &[]);
    let ratio = t.triangles.len() as f64 / c as f64;
    assert!(
        (0.8..=1.25).contains(&ratio),
        "weka {} vs C {} triangles (ratio {ratio:.3})",
        t.triangles.len(),
        c
    );
}

#[test]
fn square_min_angle_only_terminates_and_meets_angle() {
    let points = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let seg = vec![[0, 1], [1, 2], [2, 3], [3, 0]];
    let t = weka::mesh_pslg_quality(
        &points,
        &[],
        0,
        None,
        &seg,
        None,
        &[],
        &[],
        false,
        false,
        25.0,
        None,
    );
    for i in 0..t.triangles.len() {
        let p = tri_pts(&t, i);
        assert!(orient2d(p[0], p[1], p[2], false) > 0.0);
        assert!(
            min_angle_deg(p) >= 25.0 - 1e-4,
            "angle {}",
            min_angle_deg(p)
        );
    }
}

#[test]
fn quality_with_max_area_only() {
    // No angle bound, just an area cap — every triangle must respect it.
    let points = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]];
    let seg = vec![[0, 1], [1, 2], [2, 3], [3, 0]];
    let t = weka::mesh_pslg_quality(
        &points,
        &[],
        0,
        None,
        &seg,
        None,
        &[],
        &[],
        false,
        false,
        0.0,
        Some(0.05),
    );
    assert!(t.triangles.len() >= 40);
    for i in 0..t.triangles.len() {
        let p = tri_pts(&t, i);
        assert!(area(p) <= 0.05 * (1.0 + 1e-9), "area {}", area(p));
    }
}

#[test]
fn region_max_area_with_quality() {
    // Two material regions, refined with a global area cap; region ids preserved.
    let points = vec![
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];
    let seg = vec![[0, 4], [4, 1], [1, 2], [2, 5], [5, 3], [3, 0], [4, 5]];
    let regions = vec![
        RegionSpec {
            point: [0.5, 0.5],
            attribute: 1.0,
            area: f64::NAN,
        },
        RegionSpec {
            point: [1.5, 0.5],
            attribute: 2.0,
            area: f64::NAN,
        },
    ];
    // Min-angle + area refinement with two material regions.
    let t = weka::mesh_pslg_quality(
        &points,
        &[],
        0,
        None,
        &seg,
        None,
        &[],
        &regions,
        false,
        true,
        20.0,
        Some(0.05),
    );
    assert_eq!(t.triangle_attributes.len(), t.triangles.len());
    for i in 0..t.triangles.len() {
        let p = tri_pts(&t, i);
        assert!(
            min_angle_deg(p) >= 20.0 - 1e-4,
            "angle {}",
            min_angle_deg(p)
        );
        assert!(area(p) <= 0.05 * (1.0 + 1e-9));
        let id = t.triangle_attributes[i];
        assert!(id == 1.0 || id == 2.0, "unexpected region id {id}");
    }
}
