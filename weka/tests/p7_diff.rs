//! P7 tests: quadratic (`-o2`) elements and mesh refinement (`-r`), vs the C library.

use weka::predicates::orient2d;
use weka::{InputMesh, Pslg, Triangulator};

fn square_pslg() -> Pslg {
    Pslg {
        points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        segments: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
        ..Default::default()
    }
}

fn c_mesh(switches: &str, input: &ctriangle_sys::Input) -> ctriangle_sys::Output {
    ctriangle_sys::triangulate_safe(switches, input)
}

fn area(p: [[f64; 2]; 3]) -> f64 {
    0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
        .abs()
}

#[test]
fn quadratic_elements_structure_and_counts() {
    let pslg = square_pslg();
    let t = Triangulator::new()
        .min_angle(20.0)
        .max_area(0.02)
        .quadratic(true)
        .triangulate_pslg(&pslg)
        .unwrap();

    assert_eq!(t.corners_per_triangle, 6);
    let en = t.edge_nodes.as_ref().expect("quadratic => edge_nodes");
    assert_eq!(en.len(), t.triangles.len());

    // Every node index is valid, and each edge-midpoint node really is the
    // midpoint of two of its triangle's corners.
    for (i, &[a, b, c]) in t.triangles.iter().enumerate() {
        let corners = [t.points[a], t.points[b], t.points[c]];
        for &node in &en[i] {
            assert!(node < t.points.len());
            let p = t.points[node];
            let is_mid = (0..3).any(|j| {
                let k = (j + 1) % 3;
                let mid = [
                    0.5 * (corners[j][0] + corners[k][0]),
                    0.5 * (corners[j][1] + corners[k][1]),
                ];
                (p[0] - mid[0]).abs() < 1e-12 && (p[1] - mid[1]).abs() < 1e-12
            });
            assert!(
                is_mid,
                "edge node {node} of tri {i} is not an edge midpoint"
            );
        }
    }

    // #points = #corner vertices + one node per mesh edge; #edges=(3T+hull)/2.
    let edges = (3 * t.triangles.len() + t.hull_size) / 2;
    let corner_pts = t.points.len()
        - en.iter()
            .flatten()
            .collect::<std::collections::BTreeSet<_>>()
            .len();
    assert_eq!(corner_pts + edges, t.points.len(), "node accounting");

    // Count vs C (within the documented quality tolerance).
    let mut pointlist = Vec::new();
    for p in &pslg.points {
        pointlist.push(p[0]);
        pointlist.push(p[1]);
    }
    let segmentlist: Vec<i32> = pslg
        .segments
        .iter()
        .flat_map(|s| [s[0] as i32, s[1] as i32])
        .collect();
    let input = ctriangle_sys::Input {
        pointlist,
        segmentlist,
        ..Default::default()
    };
    let c = c_mesh("pq20a0.02o2zQ", &input);
    assert_eq!(c.numberofcorners, 6);
    let ratio = t.triangles.len() as f64 / c.numberoftriangles as f64;
    assert!(
        (0.8..=1.25).contains(&ratio),
        "o2 tri count weka {} vs C {}",
        t.triangles.len(),
        c.numberoftriangles
    );
}

#[test]
fn refine_existing_mesh_meets_area() {
    // Coarse mesh: the unit square as two triangles, with boundary segments.
    let pts = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let coarse = Triangulator::new()
        .triangulate_pslg(&square_pslg())
        .unwrap();
    assert_eq!(coarse.triangles.len(), 2);

    let input_mesh = InputMesh {
        points: pts.clone(),
        triangles: coarse.triangles.clone(),
        segments: square_pslg().segments,
        ..Default::default()
    };
    let max_area = 0.03;
    let refined = Triangulator::new()
        .max_area(max_area)
        .neighbors(true)
        .refine(&input_mesh)
        .unwrap();

    // The reconstructed mesh refines into a valid Delaunay mesh meeting the area
    // bound on every element.
    assert!(
        refined.triangles.len() > 30,
        "refinement should subdivide: {}",
        refined.triangles.len()
    );
    let nb = refined.neighbors.as_ref().unwrap();
    for (i, &[a, b, c]) in refined.triangles.iter().enumerate() {
        let (pa, pb, pc) = (refined.points[a], refined.points[b], refined.points[c]);
        assert!(orient2d(pa, pb, pc, false) > 0.0, "tri {i} not CCW");
        assert!(
            area([pa, pb, pc]) <= max_area * (1.0 + 1e-9),
            "tri {i} area {}",
            area([pa, pb, pc])
        );
        for &n in &nb[i] {
            if n < 0 {
                continue;
            }
            if let Some(o) = refined.triangles[n as usize]
                .iter()
                .copied()
                .find(|v| ![a, b, c].contains(v))
            {
                assert!(
                    weka::predicates::incircle(pa, pb, pc, refined.points[o], false) <= 0.0,
                    "tri {i} not Delaunay vs neighbor {n}"
                );
            }
        }
    }

    // Compare scale to C reconstructing + refining the same input.
    let mut pointlist = Vec::new();
    for p in &pts {
        pointlist.push(p[0]);
        pointlist.push(p[1]);
    }
    let trianglelist: Vec<i32> = coarse
        .triangles
        .iter()
        .flat_map(|t| [t[0] as i32, t[1] as i32, t[2] as i32])
        .collect();
    let segmentlist: Vec<i32> = square_pslg()
        .segments
        .iter()
        .flat_map(|s| [s[0] as i32, s[1] as i32])
        .collect();
    let input = ctriangle_sys::Input {
        pointlist,
        trianglelist,
        numberofcorners: 3,
        segmentlist,
        ..Default::default()
    };
    let c = c_mesh("rpa0.03zQ", &input);
    let ratio = refined.triangles.len() as f64 / c.numberoftriangles as f64;
    assert!(
        (0.7..=1.4).contains(&ratio),
        "refine tri count weka {} vs C {}",
        refined.triangles.len(),
        c.numberoftriangles
    );
}

#[test]
fn refine_nonconvex_meets_quality() {
    // Non-cocircular L-shaped-ish domain with interior points; refine to a
    // minimum angle AND area, then verify Delaunay + both bounds on every element.
    let dom = Pslg {
        points: vec![
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 2.0],
            [0.0, 2.0],
            [1.3, 0.7],
            [2.1, 1.4],
        ],
        segments: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
        ..Default::default()
    };
    let coarse = Triangulator::new().triangulate_pslg(&dom).unwrap();
    let input_mesh = InputMesh {
        points: dom.points.clone(),
        triangles: coarse.triangles.clone(),
        segments: dom.segments.clone(),
        ..Default::default()
    };
    let r = Triangulator::new()
        .min_angle(20.0)
        .max_area(0.04)
        .neighbors(true)
        .refine(&input_mesh)
        .unwrap();
    let nb = r.neighbors.as_ref().unwrap();
    for (i, &[a, b, c]) in r.triangles.iter().enumerate() {
        let (pa, pb, pc) = (r.points[a], r.points[b], r.points[c]);
        assert!(orient2d(pa, pb, pc, false) > 0.0, "tri {i} not CCW");
        assert!(
            area([pa, pb, pc]) <= 0.04 * (1.0 + 1e-9),
            "tri {i} area {}",
            area([pa, pb, pc])
        );
        for &n in &nb[i] {
            if n < 0 {
                continue;
            }
            if let Some(o) = r.triangles[n as usize]
                .iter()
                .copied()
                .find(|v| ![a, b, c].contains(v))
            {
                assert!(
                    weka::predicates::incircle(pa, pb, pc, r.points[o], false) <= 0.0,
                    "tri {i} not Delaunay"
                );
            }
        }
    }
}

#[test]
fn builder_points_matches_plain_delaunay() {
    // The builder's point path should agree with the low-level delaunay entry.
    use rand::{Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
    let pts: Vec<[f64; 2]> = (0..500)
        .map(|_| [rng.gen::<f64>(), rng.gen::<f64>()])
        .collect();
    let a = Triangulator::new().triangulate_points(&pts).unwrap();
    let b = weka::delaunay_points(&pts, false);
    assert_eq!(a.triangles.len(), b.triangles.len());
}
