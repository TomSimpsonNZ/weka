//! End-to-end FEA meshing example using the idiomatic `Triangulator` builder.
//!
//! Meshes a rectangular plate with a square hole, split into two material
//! regions, refined to a minimum angle and maximum element area, with quadratic
//! (6-node) elements and a triangle neighbor list for assembly.
//!
//! Run with: `cargo run -p weka --example fea_mesh`

use weka::{Pslg, RegionSpec, Triangulator};

fn main() {
    // Outer boundary (a 4x2 plate), an internal divider at x=2 separating two
    // materials, and a square hole in the right material.
    let points = vec![
        [0.0, 0.0], // 0
        [2.0, 0.0], // 1  (divider bottom)
        [4.0, 0.0], // 2
        [4.0, 2.0], // 3
        [2.0, 2.0], // 4  (divider top)
        [0.0, 2.0], // 5
        // square hole corners (inside the right region)
        [2.8, 0.8], // 6
        [3.2, 0.8], // 7
        [3.2, 1.2], // 8
        [2.8, 1.2], // 9
    ];
    let segments = vec![
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 0], // outer boundary
        [1, 4], // internal material divider
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 6], // hole boundary
    ];
    let pslg = Pslg {
        points,
        segments,
        holes: vec![[3.0, 1.0]], // a point inside the square hole
        regions: vec![
            RegionSpec {
                point: [1.0, 1.0],
                attribute: 1.0,
                area: f64::NAN,
            }, // left material
            RegionSpec {
                point: [3.5, 0.3],
                attribute: 2.0,
                area: f64::NAN,
            }, // right material
        ],
        ..Default::default()
    };

    let mesh = Triangulator::new()
        .min_angle(28.0)
        .max_area(0.05)
        .region_attributes(true)
        .quadratic(true)
        .neighbors(true)
        .triangulate_pslg(&pslg)
        .expect("meshing failed");

    println!("FEA mesh generated:");
    println!("  nodes              : {}", mesh.points.len());
    println!("  elements           : {}", mesh.triangles.len());
    println!("  nodes per element  : {}", mesh.corners_per_triangle);
    println!("  boundary segments  : {}", mesh.segments.len());
    println!(
        "  has edge-midpoint nodes (quadratic): {}",
        mesh.edge_nodes.is_some()
    );
    println!(
        "  has neighbor adjacency             : {}",
        mesh.neighbors.is_some()
    );

    // Material breakdown from the region attribute column.
    let mut m1 = 0usize;
    let mut m2 = 0usize;
    for &a in &mesh.triangle_attributes {
        if a == 1.0 {
            m1 += 1;
        } else if a == 2.0 {
            m2 += 1;
        }
    }
    println!("  elements in material 1 / 2: {m1} / {m2}");
}
