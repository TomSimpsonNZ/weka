//! Regression test for the subsegment flip-barrier bug: after quality
//! refinement of a two-region domain, every subsegment must be reciprocally
//! bonded and no cross-region edge may lack a subsegment barrier (which would
//! let attributes average across the boundary).

use weka::mesh::{Mesh, TriHandle};
use weka::rng::Rng;
use weka::{delaunay, holes, io_assembly, quality, segments, RegionSpec};

#[test]
fn refinement_preserves_segment_barriers() {
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

    let mut m = Mesh::new(0, 1, true, false);
    let mut rng = Rng::new();
    io_assembly::load_points(&mut m, &points, &[], 0, None);
    let hull = delaunay::delaunay(&mut m, &mut rng, true, true);
    m.hullsize = hull as i64;
    m.checksegments = true;
    segments::form_skeleton(&mut m, &mut rng, &seg, None, true, false);
    holes::carve_holes(
        &mut m,
        &mut rng,
        &[],
        &regions,
        false,
        false,
        true,
        false,
        0,
    );

    let mut q = quality::Quality::new(20.0, Some(0.05));
    quality::enforce_quality(&mut m, &mut q, &mut rng);

    assert!(
        m.subseg_problems().is_empty(),
        "subsegment bonds inconsistent: {:?}",
        m.subseg_problems()
    );

    // No cross-region adjacency may lack a subsegment barrier.
    let mut leaks = 0;
    for ti in m.live_triangles() {
        let a = m.elem_attr(TriHandle::new(ti as u32, 0), 0);
        for o in 0..3 {
            let h = TriHandle::new(ti as u32, o);
            let nb = m.sym(h);
            if !nb.is_dummy() && m.elem_attr(nb.with_orient(0), 0) != a && m.tspivot(h).is_dummy() {
                leaks += 1;
            }
        }
    }
    assert_eq!(leaks, 0, "cross-region edges without a subsegment barrier");
}
