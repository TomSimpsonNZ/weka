use weka::mesh::Mesh;
use weka::rng::Rng;
use weka::{holes, io_assembly, reconstruct, Pslg, Triangulator};

#[test]
fn reconstruct_is_consistent() {
    let coarse = Triangulator::new()
        .triangulate_pslg(&Pslg {
            points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            segments: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
            ..Default::default()
        })
        .unwrap();
    eprintln!("coarse tris: {:?}", coarse.triangles);

    let pts = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let segs = vec![[0usize, 1], [1, 2], [2, 3], [3, 0]];

    let mut m = Mesh::new(0, 0, true, false);
    let mut rng = Rng::new();
    io_assembly::load_points(&mut m, &pts, &[], 0, None);
    m.checksegments = true;
    let hull = reconstruct::reconstruct(&mut m, &coarse.triangles, &[], 0, None, &segs, None, true);
    m.hullsize = hull as i64;
    eprintln!(
        "after reconstruct: tris={} subsegs={} hull={} checkmesh={} nondelaunay={}",
        m.num_triangles(),
        m.live_subsegs().count(),
        hull,
        m.check_mesh(),
        m.count_non_delaunay(),
    );
    assert_eq!(m.check_mesh(), 0, "reconstructed mesh inconsistent");
    {
        use weka::mesh::TriHandle;
        let mut sym_dummy = 0;
        for ti in m.live_triangles() {
            for o in 0..3 {
                if m.sym(TriHandle::new(ti as u32, o)).is_dummy() {
                    sym_dummy += 1;
                }
            }
        }
        assert_eq!(
            sym_dummy as i64, hull as i64,
            "all interior edges must be bonded"
        );
    }
    assert_eq!(
        m.count_non_delaunay(),
        0,
        "reconstructed mesh must be Delaunay"
    );

    holes::carve_holes(&mut m, &mut rng, &[], &[], false, false, false, false, 0);
    eprintln!(
        "after carve: tris={} checkmesh={}",
        m.num_triangles(),
        m.check_mesh()
    );
    assert_eq!(m.check_mesh(), 0, "post-carve inconsistent");
    assert_eq!(m.num_triangles(), 2, "should still be 2 triangles");
}
