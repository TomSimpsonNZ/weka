//! P2 milestone: a hand-built mesh exercising the arena, navigation primitives,
//! and free-list passes the consistency checker.

use weka::mesh::{Mesh, TriHandle, VertexKind};

/// Build the triangulation of a unit square as two triangles sharing the
/// diagonal (0,0)-(1,1), wire up all bonds by hand, and check consistency.
fn build_square() -> Mesh {
    let mut m = Mesh::new(0, 0, false, false);
    let v00 = m.add_vertex([0.0, 0.0], &[], 0, VertexKind::Input);
    let v10 = m.add_vertex([1.0, 0.0], &[], 0, VertexKind::Input);
    let v11 = m.add_vertex([1.0, 1.0], &[], 0, VertexKind::Input);
    let v01 = m.add_vertex([0.0, 1.0], &[], 0, VertexKind::Input);

    // Triangle A: (v00, v10, v11) ccw. Triangle B: (v00, v11, v01) ccw.
    let a = m.make_triangle();
    let b = m.make_triangle();

    // Set corners so that org/dest/apex are consistent for orientation 0.
    // For orient 0: org = verts[1], dest = verts[2], apex = verts[0].
    m.set_org(a, v00);
    m.set_dest(a, v10);
    m.set_apex(a, v11);

    m.set_org(b, v00);
    m.set_dest(b, v11);
    m.set_apex(b, v01);

    // Shared edge is v00->v11 (a's edge from org v00 to apex... ) — bond the
    // edge of A whose endpoints are {v00, v11} to the matching edge of B.
    // Edge `orient` of a triangle runs from org(orient) to dest(orient).
    let ea = find_edge(&m, a, v11, v00).expect("A has edge v11->v00");
    let eb = find_edge(&m, b, v00, v11).expect("B has edge v00->v11");
    m.bond(ea, eb);

    m
}

/// Find the oriented edge of triangle `t` whose (org,dest) == (o,d).
fn find_edge(m: &Mesh, t: TriHandle, o: u32, d: u32) -> Option<TriHandle> {
    for orient in 0..3 {
        let h = TriHandle::new(t.index() as u32, orient);
        if m.org(h) == o && m.dest(h) == d {
            return Some(h);
        }
    }
    None
}

#[test]
fn hand_built_square_is_consistent() {
    let m = build_square();
    assert_eq!(m.num_triangles(), 2);
    assert_eq!(m.num_vertices(), 4);
    assert_eq!(m.check_mesh(), 0, "mesh should be topologically consistent");
}

#[test]
fn navigation_roundtrips() {
    let m = build_square();
    // sym(sym(h)) == h for the bonded interior edge.
    let a = TriHandle::new(1, 0);
    for orient in 0..3 {
        let h = TriHandle::new(a.index() as u32, orient);
        let s = m.sym(h);
        if !s.is_dummy() {
            assert_eq!(m.sym(s), h, "sym is an involution on interior edges");
        }
    }
    // lnext three times returns to start; lnext/lprev are inverses.
    let h = TriHandle::new(1, 0);
    assert_eq!(m.lnext(m.lnext(m.lnext(h))), h);
    assert_eq!(m.lprev(m.lnext(h)), h);
}

#[test]
fn free_list_reuses_slots_lifo() {
    let mut m = Mesh::new(0, 0, false, false);
    let t1 = m.make_triangle();
    let t2 = m.make_triangle();
    assert_eq!(m.num_triangles(), 2);
    m.triangle_dealloc(t2);
    m.triangle_dealloc(t1);
    assert_eq!(m.num_triangles(), 0);
    // LIFO: the most recently freed slot (t1) is handed out first.
    let t3 = m.make_triangle();
    assert_eq!(t3.index(), t1.index());
    let t4 = m.make_triangle();
    assert_eq!(t4.index(), t2.index());
    assert_eq!(m.num_triangles(), 2);
}
