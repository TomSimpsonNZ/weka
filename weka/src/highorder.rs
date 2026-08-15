//! Quadratic (second-order) elements — port of `highorder` (triangle.cpp:13747).
//!
//! Adds one vertex at the midpoint of every mesh edge and records, per triangle,
//! the three edge-midpoint nodes. The output assembler emits these as the extra
//! 3 nodes of each 6-node element.

use crate::mesh::{Mesh, TriHandle, VertexKind, Vid, NO_VERTEX};

/// Create edge-midpoint nodes for quadratic elements. Must run last (after all
/// other vertices/triangles are final).
pub fn high_order(m: &mut Mesh) {
    // Don't reuse dead vertex slots, so corner vertices keep lower output
    // indices than the new midpoint nodes (matches Triangle).
    m.clear_vertex_freelist();

    let arena = m.tri_arena_len();
    let mut high = vec![[NO_VERTEX; 3]; arena];
    let na = m.nextras;

    let live: Vec<usize> = m.live_triangles().collect();
    for ti in live {
        for orient in 0..3 {
            let h = TriHandle::new(ti as u32, orient);
            let trisym = m.sym(h);
            // Visit each edge once: when this triangle "owns" it (lower index, or
            // the edge is on the boundary).
            if trisym.is_dummy() || ti < trisym.index() {
                let torg = m.org(h);
                let tdest = m.dest(h);
                let po = m.point(torg);
                let pd = m.point(tdest);
                let mid = [0.5 * (po[0] + pd[0]), 0.5 * (po[1] + pd[1])];

                let mut attrs = vec![0.0; na];
                for (i, a) in attrs.iter_mut().enumerate() {
                    *a = 0.5 * (m.vertex_attrs(torg)[i] + m.vertex_attrs(tdest)[i]);
                }

                // Marker: 1 on the outer boundary, else 0; a subsegment edge
                // passes its own marker to the new node.
                let on_boundary = trisym.is_dummy();
                let mut mark = i32::from(on_boundary);
                if m.use_segments {
                    let s = m.tspivot(h);
                    if !s.is_dummy() {
                        mark = m.smark(s);
                    }
                }
                let v: Vid = m.add_vertex(mid, &attrs, mark, VertexKind::Free);
                high[ti][orient] = v;
                if !trisym.is_dummy() {
                    high[trisym.index()][trisym.orient()] = v;
                }
            }
        }
    }
    m.set_tri_high(high);
}
