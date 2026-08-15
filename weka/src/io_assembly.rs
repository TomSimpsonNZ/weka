//! Loading input into the mesh and assembling output arrays — the library-mode
//! counterparts of Triangle's `transfernodes` (:14120) and the `write*` routines.

use crate::mesh::{Mesh, SubHandle, TriHandle, VertexKind, Vid};
use crate::output::Triangulation;

/// Load input points (and optional attributes / markers) into the mesh, matching
/// `transfernodes`. Vertices receive ids `0..points.len()` in input order.
pub fn load_points(
    m: &mut Mesh,
    points: &[[f64; 2]],
    attrs: &[f64],
    num_attrs: usize,
    markers: Option<&[i32]>,
) {
    debug_assert_eq!(m.nextras, num_attrs);
    m.reserve(points.len());
    for (i, &xy) in points.iter().enumerate() {
        let a = &attrs[i * num_attrs..i * num_attrs + num_attrs];
        let mark = markers.map_or(0, |mk| mk[i]);
        m.add_vertex(xy, a, mark, VertexKind::Input);
    }
    m.invertices = points.len();

    // Bounding box (Triangle's transfernodes computes this for hole/region
    // location and the sweepline; we use it for the hole-in-bounds test).
    if let Some(&first) = points.first() {
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (first[0], first[0], first[1], first[1]);
        for p in points {
            xmin = xmin.min(p[0]);
            xmax = xmax.max(p[0]);
            ymin = ymin.min(p[1]);
            ymax = ymax.max(p[1]);
        }
        m.xmin = xmin;
        m.xmax = xmax;
        m.ymin = ymin;
        m.ymax = ymax;
    }
}

/// Assemble a [`Triangulation`] from the meshed arena. Output point indices equal
/// input vertex ids (no jettison / Steiner points in plain Delaunay), and each
/// triangle is emitted as `[org, dest, apex]` at orientation 0.
pub fn assemble(
    m: &Mesh,
    hull_size: usize,
    want_neighbors: bool,
    want_segments: bool,
) -> Triangulation {
    // Output points: every live vertex (input + Steiner), assigned a contiguous
    // output index. `vert_out[v]` maps a vertex id to its output index.
    let na = m.nextras;
    let mut vert_out = vec![u32::MAX; m.vert_arena_len()];
    let mut points: Vec<[f64; 2]> = Vec::with_capacity(m.num_vertices());
    let mut point_markers: Vec<i32> = Vec::with_capacity(m.num_vertices());
    let mut point_attributes = Vec::with_capacity(m.num_vertices() * na);
    for v in 0..m.vert_arena_len() as u32 {
        if m.vertex_is_dead(v) {
            continue;
        }
        vert_out[v as usize] = points.len() as u32;
        points.push(m.point(v));
        point_markers.push(m.vertex_mark(v));
        point_attributes.extend_from_slice(m.vertex_attrs(v));
    }
    let vout = |v: Vid| vert_out[v as usize] as usize;

    // The neighbor list needs a handle→output-index remap; nothing else does, so
    // only build it (in a first pass) when neighbors are requested.
    let ntri = m.num_triangles();
    let tri_index = if want_neighbors {
        let mut idx = vec![-1i32; m.tri_arena_len()];
        for (k, ti) in m.live_triangles().enumerate() {
            idx[ti] = k as i32;
        }
        idx
    } else {
        Vec::new()
    };

    let ea = m.eextras;
    let mut triangles = Vec::with_capacity(ntri);
    let mut triangle_attributes = Vec::with_capacity(ntri * ea);
    let mut neighbors = if want_neighbors {
        Some(Vec::with_capacity(ntri))
    } else {
        None
    };
    let high = m.has_high_order();
    let mut edge_nodes = if high {
        Some(Vec::with_capacity(ntri))
    } else {
        None
    };

    for ti in m.live_triangles() {
        let h = TriHandle::new(ti as u32, 0);
        let (o, d, a) = (m.org(h), m.dest(h), m.apex(h));
        triangles.push([vout(o), vout(d), vout(a)]);
        for i in 0..ea {
            triangle_attributes.push(m.elem_attr(h, i));
        }
        if let Some(en) = edge_nodes.as_mut() {
            // Triangle's order: midpoints on edges 1, 2, 0.
            let m1 = vout(m.high_node(TriHandle::new(ti as u32, 1)));
            let m2 = vout(m.high_node(TriHandle::new(ti as u32, 2)));
            let m0 = vout(m.high_node(TriHandle::new(ti as u32, 0)));
            en.push([m1, m2, m0]);
        }
        if let Some(n) = neighbors.as_mut() {
            let mut row = [-1i32; 3];
            for (orient, slot) in row.iter_mut().enumerate() {
                let edge = TriHandle::new(ti as u32, orient);
                let s = m.sym(edge);
                *slot = if s.is_dummy() { -1 } else { tri_index[s.index()] };
            }
            n.push(row);
        }
    }

    // Subsegments → output segments (each emitted once, at orientation 0).
    let mut segments = Vec::new();
    let mut segment_markers = Vec::new();
    if want_segments {
        for si in m.live_subsegs() {
            let s = SubHandle::new(si as u32, 0);
            segments.push([vout(m.sorg(s)), vout(m.sdest(s))]);
            segment_markers.push(m.smark(s));
        }
    }

    Triangulation {
        points,
        point_attributes,
        point_markers,
        triangles,
        corners_per_triangle: if high { 6 } else { 3 },
        edge_nodes,
        triangle_attributes,
        neighbors,
        segments,
        segment_markers,
        hull_size,
    }
}

/// Convenience for callers that only need the triangle corner list.
pub fn triangle_corner_points(m: &Mesh, h: TriHandle) -> [Vid; 3] {
    [m.org(h), m.dest(h), m.apex(h)]
}
