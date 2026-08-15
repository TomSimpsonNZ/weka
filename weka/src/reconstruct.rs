//! Rebuild a mesh from an explicit triangle (and optional segment) list, for the
//! refine path (Triangle's `-r`). Port of the TRILIBRARY `reconstruct()`
//! (triangle.cpp:11108).
//!
//! Triangle temporarily overloads each triangle's subsegment slots to thread a
//! per-vertex linked list of incident triangles while it rediscovers shared
//! edges. We keep that linked list in a dedicated `next` array instead, so the
//! real subsegment side-array is untouched.

use crate::mesh::{Mesh, SubHandle, TriHandle, Vid};
use crate::segments::insert_subseg;

/// Reconstruct the mesh; returns the convex-hull edge count. Input vertices must
/// already be loaded into `m`.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct(
    m: &mut Mesh,
    trianglelist: &[[usize; 3]],
    tri_attrs: &[f64],
    num_tri_attrs: usize,
    area_constraints: Option<&[f64]>,
    segments: &[[usize; 2]],
    segment_markers: Option<&[i32]>,
    poly: bool,
) -> usize {
    let elements = trianglelist.len();
    let invertices = m.invertices;

    // Allocate all triangles (indices 1..=elements) and subsegments first.
    let tris: Vec<TriHandle> = (0..elements).map(|_| m.make_triangle()).collect();
    let subs: Vec<SubHandle> = if poly {
        (0..segments.len()).map(|_| m.make_subseg()).collect()
    } else {
        Vec::new()
    };

    // Per-vertex stack of incident triangles, and the "next" link for each
    // (triangle, orientation) occurrence (replacing Triangle's subseg-slot reuse).
    let mut vertexarray = vec![TriHandle::DUMMY; invertices];
    let mut next = vec![[TriHandle::DUMMY; 3]; m.tri_arena_len()];
    let mut hullsize = 0usize;

    // Assemble triangles and link those sharing an edge.
    for (e, &t) in tris.iter().enumerate() {
        let corner = trianglelist[e];
        for j in 0..num_tri_attrs {
            m.set_elem_attr(t, j, tri_attrs[e * num_tri_attrs + j]);
        }
        if let Some(areas) = area_constraints {
            m.set_area_bound(t, areas[e]);
        }
        m.set_org(t, corner[0] as Vid);
        m.set_dest(t, corner[1] as Vid);
        m.set_apex(t, corner[2] as Vid);

        for orient in 0..3 {
            let h = TriHandle::new(t.index() as u32, orient);
            let aroundvertex = corner[orient];
            let nexttri = vertexarray[aroundvertex];
            next[t.index()][orient] = nexttri;
            vertexarray[aroundvertex] = h;

            let mut checktri = nexttri;
            if !checktri.is_dummy() {
                let tdest = m.dest(h);
                let tapex = m.apex(h);
                loop {
                    let checkdest = m.dest(checktri);
                    let checkapex = m.apex(checktri);
                    if tapex == checkdest {
                        let triangleleft = m.lprev(h);
                        m.bond(triangleleft, checktri);
                    }
                    if tdest == checkapex {
                        let checkleft = m.lprev(checktri);
                        m.bond(h, checkleft);
                    }
                    let nt = next[checktri.index()][checktri.orient()];
                    checktri = nt;
                    if checktri.is_dummy() {
                        break;
                    }
                }
            }
        }
    }

    // Mark input segments and bond them to their triangles.
    if poly {
        for (sn, &sub) in subs.iter().enumerate() {
            let end = segments[sn];
            let boundmarker = segment_markers.map_or(0, |mk| mk[sn]);
            let s0 = SubHandle::new(sub.index() as u32, 0);
            let (so, sd) = (end[0] as Vid, end[1] as Vid);
            m.set_sorg(s0, so);
            m.set_sdest(s0, sd);
            m.set_seg_org(s0, so);
            m.set_seg_dest(s0, sd);
            m.set_smark(s0, boundmarker);

            for sso in 0..2 {
                let sloop = SubHandle::new(sub.index() as u32, sso);
                let aroundvertex = end[1 - sso];
                let shorg = m.sorg(sloop);

                // Walk the incident-triangle stack for `aroundvertex`, removing
                // the matching triangle when found.
                let mut prev_head = true;
                let mut prev_tri = TriHandle::DUMMY;
                let mut checktri = vertexarray[aroundvertex];
                let mut found = false;
                while !found && !checktri.is_dummy() {
                    let checkdest = m.dest(checktri);
                    let nx = next[checktri.index()][checktri.orient()];
                    if shorg == checkdest {
                        if prev_head {
                            vertexarray[aroundvertex] = nx;
                        } else {
                            next[prev_tri.index()][prev_tri.orient()] = nx;
                        }
                        m.tsbond(checktri, sloop);
                        if m.sym(checktri).is_dummy() {
                            insert_subseg(m, checktri, 1);
                            m.set_hull_edge(checktri); // point-location start edge
                            hullsize += 1;
                        }
                        found = true;
                    } else {
                        prev_head = false;
                        prev_tri = checktri;
                        checktri = nx;
                    }
                }
            }
        }
    }

    // Remaining stacked edges have no subsegment; dissolve and count hull edges.
    for &head in vertexarray.iter() {
        let mut checktri = head;
        while !checktri.is_dummy() {
            let nt = next[checktri.index()][checktri.orient()];
            m.tsdissolve(checktri);
            if m.sym(checktri).is_dummy() {
                insert_subseg(m, checktri, 1);
                m.set_hull_edge(checktri); // point-location start edge
                hullsize += 1;
            }
            checktri = nt;
        }
    }

    hullsize
}
