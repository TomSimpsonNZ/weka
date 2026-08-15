//! Hole carving and region attribute/area spreading — port of triangle.cpp
//! `infecthull` (:12611), `plague` (:12693), `regionplague` (:12903), and
//! `carveholes` (:13016). The "virus" pool is a `Vec<TriHandle>` worklist.

use crate::mesh::{Mesh, TriHandle, VertexKind, NO_VERTEX};
use crate::rng::Rng;
use crate::segments::{locate, LocateResult};

/// Marks a segment-bounded area of a [`Pslg`](crate::Pslg) as a material region.
///
/// Place [`point`](RegionSpec::point) anywhere inside the region; every element
/// reachable from it (without crossing a segment) receives this region's
/// [`attribute`](RegionSpec::attribute) — reported in
/// [`Triangulation::triangle_attributes`](crate::Triangulation::triangle_attributes)
/// when [`region_attributes(true)`](crate::Triangulator::region_attributes) is
/// set — and, if given, its local [`area`](RegionSpec::area) limit.
///
/// ```
/// use weka::RegionSpec;
/// // A material "1" region seeded at (0.5, 0.5), with no local area cap.
/// let r = RegionSpec { point: [0.5, 0.5], attribute: 1.0, area: f64::NAN };
/// # let _ = r;
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RegionSpec {
    /// A point inside the region (used to seed the flood fill).
    pub point: [f64; 2],
    /// The attribute (e.g. material id) applied to every element of the region.
    pub attribute: f64,
    /// A maximum-area constraint local to this region (`f64::NAN` = unconstrained).
    pub area: f64,
}

/// Infect unprotected hull triangles (creating concavities); mark protected
/// hull subsegments. Port of `infecthull`.
fn infecthull(m: &mut Mesh, viri: &mut Vec<TriHandle>) {
    let mut hulltri = m.sym(TriHandle::new(0, 0));
    let starttri = hulltri;
    loop {
        if !m.infected(hulltri) {
            let hullsubseg = m.tspivot(hulltri);
            if hullsubseg.is_dummy() {
                m.infect(hulltri);
                viri.push(hulltri);
            } else if m.smark(hullsubseg) == 0 {
                m.set_smark(hullsubseg, 1);
                let horg = m.org(hulltri);
                let hdest = m.dest(hulltri);
                if m.vertex_mark(horg) == 0 {
                    m.set_vertex_mark(horg, 1);
                }
                if m.vertex_mark(hdest) == 0 {
                    m.set_vertex_mark(hdest, 1);
                }
            }
        }
        hulltri = m.lnext(hulltri);
        let mut nexttri = m.oprev(hulltri);
        while !nexttri.is_dummy() {
            hulltri = nexttri;
            nexttri = m.oprev(hulltri);
        }
        if hulltri == starttri {
            break;
        }
    }
}

/// Spread infection to unprotected neighbors and delete infected triangles.
/// Port of `plague`.
fn plague(m: &mut Mesh, viri: &mut Vec<TriHandle>) {
    // Pass 1: spread the infection. `viri` grows as we go (index walk).
    let mut i = 0;
    while i < viri.len() {
        let testtri = viri[i].with_orient(0);
        for orient in 0..3 {
            let t = testtri.with_orient(orient);
            let neighbor = m.sym(t);
            let neighborsubseg = m.tspivot(t);
            if neighbor.is_dummy() || m.infected(neighbor) {
                if !neighborsubseg.is_dummy() {
                    m.subseg_dealloc(neighborsubseg);
                    if !neighbor.is_dummy() {
                        m.tsdissolve(neighbor);
                    }
                }
            } else if neighborsubseg.is_dummy() {
                m.infect(neighbor);
                viri.push(neighbor);
            } else {
                m.stdissolve(neighborsubseg);
                if m.smark(neighborsubseg) == 0 {
                    m.set_smark(neighborsubseg, 1);
                }
                let norg = m.org(neighbor);
                let ndest = m.dest(neighbor);
                if m.vertex_mark(norg) == 0 {
                    m.set_vertex_mark(norg, 1);
                }
                if m.vertex_mark(ndest) == 0 {
                    m.set_vertex_mark(ndest, 1);
                }
            }
        }
        i += 1;
    }

    // Pass 2: detect orphaned vertices, then delete the infected triangles.
    for &v in viri.iter() {
        let testtri = v.with_orient(0);
        for orient in 0..3 {
            let t = testtri.with_orient(orient);
            let testvertex = m.org(t);
            if testvertex != NO_VERTEX {
                let mut killorg = true;
                m.set_org(t, NO_VERTEX);
                let mut neighbor = m.onext(t);
                while !neighbor.is_dummy() && neighbor != t {
                    if m.infected(neighbor) {
                        m.set_org(neighbor, NO_VERTEX);
                    } else {
                        killorg = false;
                    }
                    neighbor = m.onext(neighbor);
                }
                if neighbor.is_dummy() {
                    let mut nb = m.oprev(t);
                    while !nb.is_dummy() {
                        if m.infected(nb) {
                            m.set_org(nb, NO_VERTEX);
                        } else {
                            killorg = false;
                        }
                        nb = m.oprev(nb);
                    }
                }
                if killorg {
                    m.set_vertex_kind(testvertex, VertexKind::Undead);
                    m.undeads += 1;
                }
            }
        }
        for orient in 0..3 {
            let t = testtri.with_orient(orient);
            let neighbor = m.sym(t);
            if neighbor.is_dummy() {
                m.hullsize -= 1;
            } else {
                m.dissolve(neighbor);
                m.hullsize += 1;
            }
        }
        m.triangle_dealloc(testtri);
    }
    viri.clear();
}

/// Spread a region's attribute / area constraint through a segment-bounded
/// region. Port of `regionplague`.
fn regionplague(
    m: &mut Mesh,
    viri: &mut Vec<TriHandle>,
    attribute: f64,
    area: f64,
    regionattrib: bool,
    region_col: usize,
    vararea: bool,
) {
    let mut i = 0;
    while i < viri.len() {
        let testtri = viri[i].with_orient(0);
        if regionattrib {
            m.set_elem_attr(testtri, region_col, attribute);
        }
        if vararea {
            m.set_area_bound(testtri, area);
        }
        for orient in 0..3 {
            let t = testtri.with_orient(orient);
            let neighbor = m.sym(t);
            let neighborsubseg = m.tspivot(t);
            if !neighbor.is_dummy() && !m.infected(neighbor) && neighborsubseg.is_dummy() {
                m.infect(neighbor);
                viri.push(neighbor);
            }
        }
        i += 1;
    }
    for &v in viri.iter() {
        m.uninfect(v);
    }
    viri.clear();
}

/// Carve holes & concavities and spread region attributes/areas. Port of
/// `carveholes`. `region_col` is the triangle-attribute column that receives a
/// region's material id when `regionattrib` is set.
#[allow(clippy::too_many_arguments)]
pub fn carve_holes(
    m: &mut Mesh,
    rng: &mut Rng,
    holes: &[[f64; 2]],
    regions: &[RegionSpec],
    convex: bool,
    noholes: bool,
    regionattrib: bool,
    vararea: bool,
    region_col: usize,
) {
    let mut viri: Vec<TriHandle> = Vec::new();

    if !convex {
        infecthull(m, &mut viri);
    }

    if !holes.is_empty() && !noholes {
        for h in holes {
            if h[0] >= m.xmin && h[0] <= m.xmax && h[1] >= m.ymin && h[1] <= m.ymax {
                let mut searchtri = m.sym(TriHandle::new(0, 0));
                let searchorg = m.org(searchtri);
                let searchdest = m.dest(searchtri);
                if crate::predicates::orient2d(
                    m.point(searchorg),
                    m.point(searchdest),
                    *h,
                    m.noexact,
                ) > 0.0
                {
                    let intersect = locate(m, rng, *h, &mut searchtri);
                    if intersect != LocateResult::Outside && !m.infected(searchtri) {
                        m.infect(searchtri);
                        viri.push(searchtri);
                    }
                }
            }
        }
    }

    // Find region seed triangles before carving (locate needs a convex mesh).
    let mut regiontris: Vec<Option<TriHandle>> = vec![None; regions.len()];
    for (i, r) in regions.iter().enumerate() {
        let p = r.point;
        if p[0] >= m.xmin && p[0] <= m.xmax && p[1] >= m.ymin && p[1] <= m.ymax {
            let mut searchtri = m.sym(TriHandle::new(0, 0));
            let searchorg = m.org(searchtri);
            let searchdest = m.dest(searchtri);
            if crate::predicates::orient2d(m.point(searchorg), m.point(searchdest), p, m.noexact)
                > 0.0
            {
                let intersect = locate(m, rng, p, &mut searchtri);
                if intersect != LocateResult::Outside && !m.infected(searchtri) {
                    regiontris[i] = Some(searchtri);
                }
            }
        }
    }

    if !viri.is_empty() {
        plague(m, &mut viri);
    }

    if !regions.is_empty() {
        for (i, r) in regions.iter().enumerate() {
            if let Some(rt) = regiontris[i] {
                if !m.is_tri_dead(rt.index()) {
                    m.infect(rt);
                    viri.push(rt);
                    regionplague(
                        m,
                        &mut viri,
                        r.attribute,
                        r.area,
                        regionattrib,
                        region_col,
                        vararea,
                    );
                }
            }
        }
    }
}
