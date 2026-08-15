//! Constrained Delaunay: segment recovery and convex-hull marking. Port of
//! triangle.cpp `flip` (:7927), `insertsubseg` (:7823), `finddirection`
//! (:11608), `scoutsegment` (:11850), `delaunayfixup` (:12064),
//! `constrainededge` (:12184), `insertsegment` (:12286), `markhull` (:12398),
//! `formskeleton` (:12444), `makevertexmap` (:7413), `locate`/`preciselocate`.
//!
//! `segmentintersection` (for input segments that cross each other, rather than
//! meeting only at shared endpoints) is not implemented; such input currently
//! panics with a clear message.

use crate::mesh::{Mesh, TriHandle, Vid};
use crate::rng::Rng;

const SAMPLEFACTOR: usize = 11;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LocateResult {
    InTriangle,
    OnEdge,
    OnVertex,
    Outside,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FindDir {
    Within,
    LeftCollinear,
    RightCollinear,
}

/// Map every vertex to one incident triangle (Triangle's `makevertexmap`).
pub fn make_vertex_map(m: &mut Mesh) {
    let arena = m.tri_arena_len();
    for ti in 1..arena {
        if m.is_tri_dead(ti) {
            continue;
        }
        for orient in 0..3 {
            let h = TriHandle::new(ti as u32, orient);
            let triorg = m.org(h);
            m.set_vertex2tri(triorg, h);
        }
    }
}

// ---------------------------------------------------------------------------
// flip (triangle.cpp:7927)
// ---------------------------------------------------------------------------

/// Flip the edge held by `flipedge` counterclockwise within its quadrilateral.
/// The handle's (triangle, orientation) is unchanged; only vertices/bonds move.
pub fn flip(m: &mut Mesh, flipedge: TriHandle) {
    let rightvertex = m.org(flipedge);
    let leftvertex = m.dest(flipedge);
    let botvertex = m.apex(flipedge);
    let top = m.sym(flipedge);
    let farvertex = m.apex(top);

    let topleft = m.lprev(top);
    let toplcasing = m.sym(topleft);
    let topright = m.lnext(top);
    let toprcasing = m.sym(topright);
    let botleft = m.lnext(flipedge);
    let botlcasing = m.sym(botleft);
    let botright = m.lprev(flipedge);
    let botrcasing = m.sym(botright);

    m.bond(topleft, botlcasing);
    m.bond(botleft, botrcasing);
    m.bond(botright, toprcasing);
    m.bond(topright, toplcasing);

    if m.checksegments {
        let toplsubseg = m.tspivot(topleft);
        let botlsubseg = m.tspivot(botleft);
        let botrsubseg = m.tspivot(botright);
        let toprsubseg = m.tspivot(topright);
        if toplsubseg.is_dummy() {
            m.tsdissolve(topright);
        } else {
            m.tsbond(topright, toplsubseg);
        }
        if botlsubseg.is_dummy() {
            m.tsdissolve(topleft);
        } else {
            m.tsbond(topleft, botlsubseg);
        }
        if botrsubseg.is_dummy() {
            m.tsdissolve(botleft);
        } else {
            m.tsbond(botleft, botrsubseg);
        }
        if toprsubseg.is_dummy() {
            m.tsdissolve(botright);
        } else {
            m.tsbond(botright, toprsubseg);
        }
    }

    m.set_org(flipedge, farvertex);
    m.set_dest(flipedge, botvertex);
    m.set_apex(flipedge, rightvertex);
    m.set_org(top, botvertex);
    m.set_dest(top, farvertex);
    m.set_apex(top, leftvertex);
}

/// Flip the edge held by `flipedge` clockwise — the inverse of [`flip`]. Port of
/// `unflip` (triangle.cpp:8062). Used by `undovertex` to reverse insertions.
pub fn unflip(m: &mut Mesh, flipedge: TriHandle) {
    let rightvertex = m.org(flipedge);
    let leftvertex = m.dest(flipedge);
    let botvertex = m.apex(flipedge);
    let top = m.sym(flipedge);
    let farvertex = m.apex(top);

    let topleft = m.lprev(top);
    let toplcasing = m.sym(topleft);
    let topright = m.lnext(top);
    let toprcasing = m.sym(topright);
    let botleft = m.lnext(flipedge);
    let botlcasing = m.sym(botleft);
    let botright = m.lprev(flipedge);
    let botrcasing = m.sym(botright);

    // Rotate the quadrilateral one-quarter turn clockwise.
    m.bond(topleft, toprcasing);
    m.bond(botleft, toplcasing);
    m.bond(botright, botlcasing);
    m.bond(topright, botrcasing);

    if m.checksegments {
        let toplsubseg = m.tspivot(topleft);
        let botlsubseg = m.tspivot(botleft);
        let botrsubseg = m.tspivot(botright);
        let toprsubseg = m.tspivot(topright);
        if toplsubseg.is_dummy() {
            m.tsdissolve(botleft);
        } else {
            m.tsbond(botleft, toplsubseg);
        }
        if botlsubseg.is_dummy() {
            m.tsdissolve(botright);
        } else {
            m.tsbond(botright, botlsubseg);
        }
        if botrsubseg.is_dummy() {
            m.tsdissolve(topright);
        } else {
            m.tsbond(topright, botrsubseg);
        }
        if toprsubseg.is_dummy() {
            m.tsdissolve(topleft);
        } else {
            m.tsbond(topleft, toprsubseg);
        }
    }

    m.set_org(flipedge, botvertex);
    m.set_dest(flipedge, farvertex);
    m.set_apex(flipedge, leftvertex);
    m.set_org(top, farvertex);
    m.set_dest(top, botvertex);
    m.set_apex(top, rightvertex);
}

// ---------------------------------------------------------------------------
// insertsubseg (triangle.cpp:7823)
// ---------------------------------------------------------------------------

/// Create (or mark) a subsegment at the edge `tri`.
pub fn insert_subseg(m: &mut Mesh, tri: TriHandle, subsegmark: i32) {
    let triorg = m.org(tri);
    let tridest = m.dest(tri);
    if m.vertex_mark(triorg) == 0 {
        m.set_vertex_mark(triorg, subsegmark);
    }
    if m.vertex_mark(tridest) == 0 {
        m.set_vertex_mark(tridest, subsegmark);
    }
    let existing = m.tspivot(tri);
    if existing.is_dummy() {
        let newsubseg = m.make_subseg();
        m.set_sorg(newsubseg, tridest);
        m.set_sdest(newsubseg, triorg);
        m.set_seg_org(newsubseg, tridest);
        m.set_seg_dest(newsubseg, triorg);
        m.tsbond(tri, newsubseg);
        let oppotri = m.sym(tri);
        let newsubseg_sym = m.ssym(newsubseg);
        m.tsbond(oppotri, newsubseg_sym);
        m.set_smark(newsubseg, subsegmark);
    } else if m.smark(existing) == 0 {
        m.set_smark(existing, subsegmark);
    }
}

// ---------------------------------------------------------------------------
// finddirection (triangle.cpp:11608)
// ---------------------------------------------------------------------------

fn finddirection(m: &mut Mesh, searchtri: &mut TriHandle, searchpoint: Vid) -> FindDir {
    let startvertex = m.org(*searchtri);
    let mut rightvertex = m.dest(*searchtri);
    let mut leftvertex = m.apex(*searchtri);
    let mut leftccw = m.ccw(searchpoint, startvertex, leftvertex);
    let mut leftflag = leftccw > 0.0;
    let mut rightccw = m.ccw(startvertex, searchpoint, rightvertex);
    let mut rightflag = rightccw > 0.0;
    if leftflag && rightflag {
        let checktri = m.onext(*searchtri);
        if checktri.is_dummy() {
            leftflag = false;
        } else {
            rightflag = false;
        }
    }
    while leftflag {
        *searchtri = m.onext(*searchtri);
        assert!(
            !searchtri.is_dummy(),
            "finddirection walked off boundary (left)"
        );
        leftvertex = m.apex(*searchtri);
        rightccw = leftccw;
        leftccw = m.ccw(searchpoint, startvertex, leftvertex);
        leftflag = leftccw > 0.0;
    }
    while rightflag {
        *searchtri = m.oprev(*searchtri);
        assert!(
            !searchtri.is_dummy(),
            "finddirection walked off boundary (right)"
        );
        rightvertex = m.dest(*searchtri);
        leftccw = rightccw;
        rightccw = m.ccw(startvertex, searchpoint, rightvertex);
        rightflag = rightccw > 0.0;
    }
    let _ = (rightvertex, leftvertex);
    if leftccw == 0.0 {
        FindDir::LeftCollinear
    } else if rightccw == 0.0 {
        FindDir::RightCollinear
    } else {
        FindDir::Within
    }
}

// ---------------------------------------------------------------------------
// scoutsegment (triangle.cpp:11850)
// ---------------------------------------------------------------------------

fn scoutsegment(m: &mut Mesh, searchtri: &mut TriHandle, endpoint2: Vid, newmark: i32) -> bool {
    let collinear = finddirection(m, searchtri, endpoint2);
    let rightvertex = m.dest(*searchtri);
    let leftvertex = m.apex(*searchtri);
    let p2 = m.point(endpoint2);
    let left_is = m.point(leftvertex) == p2;
    let right_is = m.point(rightvertex) == p2;
    if left_is || right_is {
        if left_is {
            *searchtri = m.lprev(*searchtri);
        }
        insert_subseg(m, *searchtri, newmark);
        true
    } else if collinear == FindDir::LeftCollinear {
        *searchtri = m.lprev(*searchtri);
        insert_subseg(m, *searchtri, newmark);
        scoutsegment(m, searchtri, endpoint2, newmark)
    } else if collinear == FindDir::RightCollinear {
        insert_subseg(m, *searchtri, newmark);
        *searchtri = m.lnext(*searchtri);
        scoutsegment(m, searchtri, endpoint2, newmark)
    } else {
        let crosstri = m.lnext(*searchtri);
        let crosssubseg = m.tspivot(crosstri);
        if crosssubseg.is_dummy() {
            false
        } else {
            panic!(
                "weka: input segments that cross each other are not supported; \
                 split them at their intersection point before meshing"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// delaunayfixup (triangle.cpp:12064)
// ---------------------------------------------------------------------------

fn delaunayfixup(m: &mut Mesh, fixuptri: &mut TriHandle, leftside: bool) {
    let neartri = m.lnext(*fixuptri);
    let mut fartri = m.sym(neartri);
    if fartri.is_dummy() {
        return;
    }
    let faredge = m.tspivot(neartri);
    if !faredge.is_dummy() {
        return;
    }
    let nearvertex = m.apex(neartri);
    let leftvertex = m.org(neartri);
    let rightvertex = m.dest(neartri);
    let farvertex = m.apex(fartri);
    if leftside {
        if m.ccw(nearvertex, leftvertex, farvertex) <= 0.0 {
            return;
        }
    } else if m.ccw(farvertex, rightvertex, nearvertex) <= 0.0 {
        return;
    }
    if m.ccw(rightvertex, leftvertex, farvertex) > 0.0
        && m.in_circle(leftvertex, farvertex, rightvertex, nearvertex) <= 0.0
    {
        return;
    }
    flip(m, neartri);
    *fixuptri = m.lprev(*fixuptri);
    delaunayfixup(m, fixuptri, leftside);
    delaunayfixup(m, &mut fartri, leftside);
}

// ---------------------------------------------------------------------------
// constrainededge (triangle.cpp:12184)
// ---------------------------------------------------------------------------

fn constrainededge(m: &mut Mesh, starttri: TriHandle, endpoint2: Vid, newmark: i32) {
    let endpoint1 = m.org(starttri);
    let p1 = m.point(endpoint1);
    let p2 = m.point(endpoint2);
    let mut fixuptri = m.lnext(starttri);
    flip(m, fixuptri);
    let mut collision = false;
    let mut done = false;
    loop {
        let farvertex = m.org(fixuptri);
        if m.point(farvertex) == p2 {
            let mut fixuptri2 = m.oprev(fixuptri);
            delaunayfixup(m, &mut fixuptri, false);
            delaunayfixup(m, &mut fixuptri2, true);
            done = true;
        } else {
            let area = m.ccw(endpoint1, endpoint2, farvertex);
            if area == 0.0 {
                collision = true;
                let mut fixuptri2 = m.oprev(fixuptri);
                delaunayfixup(m, &mut fixuptri, false);
                delaunayfixup(m, &mut fixuptri2, true);
                done = true;
            } else {
                if area > 0.0 {
                    let mut fixuptri2 = m.oprev(fixuptri);
                    delaunayfixup(m, &mut fixuptri2, true);
                    fixuptri = m.lprev(fixuptri);
                } else {
                    delaunayfixup(m, &mut fixuptri, false);
                    fixuptri = m.oprev(fixuptri);
                }
                let crosssubseg = m.tspivot(fixuptri);
                if crosssubseg.is_dummy() {
                    flip(m, fixuptri);
                } else {
                    let _ = (p1,);
                    panic!(
                        "weka: input segments that cross each other are not supported; \
                         split them at their intersection point before meshing"
                    );
                }
            }
        }
        if done {
            break;
        }
    }
    insert_subseg(m, fixuptri, newmark);
    if collision && !scoutsegment(m, &mut fixuptri, endpoint2, newmark) {
        constrainededge(m, fixuptri, endpoint2, newmark);
    }
}

// ---------------------------------------------------------------------------
// preciselocate / locate (triangle.cpp:7508 / :7652)
// ---------------------------------------------------------------------------

pub fn preciselocate(
    m: &mut Mesh,
    searchpoint: [f64; 2],
    searchtri: &mut TriHandle,
    stopatsubsegment: bool,
) -> LocateResult {
    let mut forg = m.point(m.org(*searchtri));
    let mut fdest = m.point(m.dest(*searchtri));
    let mut fapex = m.point(m.apex(*searchtri));
    loop {
        if fapex == searchpoint {
            *searchtri = m.lprev(*searchtri);
            return LocateResult::OnVertex;
        }
        let destorient = crate::predicates::orient2d(forg, fapex, searchpoint, m.noexact);
        let orgorient = crate::predicates::orient2d(fapex, fdest, searchpoint, m.noexact);
        let moveleft;
        if destorient > 0.0 {
            if orgorient > 0.0 {
                moveleft = (fapex[0] - searchpoint[0]) * (fdest[0] - forg[0])
                    + (fapex[1] - searchpoint[1]) * (fdest[1] - forg[1])
                    > 0.0;
            } else {
                moveleft = true;
            }
        } else if orgorient > 0.0 {
            moveleft = false;
        } else {
            if destorient == 0.0 {
                *searchtri = m.lprev(*searchtri);
                return LocateResult::OnEdge;
            }
            if orgorient == 0.0 {
                *searchtri = m.lnext(*searchtri);
                return LocateResult::OnEdge;
            }
            return LocateResult::InTriangle;
        }
        let backtracktri = if moveleft {
            fdest = fapex;
            m.lprev(*searchtri)
        } else {
            forg = fapex;
            m.lnext(*searchtri)
        };
        *searchtri = m.sym(backtracktri);
        if m.checksegments && stopatsubsegment && !m.tspivot(backtracktri).is_dummy() {
            *searchtri = backtracktri;
            return LocateResult::Outside;
        }
        if searchtri.is_dummy() {
            *searchtri = backtracktri;
            return LocateResult::Outside;
        }
        fapex = m.point(m.apex(*searchtri));
    }
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
}

pub fn locate(
    m: &mut Mesh,
    rng: &mut Rng,
    searchpoint: [f64; 2],
    searchtri: &mut TriHandle,
) -> LocateResult {
    let mut searchdist = dist2(searchpoint, m.point(m.org(*searchtri)));

    if let Some(rt) = m.recenttri {
        if !m.is_tri_dead(rt.index()) {
            let torg = m.point(m.org(rt));
            if torg == searchpoint {
                *searchtri = rt;
                return LocateResult::OnVertex;
            }
            let d = dist2(searchpoint, torg);
            if d < searchdist {
                *searchtri = rt;
                searchdist = d;
            }
        }
    }

    // Random-sample live triangles for a closer starting point.
    let items = m.num_triangles();
    let mut samples = 1usize;
    while SAMPLEFACTOR * samples * samples * samples < items {
        samples += 1;
    }
    let arena = m.tri_arena_len();
    for _ in 0..samples {
        // Draw a random arena slot; skip dead/dummy.
        let idx = 1 + rng.randomnation((arena - 1).max(1) as u32) as usize;
        if idx >= arena || m.is_tri_dead(idx) {
            continue;
        }
        let cand = TriHandle::new(idx as u32, 0);
        let d = dist2(searchpoint, m.point(m.org(cand)));
        if d < searchdist {
            *searchtri = cand;
            searchdist = d;
        }
    }

    let torg = m.point(m.org(*searchtri));
    let tdest = m.point(m.dest(*searchtri));
    if torg == searchpoint {
        return LocateResult::OnVertex;
    }
    if tdest == searchpoint {
        *searchtri = m.lnext(*searchtri);
        return LocateResult::OnVertex;
    }
    let ahead = crate::predicates::orient2d(torg, tdest, searchpoint, m.noexact);
    if ahead < 0.0 {
        *searchtri = m.sym(*searchtri);
    } else if ahead == 0.0
        && (torg[0] < searchpoint[0]) == (searchpoint[0] < tdest[0])
        && (torg[1] < searchpoint[1]) == (searchpoint[1] < tdest[1])
    {
        return LocateResult::OnEdge;
    }
    preciselocate(m, searchpoint, searchtri, false)
}

// ---------------------------------------------------------------------------
// insertsegment (triangle.cpp:12286)
// ---------------------------------------------------------------------------

fn insertsegment(m: &mut Mesh, rng: &mut Rng, mut endpoint1: Vid, endpoint2: Vid, newmark: i32) {
    let mut searchtri1 = m.vertex2tri(endpoint1);
    if searchtri1.is_dummy() || m.org(searchtri1) != endpoint1 {
        // Locate by point search from a hull edge.
        searchtri1 = m.sym(TriHandle::new(0, 0));
        let p = m.point(endpoint1);
        let res = locate(m, rng, p, &mut searchtri1);
        assert_eq!(res, LocateResult::OnVertex, "cannot locate PSLG endpoint1");
    }
    m.recenttri = Some(searchtri1);
    if scoutsegment(m, &mut searchtri1, endpoint2, newmark) {
        return;
    }
    endpoint1 = m.org(searchtri1);

    let mut searchtri2 = m.vertex2tri(endpoint2);
    if searchtri2.is_dummy() || m.org(searchtri2) != endpoint2 {
        searchtri2 = m.sym(TriHandle::new(0, 0));
        let p = m.point(endpoint2);
        let res = locate(m, rng, p, &mut searchtri2);
        assert_eq!(res, LocateResult::OnVertex, "cannot locate PSLG endpoint2");
    }
    m.recenttri = Some(searchtri2);
    if scoutsegment(m, &mut searchtri2, endpoint1, newmark) {
        return;
    }
    // Force the segment in directly (CDT). Conforming (-D / splitseg) is deferred.
    constrainededge(m, searchtri1, endpoint2, newmark);
}

// ---------------------------------------------------------------------------
// markhull (triangle.cpp:12398)
// ---------------------------------------------------------------------------

fn markhull(m: &mut Mesh) {
    let mut hulltri = m.sym(TriHandle::new(0, 0));
    let starttri = hulltri;
    loop {
        insert_subseg(m, hulltri, 1);
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

// ---------------------------------------------------------------------------
// formskeleton (triangle.cpp:12444, TRILIBRARY variant)
// ---------------------------------------------------------------------------

/// Recover PSLG segments and (optionally) enclose the convex hull.
pub fn form_skeleton(
    m: &mut Mesh,
    rng: &mut Rng,
    segments: &[[usize; 2]],
    segment_markers: Option<&[i32]>,
    poly: bool,
    convex: bool,
) {
    if poly {
        if m.num_triangles() == 0 {
            return;
        }
        if !segments.is_empty() {
            make_vertex_map(m);
        }
        for (i, &[end1, end2]) in segments.iter().enumerate() {
            if end1 >= m.invertices || end2 >= m.invertices {
                continue; // invalid endpoint; ignore (matches C warning behavior)
            }
            let boundmarker = segment_markers.map_or(0, |mk| mk[i]);
            let (e1, e2) = (end1 as Vid, end2 as Vid);
            if m.point(e1) == m.point(e2) {
                continue; // coincident endpoints
            }
            insertsegment(m, rng, e1, e2, boundmarker);
        }
    }
    if convex || !poly {
        markhull(m);
    }
}
