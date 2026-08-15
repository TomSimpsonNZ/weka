//! Divide-and-conquer Delaunay triangulation — port of triangle.cpp:9217-10050
//! (`vertexsort`, `vertexmedian`, `alternateaxes`, `divconqrecurse`,
//! `mergehulls`, `removeghosts`, `divconqdelaunay`).
//!
//! Bounding ("ghost") triangles are represented exactly as in Triangle: a
//! triangle whose apex is [`NO_VERTEX`]. Self-mutating handle macros
//! (`lnextself`, `symself`, …) become plain reassignment of the `Copy` handle.

use crate::mesh::{Mesh, TriHandle, VertexKind, Vid, NO_VERTEX};
use crate::rng::Rng;

/// A vertex carried through the presort with its coordinates inlined, so the
/// comparison-heavy sort never indexes back into the vertex arena (avoiding a
/// per-comparison bounds check) and stays cache-local.
type SortPt = ([f64; 2], Vid);

/// Sort vertices lexicographically by (x, y). Port of `vertexsort` (:9217),
/// including the randomized pivot so the recursion order matches C.
fn vertexsort(arr: &mut [SortPt], rng: &mut Rng) {
    let n = arr.len();
    if n == 2 {
        let (p0, p1) = (arr[0].0, arr[1].0);
        if p0[0] > p1[0] || (p0[0] == p1[0] && p0[1] > p1[1]) {
            arr.swap(0, 1);
        }
        return;
    }
    let pivot = rng.randomnation(n as u32) as usize;
    let (pivotx, pivoty) = (arr[pivot].0[0], arr[pivot].0[1]);
    let mut left: isize = -1;
    let mut right: isize = n as isize;
    while left < right {
        loop {
            left += 1;
            if !(left <= right && {
                let p = arr[left as usize].0;
                p[0] < pivotx || (p[0] == pivotx && p[1] < pivoty)
            }) {
                break;
            }
        }
        loop {
            right -= 1;
            if !(left <= right && {
                let p = arr[right as usize].0;
                p[0] > pivotx || (p[0] == pivotx && p[1] > pivoty)
            }) {
                break;
            }
        }
        if left < right {
            arr.swap(left as usize, right as usize);
        }
    }
    if left > 1 {
        vertexsort(&mut arr[..left as usize], rng);
    }
    if right < n as isize - 2 {
        vertexsort(&mut arr[(right as usize + 1)..], rng);
    }
}

/// Shuffle so the first `median` vertices precede the rest along `axis`. Port of
/// `vertexmedian` (:9291).
fn vertexmedian(arr: &mut [SortPt], median: usize, axis: usize, rng: &mut Rng) {
    let n = arr.len();
    if n == 2 {
        let (p0, p1) = (arr[0].0, arr[1].0);
        if p0[axis] > p1[axis] || (p0[axis] == p1[axis] && p0[1 - axis] > p1[1 - axis]) {
            arr.swap(0, 1);
        }
        return;
    }
    let pivot = rng.randomnation(n as u32) as usize;
    let (pivot1, pivot2) = (arr[pivot].0[axis], arr[pivot].0[1 - axis]);
    let mut left: isize = -1;
    let mut right: isize = n as isize;
    while left < right {
        loop {
            left += 1;
            if !(left <= right && {
                let p = arr[left as usize].0;
                p[axis] < pivot1 || (p[axis] == pivot1 && p[1 - axis] < pivot2)
            }) {
                break;
            }
        }
        loop {
            right -= 1;
            if !(left <= right && {
                let p = arr[right as usize].0;
                p[axis] > pivot1 || (p[axis] == pivot1 && p[1 - axis] > pivot2)
            }) {
                break;
            }
        }
        if left < right {
            arr.swap(left as usize, right as usize);
        }
    }
    if left as usize > median {
        vertexmedian(&mut arr[..left as usize], median, axis, rng);
    }
    if right < median as isize - 1 {
        let off = right as usize + 1;
        vertexmedian(&mut arr[off..], median - off, axis, rng);
    }
}

/// Alternating-cut presort. Port of `alternateaxes` (:9369).
fn alternateaxes(arr: &mut [SortPt], mut axis: usize, rng: &mut Rng) {
    let n = arr.len();
    let divider = n >> 1;
    if n <= 3 {
        axis = 0;
    }
    vertexmedian(arr, divider, axis, rng);
    if n - divider >= 2 {
        if divider >= 2 {
            alternateaxes(&mut arr[..divider], 1 - axis, rng);
        }
        alternateaxes(&mut arr[divider..], 1 - axis, rng);
    }
}

/// Recursively triangulate `arr` (already presorted). Returns `(farleft,
/// farright)` bounding triangles. Port of `divconqrecurse` (:9760).
fn divconqrecurse(
    m: &mut Mesh,
    arr: &[Vid],
    axis: usize,
    dwyer: bool,
) -> (TriHandle, TriHandle) {
    let n = arr.len();
    if n == 2 {
        let mut farleft = m.make_triangle();
        m.set_org(farleft, arr[0]);
        m.set_dest(farleft, arr[1]);
        let mut farright = m.make_triangle();
        m.set_org(farright, arr[1]);
        m.set_dest(farright, arr[0]);
        m.bond(farleft, farright);
        farleft = m.lprev(farleft);
        farright = m.lnext(farright);
        m.bond(farleft, farright);
        farleft = m.lprev(farleft);
        farright = m.lnext(farright);
        m.bond(farleft, farright);
        // Ensure origin of farleft is arr[0].
        let farleft = m.lprev(farright);
        (farleft, farright)
    } else if n == 3 {
        let midtri = m.make_triangle();
        let tri1 = m.make_triangle();
        let tri2 = m.make_triangle();
        let tri3 = m.make_triangle();
        let area = m.ccw(arr[0], arr[1], arr[2]);
        if area == 0.0 {
            // Collinear: two edges.
            m.set_org(midtri, arr[0]);
            m.set_dest(midtri, arr[1]);
            m.set_org(tri1, arr[1]);
            m.set_dest(tri1, arr[0]);
            m.set_org(tri2, arr[2]);
            m.set_dest(tri2, arr[1]);
            m.set_org(tri3, arr[1]);
            m.set_dest(tri3, arr[2]);
            m.bond(midtri, tri1);
            m.bond(tri2, tri3);
            let midtri = m.lnext(midtri);
            let tri1 = m.lprev(tri1);
            let tri2 = m.lnext(tri2);
            let tri3 = m.lprev(tri3);
            m.bond(midtri, tri3);
            m.bond(tri1, tri2);
            let midtri = m.lnext(midtri);
            let tri1 = m.lprev(tri1);
            let tri2 = m.lnext(tri2);
            let tri3 = m.lprev(tri3);
            m.bond(midtri, tri1);
            m.bond(tri2, tri3);
            (tri1, tri2)
        } else {
            m.set_org(midtri, arr[0]);
            m.set_dest(tri1, arr[0]);
            m.set_org(tri3, arr[0]);
            if area > 0.0 {
                m.set_dest(midtri, arr[1]);
                m.set_org(tri1, arr[1]);
                m.set_dest(tri2, arr[1]);
                m.set_apex(midtri, arr[2]);
                m.set_org(tri2, arr[2]);
                m.set_dest(tri3, arr[2]);
            } else {
                m.set_dest(midtri, arr[2]);
                m.set_org(tri1, arr[2]);
                m.set_dest(tri2, arr[2]);
                m.set_apex(midtri, arr[1]);
                m.set_org(tri2, arr[1]);
                m.set_dest(tri3, arr[1]);
            }
            m.bond(midtri, tri1);
            let midtri = m.lnext(midtri);
            m.bond(midtri, tri2);
            let midtri = m.lnext(midtri);
            m.bond(midtri, tri3);
            let tri1 = m.lprev(tri1);
            let tri2 = m.lnext(tri2);
            m.bond(tri1, tri2);
            let tri1 = m.lprev(tri1);
            let tri3 = m.lprev(tri3);
            m.bond(tri1, tri3);
            let tri2 = m.lnext(tri2);
            let tri3 = m.lprev(tri3);
            m.bond(tri2, tri3);
            let farleft = tri1;
            let farright = if area > 0.0 { tri2 } else { m.lnext(farleft) };
            (farleft, farright)
        }
    } else {
        let divider = n >> 1;
        let (farleft, mut innerleft) = divconqrecurse(m, &arr[..divider], 1 - axis, dwyer);
        let (mut innerright, farright) = divconqrecurse(m, &arr[divider..], 1 - axis, dwyer);
        let mut farleft = farleft;
        let mut farright = farright;
        mergehulls(
            m,
            &mut farleft,
            &mut innerleft,
            &mut innerright,
            &mut farright,
            axis,
            dwyer,
        );
        (farleft, farright)
    }
}

/// Merge two adjacent Delaunay triangulations. Port of `mergehulls` (:9433).
#[allow(clippy::too_many_arguments)]
fn mergehulls(
    m: &mut Mesh,
    farleft: &mut TriHandle,
    innerleft: &mut TriHandle,
    innerright: &mut TriHandle,
    farright: &mut TriHandle,
    axis: usize,
    dwyer: bool,
) {
    let mut innerleftdest = m.dest(*innerleft);
    let mut innerleftapex = m.apex(*innerleft);
    let mut innerrightorg = m.org(*innerright);
    let mut innerrightapex = m.apex(*innerright);

    if dwyer && axis == 1 {
        let mut farleftpt = m.org(*farleft);
        let mut farleftapex = m.apex(*farleft);
        let mut farrightpt = m.dest(*farright);
        let mut farrightapex = m.apex(*farright);
        while m.point(farleftapex)[1] < m.point(farleftpt)[1] {
            *farleft = m.lnext(*farleft);
            *farleft = m.sym(*farleft);
            farleftpt = farleftapex;
            farleftapex = m.apex(*farleft);
        }
        let mut checkedge = m.sym(*innerleft);
        let mut checkvertex = m.apex(checkedge);
        while m.point(checkvertex)[1] > m.point(innerleftdest)[1] {
            *innerleft = m.lnext(checkedge);
            innerleftapex = innerleftdest;
            innerleftdest = checkvertex;
            checkedge = m.sym(*innerleft);
            checkvertex = m.apex(checkedge);
        }
        while m.point(innerrightapex)[1] < m.point(innerrightorg)[1] {
            *innerright = m.lnext(*innerright);
            *innerright = m.sym(*innerright);
            innerrightorg = innerrightapex;
            innerrightapex = m.apex(*innerright);
        }
        checkedge = m.sym(*farright);
        checkvertex = m.apex(checkedge);
        while m.point(checkvertex)[1] > m.point(farrightpt)[1] {
            *farright = m.lnext(checkedge);
            farrightapex = farrightpt;
            farrightpt = checkvertex;
            checkedge = m.sym(*farright);
            checkvertex = m.apex(checkedge);
        }
        let _ = (farleftapex, farrightapex);
    }

    // Find a line tangent to and below both hulls.
    loop {
        let mut changemade = false;
        if m.ccw(innerleftdest, innerleftapex, innerrightorg) > 0.0 {
            *innerleft = m.lprev(*innerleft);
            *innerleft = m.sym(*innerleft);
            innerleftdest = innerleftapex;
            innerleftapex = m.apex(*innerleft);
            changemade = true;
        }
        if m.ccw(innerrightapex, innerrightorg, innerleftdest) > 0.0 {
            *innerright = m.lnext(*innerright);
            *innerright = m.sym(*innerright);
            innerrightorg = innerrightapex;
            innerrightapex = m.apex(*innerright);
            changemade = true;
        }
        if !changemade {
            break;
        }
    }

    let mut leftcand = m.sym(*innerleft);
    let mut rightcand = m.sym(*innerright);
    let mut baseedge = m.make_triangle();
    m.bond(baseedge, *innerleft);
    baseedge = m.lnext(baseedge);
    m.bond(baseedge, *innerright);
    baseedge = m.lnext(baseedge);
    m.set_org(baseedge, innerrightorg);
    m.set_dest(baseedge, innerleftdest);
    // Apex of baseedge intentionally NULL.

    let farleftpt = m.org(*farleft);
    if innerleftdest == farleftpt {
        *farleft = m.lnext(baseedge);
    }
    let farrightpt = m.dest(*farright);
    if innerrightorg == farrightpt {
        *farright = m.lprev(baseedge);
    }

    let mut lowerleft = innerleftdest;
    let mut lowerright = innerrightorg;
    let mut upperleft = m.apex(leftcand);
    let mut upperright = m.apex(rightcand);

    loop {
        let leftfinished = m.ccw(upperleft, lowerleft, lowerright) <= 0.0;
        let rightfinished = m.ccw(upperright, lowerleft, lowerright) <= 0.0;
        if leftfinished && rightfinished {
            let mut nextedge = m.make_triangle();
            m.set_org(nextedge, lowerleft);
            m.set_dest(nextedge, lowerright);
            m.bond(nextedge, baseedge);
            nextedge = m.lnext(nextedge);
            m.bond(nextedge, rightcand);
            nextedge = m.lnext(nextedge);
            m.bond(nextedge, leftcand);

            if dwyer && axis == 1 {
                let mut farleftpt = m.org(*farleft);
                let mut farleftapex = m.apex(*farleft);
                let mut farrightpt = m.dest(*farright);
                let mut farrightapex = m.apex(*farright);
                let mut checkedge = m.sym(*farleft);
                let mut checkvertex = m.apex(checkedge);
                while m.point(checkvertex)[0] < m.point(farleftpt)[0] {
                    *farleft = m.lprev(checkedge);
                    farleftapex = farleftpt;
                    farleftpt = checkvertex;
                    checkedge = m.sym(*farleft);
                    checkvertex = m.apex(checkedge);
                }
                while m.point(farrightapex)[0] > m.point(farrightpt)[0] {
                    *farright = m.lprev(*farright);
                    *farright = m.sym(*farright);
                    farrightpt = farrightapex;
                    farrightapex = m.apex(*farright);
                }
                let _ = (farleftapex,);
            }
            return;
        }

        if !leftfinished {
            let mut nextedge = m.lprev(leftcand);
            nextedge = m.sym(nextedge);
            let mut nextapex = m.apex(nextedge);
            if nextapex != NO_VERTEX {
                let mut badedge = m.in_circle(lowerleft, lowerright, upperleft, nextapex) > 0.0;
                while badedge {
                    nextedge = m.lnext(nextedge);
                    let topcasing = m.sym(nextedge);
                    nextedge = m.lnext(nextedge);
                    let sidecasing = m.sym(nextedge);
                    m.bond(nextedge, topcasing);
                    m.bond(leftcand, sidecasing);
                    leftcand = m.lnext(leftcand);
                    let outercasing = m.sym(leftcand);
                    nextedge = m.lprev(nextedge);
                    m.bond(nextedge, outercasing);
                    m.set_org(leftcand, lowerleft);
                    m.set_dest(leftcand, NO_VERTEX);
                    m.set_apex(leftcand, nextapex);
                    m.set_org(nextedge, NO_VERTEX);
                    m.set_dest(nextedge, upperleft);
                    m.set_apex(nextedge, nextapex);
                    upperleft = nextapex;
                    nextedge = sidecasing;
                    nextapex = m.apex(nextedge);
                    badedge = if nextapex != NO_VERTEX {
                        m.in_circle(lowerleft, lowerright, upperleft, nextapex) > 0.0
                    } else {
                        false
                    };
                }
            }
        }

        if !rightfinished {
            let mut nextedge = m.lnext(rightcand);
            nextedge = m.sym(nextedge);
            let mut nextapex = m.apex(nextedge);
            if nextapex != NO_VERTEX {
                let mut badedge = m.in_circle(lowerleft, lowerright, upperright, nextapex) > 0.0;
                while badedge {
                    nextedge = m.lprev(nextedge);
                    let topcasing = m.sym(nextedge);
                    nextedge = m.lprev(nextedge);
                    let sidecasing = m.sym(nextedge);
                    m.bond(nextedge, topcasing);
                    m.bond(rightcand, sidecasing);
                    rightcand = m.lprev(rightcand);
                    let outercasing = m.sym(rightcand);
                    nextedge = m.lnext(nextedge);
                    m.bond(nextedge, outercasing);
                    m.set_org(rightcand, NO_VERTEX);
                    m.set_dest(rightcand, lowerright);
                    m.set_apex(rightcand, nextapex);
                    m.set_org(nextedge, upperright);
                    m.set_dest(nextedge, NO_VERTEX);
                    m.set_apex(nextedge, nextapex);
                    upperright = nextapex;
                    nextedge = sidecasing;
                    nextapex = m.apex(nextedge);
                    badedge = if nextapex != NO_VERTEX {
                        m.in_circle(lowerleft, lowerright, upperright, nextapex) > 0.0
                    } else {
                        false
                    };
                }
            }
        }

        if leftfinished
            || (!rightfinished
                && m.in_circle(upperleft, lowerleft, lowerright, upperright) > 0.0)
        {
            // Knit: edge from lowerleft to upperright.
            m.bond(baseedge, rightcand);
            baseedge = m.lprev(rightcand);
            m.set_dest(baseedge, lowerleft);
            lowerright = upperright;
            rightcand = m.sym(baseedge);
            upperright = m.apex(rightcand);
        } else {
            // Knit: edge from upperleft to lowerright.
            m.bond(baseedge, leftcand);
            baseedge = m.lnext(leftcand);
            m.set_org(baseedge, lowerright);
            lowerleft = upperleft;
            leftcand = m.sym(baseedge);
            upperleft = m.apex(leftcand);
        }
    }
}

/// Remove the bounding ("ghost") triangles, set hull vertex markers (when no
/// PSLG), and count the hull edges. Port of `removeghosts` (:9924).
fn removeghosts(m: &mut Mesh, startghost: TriHandle, poly: bool) -> usize {
    // Record a hull edge as the point-location start.
    let mut searchedge = m.lprev(startghost);
    searchedge = m.sym(searchedge);
    m.set_hull_edge(searchedge);

    let mut dissolveedge = startghost;
    let mut hullsize = 0usize;
    loop {
        hullsize += 1;
        let deadtriangle = m.lnext(dissolveedge);
        dissolveedge = m.lprev(dissolveedge);
        dissolveedge = m.sym(dissolveedge);
        if !poly && !dissolveedge.is_dummy() {
            let markorg = m.org(dissolveedge);
            if markorg != NO_VERTEX && m.vertex_mark(markorg) == 0 {
                m.set_vertex_mark(markorg, 1);
            }
        }
        m.dissolve(dissolveedge);
        dissolveedge = m.sym(deadtriangle);
        m.triangle_dealloc(deadtriangle);
        if dissolveedge == startghost {
            break;
        }
    }
    hullsize
}

/// Form the Delaunay triangulation of all input vertices via divide-and-conquer.
/// Returns the convex-hull size. Port of `divconqdelaunay` (:9987).
pub fn divconq_delaunay(m: &mut Mesh, rng: &mut Rng, dwyer: bool, poly: bool) -> usize {
    let n = m.invertices;
    // Carry coordinates inline through the presort (no arena indexing per compare).
    let mut sortarray: Vec<SortPt> = (0..n as u32).map(|v| (m.point(v), v)).collect();
    vertexsort(&mut sortarray, rng);

    // Discard duplicate vertices.
    let mut i = 0usize;
    for j in 1..n {
        if sortarray[i].0 == sortarray[j].0 {
            m.set_vertex_kind(sortarray[j].1, VertexKind::Undead);
            m.undeads += 1;
        } else {
            i += 1;
            sortarray[i] = sortarray[j];
        }
    }
    i += 1;
    sortarray.truncate(i);

    if dwyer {
        let divider = i >> 1;
        if i - divider >= 2 {
            if divider >= 2 {
                alternateaxes(&mut sortarray[..divider], 1, rng);
            }
            alternateaxes(&mut sortarray[divider..], 1, rng);
        }
    }

    // Extract the sorted vertex ids for the recursion.
    let ids: Vec<Vid> = sortarray.iter().map(|p| p.1).collect();
    let (hullleft, _hullright) = divconqrecurse(m, &ids, 0, dwyer);
    removeghosts(m, hullleft, poly)
}
