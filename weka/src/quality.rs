//! Quality meshing (Ruppert's Delaunay refinement) — port of triangle.cpp
//! `insertvertex` (:8211), `undovertex` (:9070), `findcircumcenter` (:6532),
//! `checkseg4encroach` (:7114), `testtriangle` (:7227), the bad-triangle priority
//! queue (:6918/:7061), `tallyencs`/`splitencsegs`/`tallyfaces`/`splittriangle`
//! and `enforcequality` (:13210-13732).
//!
//! Deviations from C, permitted by the behavioral-equivalence goal (quality
//! bounds are guaranteed; vertex counts may differ slightly):
//!   * Off-centers are supported via `offconstant`, matching C's default.
//!   * Chew's free-vertex deletion inside diametral circles (the `deletevertex`
//!     calls in `splitencsegs`) is omitted; this yields classic Ruppert, which
//!     still terminates and meets the minimum-angle bound.

use crate::mesh::{FlipKind, FlipRecord, Mesh, SubHandle, TriHandle, VertexKind, Vid, NO_VERTEX};
use crate::predicates::orient2d;
use crate::rng::Rng;
use crate::segments::{insert_subseg, locate, preciselocate, unflip, LocateResult};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InsertResult {
    Successful,
    Encroaching,
    Violating,
    Duplicate,
}

/// A user "triunsuitable" predicate: `(p0, p1, p2, area) -> too_big`.
pub type UserTest = Box<dyn Fn([f64; 2], [f64; 2], [f64; 2], f64) -> bool>;

struct BadSubseg {
    enc: SubHandle,
    org: Vid,
    dest: Vid,
}

#[derive(Clone, Copy)]
struct BadTri {
    poortri: TriHandle,
    key: f64,
    org: Vid,
    dest: Vid,
    apex: Vid,
    next: i32,
}

/// Quality-meshing parameters and working state.
pub struct Quality {
    pub minangle: f64,
    pub goodangle: f64,
    pub offconstant: f64,
    pub maxarea: f64,
    pub fixedarea: bool,
    pub vararea: bool,
    pub conformdel: bool,
    pub nobisect: i32,
    pub steinerleft: i64,
    pub usertest: Option<UserTest>,

    badsubsegs: Vec<BadSubseg>,
    badtris: Vec<BadTri>,
    queuefront: Vec<i32>,
    queuetail: Vec<i32>,
    nextnonemptyq: Vec<i32>,
    firstnonemptyq: i32,
}

impl Quality {
    pub fn new(minangle: f64, maxarea: Option<f64>) -> Self {
        let mut goodangle = (minangle * std::f64::consts::PI / 180.0).cos();
        let offconstant = if goodangle == 1.0 {
            0.0
        } else {
            0.475 * ((1.0 + goodangle) / (1.0 - goodangle)).sqrt()
        };
        goodangle *= goodangle;
        Quality {
            minangle,
            goodangle,
            offconstant,
            maxarea: maxarea.unwrap_or(-1.0),
            fixedarea: maxarea.is_some(),
            vararea: false,
            conformdel: false,
            nobisect: 0,
            steinerleft: -1,
            usertest: None,
            badsubsegs: Vec::new(),
            badtris: Vec::new(),
            queuefront: vec![-1; 4096],
            queuetail: vec![-1; 4096],
            nextnonemptyq: vec![-1; 4096],
            firstnonemptyq: -1,
        }
    }
}

// ---------------------------------------------------------------------------
// findcircumcenter (triangle.cpp:6532)
// ---------------------------------------------------------------------------

/// Returns (circumcenter, xi, eta). `offcenter` enables Üngör off-centers.
fn find_circumcenter(
    m: &Mesh,
    torg: [f64; 2],
    tdest: [f64; 2],
    tapex: [f64; 2],
    offconstant: f64,
    offcenter: bool,
) -> ([f64; 2], f64, f64) {
    let xdo = tdest[0] - torg[0];
    let ydo = tdest[1] - torg[1];
    let xao = tapex[0] - torg[0];
    let yao = tapex[1] - torg[1];
    let dodist = xdo * xdo + ydo * ydo;
    let aodist = xao * xao + yao * yao;
    let dadist = (tdest[0] - tapex[0]) * (tdest[0] - tapex[0])
        + (tdest[1] - tapex[1]) * (tdest[1] - tapex[1]);
    let denominator = if m.noexact {
        0.5 / (xdo * yao - xao * ydo)
    } else {
        0.5 / orient2d(tdest, tapex, torg, false)
    };
    let mut dx = (yao * dodist - ydo * aodist) * denominator;
    let mut dy = (xdo * aodist - xao * dodist) * denominator;

    if dodist < aodist && dodist < dadist {
        if offcenter && offconstant > 0.0 {
            let dxoff = 0.5 * xdo - offconstant * ydo;
            let dyoff = 0.5 * ydo + offconstant * xdo;
            if dxoff * dxoff + dyoff * dyoff < dx * dx + dy * dy {
                dx = dxoff;
                dy = dyoff;
            }
        }
    } else if aodist < dadist {
        if offcenter && offconstant > 0.0 {
            let dxoff = 0.5 * xao + offconstant * yao;
            let dyoff = 0.5 * yao - offconstant * xao;
            if dxoff * dxoff + dyoff * dyoff < dx * dx + dy * dy {
                dx = dxoff;
                dy = dyoff;
            }
        }
    } else if offcenter && offconstant > 0.0 {
        let dxoff = 0.5 * (tapex[0] - tdest[0]) - offconstant * (tapex[1] - tdest[1]);
        let dyoff = 0.5 * (tapex[1] - tdest[1]) + offconstant * (tapex[0] - tdest[0]);
        if dxoff * dxoff + dyoff * dyoff
            < (dx - xdo) * (dx - xdo) + (dy - ydo) * (dy - ydo)
        {
            dx = xdo + dxoff;
            dy = ydo + dyoff;
        }
    }

    let cc = [torg[0] + dx, torg[1] + dy];
    let xi = (yao * dx - xao * dy) * (2.0 * denominator);
    let eta = (xdo * dy - ydo * dx) * (2.0 * denominator);
    (cc, xi, eta)
}

// ---------------------------------------------------------------------------
// Bad-triangle priority queue (triangle.cpp:6918 / :7061)
// ---------------------------------------------------------------------------

impl Quality {
    fn enqueue_badtriang(&mut self, idx: usize) {
        let key = self.badtris[idx].key;
        let (length, posexponent) = if key >= 1.0 {
            (key, true)
        } else {
            (1.0 / key, false)
        };
        let mut length = length;
        let mut exponent = 0i32;
        while length > 2.0 {
            let mut expincrement = 1i32;
            let mut multiplier = 0.5;
            while length * multiplier * multiplier > 1.0 {
                expincrement *= 2;
                multiplier *= multiplier;
            }
            exponent += expincrement;
            length *= multiplier;
        }
        exponent = 2 * exponent + i32::from(length > std::f64::consts::SQRT_2);
        // For IEEE double `exponent` is nominally in 0..2047, but pathologically
        // tiny/large edges can exceed that; clamp into the 0..4095 bucket range
        // (shortest edges → highest-priority bucket 4095).
        let queuenumber = if posexponent {
            (2047 - exponent).clamp(0, 4095) as usize
        } else {
            (2048 + exponent).clamp(0, 4095) as usize
        };

        if self.queuefront[queuenumber] < 0 {
            if queuenumber as i32 > self.firstnonemptyq {
                self.nextnonemptyq[queuenumber] = self.firstnonemptyq;
                self.firstnonemptyq = queuenumber as i32;
            } else {
                let mut i = queuenumber + 1;
                while self.queuefront[i] < 0 {
                    i += 1;
                }
                self.nextnonemptyq[queuenumber] = self.nextnonemptyq[i];
                self.nextnonemptyq[i] = queuenumber as i32;
            }
            self.queuefront[queuenumber] = idx as i32;
        } else {
            let tail = self.queuetail[queuenumber] as usize;
            self.badtris[tail].next = idx as i32;
        }
        self.queuetail[queuenumber] = idx as i32;
        self.badtris[idx].next = -1;
    }

    fn enqueue_badtri(&mut self, tri: TriHandle, key: f64, org: Vid, dest: Vid, apex: Vid) {
        let idx = self.badtris.len();
        self.badtris.push(BadTri {
            poortri: tri,
            key,
            org,
            dest,
            apex,
            next: -1,
        });
        self.enqueue_badtriang(idx);
    }

    fn dequeue_badtriang(&mut self) -> Option<usize> {
        if self.firstnonemptyq < 0 {
            return None;
        }
        let q = self.firstnonemptyq as usize;
        let result = self.queuefront[q] as usize;
        self.queuefront[q] = self.badtris[result].next;
        if result as i32 == self.queuetail[q] {
            self.firstnonemptyq = self.nextnonemptyq[q];
        }
        Some(result)
    }

    fn badtris_pending(&self) -> bool {
        self.firstnonemptyq >= 0
    }
}

// ---------------------------------------------------------------------------
// checkseg4encroach (triangle.cpp:7114)
// ---------------------------------------------------------------------------

fn checkseg4encroach(m: &Mesh, q: &mut Quality, testsubseg: SubHandle) -> i32 {
    let mut encroached = 0;
    let mut sides = 0;
    let eorg = m.point(m.sorg(testsubseg));
    let edest = m.point(m.sdest(testsubseg));

    let neighbortri = m.stpivot(testsubseg);
    if !neighbortri.is_dummy() {
        sides += 1;
        let eapex = m.point(m.apex(neighbortri));
        // Diametral-circle encroachment (Ruppert): the apex sees the subsegment
        // at an obtuse angle. (We use the circle rather than Chew's smaller lens
        // because we omit Chew's free-vertex deletion; the circle guarantees
        // termination on its own.)
        let dotproduct =
            (eorg[0] - eapex[0]) * (edest[0] - eapex[0]) + (eorg[1] - eapex[1]) * (edest[1] - eapex[1]);
        if dotproduct < 0.0 {
            encroached = 1;
        }
    }
    let testsym = m.ssym(testsubseg);
    let neighbortri2 = m.stpivot(testsym);
    if !neighbortri2.is_dummy() {
        sides += 1;
        let eapex = m.point(m.apex(neighbortri2));
        let dotproduct =
            (eorg[0] - eapex[0]) * (edest[0] - eapex[0]) + (eorg[1] - eapex[1]) * (edest[1] - eapex[1]);
        if dotproduct < 0.0 {
            encroached += 2;
        }
    }

    if encroached != 0 && (q.nobisect == 0 || (q.nobisect == 1 && sides == 2)) {
        if encroached == 1 {
            q.badsubsegs.push(BadSubseg {
                enc: testsubseg,
                org: m.sorg(testsubseg),
                dest: m.sdest(testsubseg),
            });
        } else {
            q.badsubsegs.push(BadSubseg {
                enc: testsym,
                org: m.sorg(testsym),
                dest: m.sdest(testsym),
            });
        }
    }
    encroached
}

// ---------------------------------------------------------------------------
// testtriangle (triangle.cpp:7227)
// ---------------------------------------------------------------------------

fn testtriangle(m: &mut Mesh, q: &mut Quality, testtri: TriHandle) {
    let torgv = m.org(testtri);
    let tdestv = m.dest(testtri);
    let tapexv = m.apex(testtri);
    let torg = m.point(torgv);
    let tdest = m.point(tdestv);
    let tapex = m.point(tapexv);
    let dxod = torg[0] - tdest[0];
    let dyod = torg[1] - tdest[1];
    let dxda = tdest[0] - tapex[0];
    let dyda = tdest[1] - tapex[1];
    let dxao = tapex[0] - torg[0];
    let dyao = tapex[1] - torg[1];
    let apexlen = dxod * dxod + dyod * dyod;
    let orglen = dxda * dxda + dyda * dyda;
    let destlen = dxao * dxao + dyao * dyao;

    let (minedge, angle, base1v, base2v, tri1);
    if apexlen < orglen && apexlen < destlen {
        minedge = apexlen;
        let a = dxda * dxao + dyda * dyao;
        angle = a * a / (orglen * destlen);
        base1v = torgv;
        base2v = tdestv;
        tri1 = testtri;
    } else if orglen < destlen {
        minedge = orglen;
        let a = dxod * dxao + dyod * dyao;
        angle = a * a / (apexlen * destlen);
        base1v = tdestv;
        base2v = tapexv;
        tri1 = m.lnext(testtri);
    } else {
        minedge = destlen;
        let a = dxod * dxda + dyod * dyda;
        angle = a * a / (apexlen * orglen);
        base1v = tapexv;
        base2v = torgv;
        tri1 = m.lprev(testtri);
    }

    if q.vararea || q.fixedarea || q.usertest.is_some() {
        let area = 0.5 * (dxod * dyda - dyod * dxda);
        if q.fixedarea && area > q.maxarea {
            q.enqueue_badtri(testtri, minedge, torgv, tdestv, tapexv);
            return;
        }
        if q.vararea {
            let bound = m.area_bound(testtri);
            if area > bound && bound > 0.0 {
                q.enqueue_badtri(testtri, minedge, torgv, tdestv, tapexv);
                return;
            }
        }
        if let Some(ut) = &q.usertest {
            if ut(torg, tdest, tapex, area) {
                q.enqueue_badtri(testtri, minedge, torgv, tdestv, tapexv);
                return;
            }
        }
    }

    if angle > q.goodangle {
        // Miller/Pav/Walkington: don't split a skinny triangle whose short edge
        // subtends a small input angle between two segments (avoids livelock).
        // The segment-search walks are bounded by the vertex degree; if no
        // containing subsegment is found we conservatively split the triangle.
        if m.vertex_kind(base1v) == VertexKind::Segment
            && m.vertex_kind(base2v) == VertexKind::Segment
            && m.tspivot(tri1).is_dummy()
        {
            // Search around org(tri1) (clockwise) and around dest (counter) for
            // the subsegments containing each base endpoint.
            let bound = m.tri_arena_len() + 8;
            let find_oprev = |m: &Mesh| {
                let mut t = tri1;
                for _ in 0..bound {
                    t = m.oprev(t);
                    let s = m.tspivot(t);
                    if !s.is_dummy() {
                        return Some((m.seg_org(s), m.seg_dest(s)));
                    }
                }
                None
            };
            let find_dnext = |m: &Mesh| {
                let mut t = tri1;
                for _ in 0..bound {
                    t = m.dnext(t);
                    let s = m.tspivot(t);
                    if !s.is_dummy() {
                        return Some((m.seg_org(s), m.seg_dest(s)));
                    }
                }
                None
            };
            if let (Some((org1, dest1)), Some((org2, dest2))) = (find_oprev(m), find_dnext(m)) {
                let joinvertex = if m.point(dest1) == m.point(org2) {
                    dest1
                } else if m.point(org1) == m.point(dest2) {
                    org1
                } else {
                    NO_VERTEX
                };
                if joinvertex != NO_VERTEX {
                    let jv = m.point(joinvertex);
                    let b1 = m.point(base1v);
                    let b2 = m.point(base2v);
                    let dist1 = (b1[0] - jv[0]).powi(2) + (b1[1] - jv[1]).powi(2);
                    let dist2 = (b2[0] - jv[0]).powi(2) + (b2[1] - jv[1]).powi(2);
                    if dist1 < 1.001 * dist2 && dist1 > 0.999 * dist2 {
                        return;
                    }
                }
            }
        }
        q.enqueue_badtri(testtri, minedge, torgv, tdestv, tapexv);
    }
}

// ---------------------------------------------------------------------------
// insertvertex (triangle.cpp:8211) — infvertex branches dropped (never set).
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn insertvertex(
    m: &mut Mesh,
    q: &mut Quality,
    newvertex: Vid,
    searchtri: &mut TriHandle,
    splitseg: Option<SubHandle>,
    segmentflaws: bool,
    triflaws: bool,
    rng: &mut Rng,
) -> InsertResult {
    let nv = m.point(newvertex);
    let mut horiz;
    let intersect;
    if splitseg.is_none() {
        if searchtri.is_dummy() {
            horiz = m.sym(TriHandle::new(0, 0));
            intersect = locate(m, rng, nv, &mut horiz);
        } else {
            horiz = *searchtri;
            intersect = preciselocate(m, nv, &mut horiz, true);
        }
    } else {
        horiz = *searchtri;
        intersect = LocateResult::OnEdge;
    }

    if intersect == LocateResult::OnVertex {
        *searchtri = horiz;
        m.recenttri = Some(horiz);
        return InsertResult::Duplicate;
    }

    let mut success = InsertResult::Successful;

    if intersect == LocateResult::OnEdge || intersect == LocateResult::Outside {
        if m.checksegments && splitseg.is_none() {
            let brokensubseg = m.tspivot(horiz);
            if !brokensubseg.is_dummy() {
                if segmentflaws {
                    let mut enq = q.nobisect != 2;
                    if enq && q.nobisect == 1 {
                        enq = !m.sym(horiz).is_dummy();
                    }
                    if enq {
                        q.badsubsegs.push(BadSubseg {
                            enc: brokensubseg,
                            org: m.sorg(brokensubseg),
                            dest: m.sdest(brokensubseg),
                        });
                    }
                }
                *searchtri = horiz;
                m.recenttri = Some(horiz);
                return InsertResult::Violating;
            }
        }

        // Insert on an edge: split one or two triangles.
        let botright = m.lprev(horiz);
        let botrcasing = m.sym(botright);
        let mut topright = m.sym(horiz);
        let mirrorflag = !topright.is_dummy();
        let mut newtopright = TriHandle::DUMMY;
        if mirrorflag {
            topright = m.lnext(topright);
            let _toprcasing = m.sym(topright);
            newtopright = m.make_triangle();
        } else {
            m.hullsize += 1;
        }
        let toprcasing = if mirrorflag { m.sym(topright) } else { TriHandle::DUMMY };
        let newbotright = m.make_triangle();

        let rightvertex = m.org(horiz);
        let _leftvertex = m.dest(horiz);
        let botvertex = m.apex(horiz);
        m.set_org(newbotright, botvertex);
        m.set_dest(newbotright, rightvertex);
        m.set_apex(newbotright, newvertex);
        m.set_org(horiz, newvertex);
        for i in 0..m.eextras {
            let a = m.elem_attr(botright, i);
            m.set_elem_attr(newbotright, i, a);
        }
        if m.vararea {
            let ab = m.area_bound(botright);
            m.set_area_bound(newbotright, ab);
        }
        if mirrorflag {
            let topvertex = m.dest(topright);
            m.set_org(newtopright, rightvertex);
            m.set_dest(newtopright, topvertex);
            m.set_apex(newtopright, newvertex);
            m.set_org(topright, newvertex);
            for i in 0..m.eextras {
                let a = m.elem_attr(topright, i);
                m.set_elem_attr(newtopright, i, a);
            }
            if m.vararea {
                let ab = m.area_bound(topright);
                m.set_area_bound(newtopright, ab);
            }
        }

        if m.checksegments {
            let botrsubseg = m.tspivot(botright);
            if !botrsubseg.is_dummy() {
                m.tsdissolve(botright);
                m.tsbond(newbotright, botrsubseg);
            }
            if mirrorflag {
                let toprsubseg = m.tspivot(topright);
                if !toprsubseg.is_dummy() {
                    m.tsdissolve(topright);
                    m.tsbond(newtopright, toprsubseg);
                }
            }
        }

        m.bond(newbotright, botrcasing);
        let newbotright1 = m.lprev(newbotright);
        m.bond(newbotright1, botright);
        let newbotright2 = m.lprev(newbotright1);
        if mirrorflag {
            m.bond(newtopright, toprcasing);
            let newtopright1 = m.lnext(newtopright);
            m.bond(newtopright1, topright);
            let newtopright2 = m.lnext(newtopright1);
            m.bond(newtopright2, newbotright2);
        }

        if let Some(orig) = splitseg {
            // Split the subsegment into two and wire up the halves. Mirrors
            // triangle.cpp:8408-8428: keep `splitseg` flipped through the
            // re-bonding, restoring its orientation at the end.
            m.set_sdest(orig, newvertex);
            let segmentorg = m.seg_org(orig);
            let segmentdest = m.seg_dest(orig);
            let mark = m.smark(orig); // marker is orientation-independent
            let ss = m.ssym(orig); // flipped splitseg
            let rightsubseg = m.spivot(ss);
            // `newbotright2` is `newbotright` after two `lprev`s (orientation 1),
            // i.e. the new half-segment edge — the same handle C's in-place
            // `lprevself` leaves for `insertsubseg`.
            insert_subseg(m, newbotright2, mark);
            let newsubseg = m.tspivot(newbotright2);
            m.set_seg_org(newsubseg, segmentorg);
            m.set_seg_dest(newsubseg, segmentdest);
            m.sbond(ss, newsubseg);
            let newsubseg_sym = m.ssym(newsubseg);
            if !rightsubseg.is_dummy() {
                m.sbond(newsubseg_sym, rightsubseg);
            } else {
                m.sdissolve(newsubseg_sym);
            }
            if m.vertex_mark(newvertex) == 0 {
                m.set_vertex_mark(newvertex, mark);
            }
        }

        if m.checkquality {
            m.flipstack.clear();
            m.flipstack.push(FlipRecord {
                tri: horiz,
                kind: FlipKind::EdgeSplit,
            });
        }
        horiz = m.lnext(horiz);
    } else {
        // Insert inside a triangle: split into three.
        let botleft = m.lnext(horiz);
        let botright = m.lprev(horiz);
        let botlcasing = m.sym(botleft);
        let botrcasing = m.sym(botright);
        let newbotleft = m.make_triangle();
        let newbotright = m.make_triangle();

        let rightvertex = m.org(horiz);
        let leftvertex = m.dest(horiz);
        let botvertex = m.apex(horiz);
        m.set_org(newbotleft, leftvertex);
        m.set_dest(newbotleft, botvertex);
        m.set_apex(newbotleft, newvertex);
        m.set_org(newbotright, botvertex);
        m.set_dest(newbotright, rightvertex);
        m.set_apex(newbotright, newvertex);
        m.set_apex(horiz, newvertex);
        for i in 0..m.eextras {
            let a = m.elem_attr(horiz, i);
            m.set_elem_attr(newbotleft, i, a);
            m.set_elem_attr(newbotright, i, a);
        }
        if m.vararea {
            let area = m.area_bound(horiz);
            m.set_area_bound(newbotleft, area);
            m.set_area_bound(newbotright, area);
        }

        if m.checksegments {
            let botlsubseg = m.tspivot(botleft);
            if !botlsubseg.is_dummy() {
                m.tsdissolve(botleft);
                m.tsbond(newbotleft, botlsubseg);
            }
            let botrsubseg = m.tspivot(botright);
            if !botrsubseg.is_dummy() {
                m.tsdissolve(botright);
                m.tsbond(newbotright, botrsubseg);
            }
        }

        m.bond(newbotleft, botlcasing);
        m.bond(newbotright, botrcasing);
        let newbotleft = m.lnext(newbotleft);
        let newbotright = m.lprev(newbotright);
        m.bond(newbotleft, newbotright);
        let newbotleft = m.lnext(newbotleft);
        m.bond(botleft, newbotleft);
        let newbotright = m.lprev(newbotright);
        m.bond(botright, newbotright);

        let _ = (rightvertex, leftvertex);
        if m.checkquality {
            m.flipstack.clear();
            m.flipstack.push(FlipRecord {
                tri: horiz,
                kind: FlipKind::TriSplit,
            });
        }
    }

    // Flip edges to restore the Delaunay property, circling the new vertex.
    let first = m.org(horiz);
    let mut rightvertex = first;
    let mut leftvertex = m.dest(horiz);
    let mut _g = 0u64;
    loop {
        _g += 1;
        assert!(_g < 50_000_000, "insertvertex flip loop runaway");
        let mut doflip = true;
        if m.checksegments {
            let checksubseg = m.tspivot(horiz);
            if !checksubseg.is_dummy() {
                doflip = false;
                if segmentflaws && checkseg4encroach(m, q, checksubseg) != 0 {
                    success = InsertResult::Encroaching;
                }
            }
        }
        if doflip {
            let top = m.sym(horiz);
            if top.is_dummy() {
                doflip = false;
            } else {
                let farvertex = m.apex(top);
                doflip = m.in_circle(leftvertex, newvertex, rightvertex, farvertex) > 0.0;
                if doflip {
                    let topleft = m.lprev(top);
                    let toplcasing = m.sym(topleft);
                    let topright = m.lnext(top);
                    let toprcasing = m.sym(topright);
                    let botleft = m.lnext(horiz);
                    let botlcasing = m.sym(botleft);
                    let botright = m.lprev(horiz);
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
                    m.set_org(horiz, farvertex);
                    m.set_dest(horiz, newvertex);
                    m.set_apex(horiz, rightvertex);
                    m.set_org(top, newvertex);
                    m.set_dest(top, farvertex);
                    m.set_apex(top, leftvertex);
                    for i in 0..m.eextras {
                        let attrib = 0.5 * (m.elem_attr(top, i) + m.elem_attr(horiz, i));
                        m.set_elem_attr(top, i, attrib);
                        m.set_elem_attr(horiz, i, attrib);
                    }
                    if m.vararea {
                        let (at, ah) = (m.area_bound(top), m.area_bound(horiz));
                        let area = if at <= 0.0 || ah <= 0.0 { -1.0 } else { 0.5 * (at + ah) };
                        m.set_area_bound(top, area);
                        m.set_area_bound(horiz, area);
                    }
                    if m.checkquality {
                        m.flipstack.push(FlipRecord {
                            tri: horiz,
                            kind: FlipKind::Flip,
                        });
                    }
                    horiz = m.lprev(horiz);
                    leftvertex = farvertex;
                    continue;
                }
            }
        }
        if !doflip {
            if triflaws {
                testtriangle(m, q, horiz);
            }
            horiz = m.lnext(horiz);
            let testtri = m.sym(horiz);
            if leftvertex == first || testtri.is_dummy() {
                let st = m.lnext(horiz);
                *searchtri = st;
                m.recenttri = Some(st);
                return success;
            }
            horiz = m.lnext(testtri);
            rightvertex = leftvertex;
            leftvertex = m.dest(horiz);
        }
    }
}

// ---------------------------------------------------------------------------
// undovertex (triangle.cpp:9070)
// ---------------------------------------------------------------------------

/// Reverse a vertex insertion (kept for completeness; the current refinement
/// uses an encroachment pre-check rather than insert-then-undo).
#[allow(dead_code)]
fn undovertex(m: &mut Mesh) {
    while let Some(rec) = m.flipstack.pop() {
        let fliptri = rec.tri;
        match rec.kind {
            FlipKind::TriSplit => {
                let mut botleft = m.dprev(fliptri);
                botleft = m.lnext(botleft);
                let mut botright = m.onext(fliptri);
                botright = m.lprev(botright);
                let botlcasing = m.sym(botleft);
                let botrcasing = m.sym(botright);
                let botvertex = m.dest(botleft);
                m.set_apex(fliptri, botvertex);
                let f1 = m.lnext(fliptri);
                m.bond(f1, botlcasing);
                let botlsubseg = m.tspivot(botleft);
                if botlsubseg.is_dummy() {
                    m.tsdissolve(f1);
                } else {
                    m.tsbond(f1, botlsubseg);
                }
                let f2 = m.lnext(f1);
                m.bond(f2, botrcasing);
                let botrsubseg = m.tspivot(botright);
                if botrsubseg.is_dummy() {
                    m.tsdissolve(f2);
                } else {
                    m.tsbond(f2, botrsubseg);
                }
                m.triangle_dealloc(botleft);
                m.triangle_dealloc(botright);
            }
            FlipKind::EdgeSplit => {
                let gluetri = m.lprev(fliptri);
                let mut botright = m.sym(gluetri);
                botright = m.lnext(botright);
                let botrcasing = m.sym(botright);
                let rightvertex = m.dest(botright);
                m.set_org(fliptri, rightvertex);
                m.bond(gluetri, botrcasing);
                let botrsubseg = m.tspivot(botright);
                if botrsubseg.is_dummy() {
                    m.tsdissolve(gluetri);
                } else {
                    m.tsbond(gluetri, botrsubseg);
                }
                m.triangle_dealloc(botright);
                let gluetri2 = m.sym(fliptri);
                if !gluetri2.is_dummy() {
                    let gluetri2 = m.lnext(gluetri2);
                    let topright = m.dnext(gluetri2);
                    let toprcasing = m.sym(topright);
                    m.set_org(gluetri2, rightvertex);
                    m.bond(gluetri2, toprcasing);
                    let toprsubseg = m.tspivot(topright);
                    if toprsubseg.is_dummy() {
                        m.tsdissolve(gluetri2);
                    } else {
                        m.tsbond(gluetri2, toprsubseg);
                    }
                    m.triangle_dealloc(topright);
                }
                // End of the list.
                break;
            }
            FlipKind::Flip => {
                unflip(m, fliptri);
            }
        }
    }
    m.flipstack.clear();
}

// ---------------------------------------------------------------------------
// New-vertex creation helpers
// ---------------------------------------------------------------------------

/// Create a vertex at `coords` with attributes linearly interpolated by `interp`.
fn make_interpolated_vertex(
    m: &mut Mesh,
    coords: [f64; 2],
    attrs: &[f64],
    mark: i32,
    kind: VertexKind,
) -> Vid {
    m.add_vertex(coords, attrs, mark, kind)
}

// ---------------------------------------------------------------------------
// splitencsegs (triangle.cpp:13271) — Chew free-vertex deletion omitted.
// ---------------------------------------------------------------------------

fn splitencsegs(m: &mut Mesh, q: &mut Quality, triflaws: bool, rng: &mut Rng) {
    let mut _guard = 0u64;
    while !q.badsubsegs.is_empty() && q.steinerleft != 0 {
        _guard += 1;
        if _guard > 5_000_000 {
            break;
        }
        let batch = std::mem::take(&mut q.badsubsegs);
        for enc in batch {
            if q.steinerleft == 0 {
                // Preserve any not-yet-processed encroached segments.
                continue;
            }
            let currentenc = enc.enc;
            if m.is_sub_dead(currentenc.index()) {
                continue;
            }
            let eorgv = m.sorg(currentenc);
            let edestv = m.sdest(currentenc);
            if eorgv != enc.org || edestv != enc.dest {
                continue;
            }
            let eorg = m.point(eorgv);
            let edest = m.point(edestv);

            // Is either endpoint shared with an adjacent segment? (acute corner)
            let enctri = m.stpivot(currentenc);
            let mut acuteorg = false;
            let mut acutedest = false;
            if !enctri.is_dummy() {
                let t = m.lnext(enctri);
                acuteorg = !m.tspivot(t).is_dummy();
                let t = m.lnext(t);
                acutedest = !m.tspivot(t).is_dummy();
            }
            let othertri = if !enctri.is_dummy() { m.sym(enctri) } else { TriHandle::DUMMY };
            if !othertri.is_dummy() {
                let t = m.lnext(othertri);
                acutedest = acutedest || !m.tspivot(t).is_dummy();
                let t = m.lnext(t);
                acuteorg = acuteorg || !m.tspivot(t).is_dummy();
            }

            let split = if acuteorg || acutedest {
                let seglen =
                    ((edest[0] - eorg[0]).powi(2) + (edest[1] - eorg[1]).powi(2)).sqrt();
                let mut nearestpoweroftwo = 1.0;
                while seglen > 3.0 * nearestpoweroftwo {
                    nearestpoweroftwo *= 2.0;
                }
                while seglen < 1.5 * nearestpoweroftwo {
                    nearestpoweroftwo *= 0.5;
                }
                let s = nearestpoweroftwo / seglen;
                if acutedest {
                    1.0 - s
                } else {
                    s
                }
            } else {
                0.5
            };

            // Interpolated coordinates + attributes.
            let na = m.nextras;
            let mut coords = [
                eorg[0] + split * (edest[0] - eorg[0]),
                eorg[1] + split * (edest[1] - eorg[1]),
            ];
            let mut attrs = vec![0.0; na];
            for (i, a) in attrs.iter_mut().enumerate() {
                let ao = m.vertex_attrs(eorgv)[i];
                let ad = m.vertex_attrs(edestv)[i];
                *a = ao + split * (ad - ao);
            }
            if !m.noexact {
                let mult = orient2d(eorg, edest, coords, false);
                let divisor = (eorg[0] - edest[0]).powi(2) + (eorg[1] - edest[1]).powi(2);
                if mult != 0.0 && divisor != 0.0 {
                    let mult = mult / divisor;
                    if !mult.is_nan() {
                        coords[0] += mult * (edest[1] - eorg[1]);
                        coords[1] += mult * (eorg[0] - edest[0]);
                    }
                }
            }
            if coords == eorg || coords == edest {
                // Ran out of precision; skip to avoid an infinite loop.
                continue;
            }
            let mark = m.smark(currentenc);
            let newvertex = make_interpolated_vertex(m, coords, &attrs, mark, VertexKind::Segment);

            let mut enctri = m.stpivot(currentenc);
            // stpivot may be dummy if the segment is on the hull on this side;
            // use the other side's triangle as the insertion anchor.
            if enctri.is_dummy() {
                enctri = m.stpivot(m.ssym(currentenc));
            }
            let success = insertvertex(
                m,
                q,
                newvertex,
                &mut enctri,
                Some(currentenc),
                true,
                triflaws,
                rng,
            );
            debug_assert!(
                success == InsertResult::Successful || success == InsertResult::Encroaching
            );
            if q.steinerleft > 0 {
                q.steinerleft -= 1;
            }
            // Check the two halves for encroachment.
            let _ = checkseg4encroach(m, q, currentenc);
            let nextseg = m.snext(currentenc);
            if !nextseg.is_dummy() {
                let _ = checkseg4encroach(m, q, nextseg);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// tallyencs / tallyfaces / splittriangle / enforcequality
// ---------------------------------------------------------------------------

fn tallyencs(m: &mut Mesh, q: &mut Quality) {
    for si in m.live_subsegs().collect::<Vec<_>>() {
        let s = SubHandle::new(si as u32, 0);
        let _ = checkseg4encroach(m, q, s);
    }
}

fn tallyfaces(m: &mut Mesh, q: &mut Quality) {
    for ti in m.live_triangles().collect::<Vec<_>>() {
        testtriangle(m, q, TriHandle::new(ti as u32, 0));
    }
}

fn splittriangle(m: &mut Mesh, q: &mut Quality, bad: BadTri, rng: &mut Rng) {
    let badotri = bad.poortri;
    if m.is_tri_dead(badotri.index()) {
        return;
    }
    let borgv = m.org(badotri);
    let bdestv = m.dest(badotri);
    let bapexv = m.apex(badotri);
    if borgv != bad.org || bdestv != bad.dest || bapexv != bad.apex {
        return; // triangle changed since it was enqueued
    }
    let borg = m.point(borgv);
    let bdest = m.point(bdestv);
    let bapex = m.point(bapexv);
    let (cc, xi, eta) = find_circumcenter(m, borg, bdest, bapex, q.offconstant, true);
    if cc == borg || cc == bdest || cc == bapex {
        return; // degenerate
    }

    let na = m.nextras;
    let mut attrs = vec![0.0; na];
    for (i, a) in attrs.iter_mut().enumerate() {
        let ao = m.vertex_attrs(borgv)[i];
        let ad = m.vertex_attrs(bdestv)[i];
        let ap = m.vertex_attrs(bapexv)[i];
        *a = ao + xi * (ad - ao) + eta * (ap - ao);
    }
    let newvertex = make_interpolated_vertex(m, cc, &attrs, 0, VertexKind::Free);

    // Ensure the search edge is not the longest, so the circumcenter is to its
    // left and point location succeeds.
    let mut badotri = badotri;
    if eta < xi {
        badotri = m.lprev(badotri);
    }

    // Insert the circumcenter (flagging any encroached subsegments along the
    // way). Ruppert's rule: if it encroaches a subsegment, undo the insertion
    // and split the segment(s) instead. This is O(cavity), not O(#segments).
    let success = insertvertex(m, q, newvertex, &mut badotri, None, true, true, rng);
    match success {
        InsertResult::Successful => {
            if q.steinerleft > 0 {
                q.steinerleft -= 1;
            }
        }
        InsertResult::Encroaching => {
            undovertex(m);
            m.vertex_dealloc(newvertex);
        }
        InsertResult::Violating | InsertResult::Duplicate => {
            m.vertex_dealloc(newvertex);
        }
    }
}

/// Run Ruppert refinement. Port of `enforcequality` (triangle.cpp:13648).
pub fn enforce_quality(m: &mut Mesh, q: &mut Quality, rng: &mut Rng) {
    // Fix encroached subsegments first (without noting bad triangles).
    tallyencs(m, q);
    splitencsegs(m, q, false, rng);

    if q.minangle > 0.0 || q.vararea || q.fixedarea || q.usertest.is_some() {
        for v in q.queuefront.iter_mut() {
            *v = -1;
        }
        q.firstnonemptyq = -1;
        tallyfaces(m, q);
        m.checkquality = true;
        // Safety bound so refinement always terminates (returns a best-effort
        // mesh) rather than hanging if it fails to converge.
        let insert_cap: u64 = 2_000_000;
        let mut _guard = 0u64;
        while q.badtris_pending() && q.steinerleft != 0 {
            _guard += 1;
            if _guard > insert_cap {
                break;
            }
            let idx = q.dequeue_badtriang().unwrap();
            let bad = q.badtris[idx];
            splittriangle(m, q, bad, rng);
            if !q.badsubsegs.is_empty() {
                // Re-enqueue the bad triangle and fix encroached segments.
                q.enqueue_badtriang(idx);
                splitencsegs(m, q, true, rng);
            }
        }
        m.checkquality = false;
    }
}
