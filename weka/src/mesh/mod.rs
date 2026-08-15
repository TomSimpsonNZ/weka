//! The mesh: index-based arenas of triangles, subsegments and vertices, with the
//! O(1) navigation primitives ported from Triangle's `otri`/`osub` macros
//! (triangle.cpp:947-1331). Arena index `0` is the `dummytri`/`dummysub`
//! sentinel, mirroring Triangle's "outer space" object.

pub mod handle;
pub mod records;

pub use handle::{SubHandle, TriHandle, Vid, MINUS1MOD3, NO_VERTEX, PLUS1MOD3};
pub use records::{SubRecord, TriRecord, VertRecord, VertexKind};

/// Which kind of transformation a [`FlipRecord`] records, for undo
/// (Triangle encodes this via sentinel `prevflip` pointer values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlipKind {
    /// A triangle split into three (vertex inserted inside a triangle).
    TriSplit,
    /// Two triangles split into four (vertex inserted on an edge).
    EdgeSplit,
    /// An edge flip.
    Flip,
}

/// One entry in the vertex-insertion undo log.
#[derive(Clone, Copy, Debug)]
pub struct FlipRecord {
    pub tri: TriHandle,
    pub kind: FlipKind,
}

use crate::predicates::orient2d;

/// The triangulation arena and all per-element side data.
pub struct Mesh {
    tris: Vec<TriRecord>,
    subs: Vec<SubRecord>,
    verts: Vec<VertRecord>,

    /// Adjoining subsegments per triangle (3 each), kept in lockstep with `tris`
    /// only when `use_segments` is set; empty otherwise so the Delaunay-only path
    /// carries a smaller, cache-friendlier triangle record.
    tri_subs: Vec<[SubHandle; 3]>,

    tri_free: Vec<u32>,
    sub_free: Vec<u32>,
    vert_free: Vec<u32>,

    /// `nextras` attributes per vertex, row-major.
    vert_attrs: Vec<f64>,
    /// `eextras` attributes per triangle, row-major.
    tri_attrs: Vec<f64>,
    /// One area bound per triangle (`-1.0` = unconstrained).
    tri_area: Vec<f64>,
    /// Edge-midpoint node ids per triangle for quadratic (`-o2`) elements;
    /// empty unless `highorder` has run.
    tri_high: Vec<[Vid; 3]>,

    pub nextras: usize,
    pub eextras: usize,
    pub use_segments: bool,
    pub vararea: bool,

    /// Disable robust arithmetic (Triangle's `-X`).
    pub noexact: bool,

    /// Number of input vertices (before any Steiner points are added).
    pub invertices: usize,
    /// Count of "undead" (duplicate, ignored) input vertices.
    pub undeads: usize,
    /// Whether subsegments are present (enables flip's subseg rebonding etc.).
    pub checksegments: bool,
    /// Most recently located triangle, for point-location warm starts.
    pub recenttri: Option<TriHandle>,
    /// Convex-hull edge count (mutated by hole carving).
    pub hullsize: i64,
    /// Quality-meshing stage active (enables flip-stack recording for undo).
    pub checkquality: bool,
    /// Undo log of mesh transformations during a single vertex insertion.
    pub flipstack: Vec<FlipRecord>,
    /// Input bounding box (set by the point loader).
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,

    // Live element counts (Triangle's `pool.items`).
    tri_items: usize,
    sub_items: usize,
    vert_items: usize,
}

impl Mesh {
    /// Create an empty mesh. Index 0 of the triangle and subsegment arenas is
    /// initialized as the sentinel ("outer space"), matching `dummyinit`.
    pub fn new(nextras: usize, eextras: usize, use_segments: bool, vararea: bool) -> Self {
        let mut m = Mesh {
            tris: Vec::new(),
            subs: Vec::new(),
            verts: Vec::new(),
            tri_subs: Vec::new(),
            tri_free: Vec::new(),
            sub_free: Vec::new(),
            vert_free: Vec::new(),
            vert_attrs: Vec::new(),
            tri_attrs: Vec::new(),
            tri_area: Vec::new(),
            tri_high: Vec::new(),
            nextras,
            eextras,
            use_segments,
            vararea,
            noexact: false,
            invertices: 0,
            undeads: 0,
            checksegments: false,
            recenttri: None,
            hullsize: 0,
            checkquality: false,
            flipstack: Vec::new(),
            xmin: 0.0,
            xmax: 0.0,
            ymin: 0.0,
            ymax: 0.0,
            tri_items: 0,
            sub_items: 0,
            vert_items: 0,
        };
        // dummytri at index 0: neighbors all point to itself, no vertices/subs.
        m.tris.push(TriRecord::fresh());
        if use_segments {
            m.tri_subs.push([SubHandle::DUMMY; 3]);
        }
        m.tri_attrs.resize(eextras, 0.0);
        if vararea {
            m.tri_area.push(-1.0);
        }
        // dummysub at index 0.
        m.subs.push(SubRecord::fresh());
        m
    }

    /// Reserve arena capacity for roughly `n` input points (a planar
    /// triangulation has ~`2n` triangles and ~`3n` edges), avoiding repeated
    /// reallocation during the build.
    pub fn reserve(&mut self, n: usize) {
        self.verts.reserve(n);
        self.vert_attrs.reserve(n * self.nextras);
        let tris = 2 * n + 1;
        self.tris.reserve(tris);
        self.tri_attrs.reserve(tris * self.eextras);
        if self.vararea {
            self.tri_area.reserve(tris);
        }
        if self.use_segments {
            self.tri_subs.reserve(tris);
        }
    }

    // ----- counts ---------------------------------------------------------

    pub fn num_triangles(&self) -> usize {
        self.tri_items
    }
    pub fn num_subsegs(&self) -> usize {
        self.sub_items
    }
    pub fn num_vertices(&self) -> usize {
        self.vert_items
    }
    /// Total triangle arena slots (live + dead + the dummy at index 0).
    pub fn tri_arena_len(&self) -> usize {
        self.tris.len()
    }
    /// Total subsegment arena slots.
    pub fn sub_arena_len(&self) -> usize {
        self.subs.len()
    }
    /// Whether triangle arena slot `i` is dead (deallocated) or the dummy.
    #[inline]
    pub fn is_tri_dead(&self, i: usize) -> bool {
        i == 0 || self.tris[i].dead
    }
    /// Whether subsegment arena slot `i` is dead (deallocated) or the dummy.
    #[inline]
    pub fn is_sub_dead(&self, i: usize) -> bool {
        i == 0 || self.subs[i].dead
    }

    // ----- allocation -----------------------------------------------------

    /// Create a new triangle (orientation 0), reusing a freed slot if available
    /// (LIFO, matching Triangle's `deaditemstack`).
    pub fn make_triangle(&mut self) -> TriHandle {
        let idx = if let Some(i) = self.tri_free.pop() {
            let rec = &mut self.tris[i as usize];
            *rec = TriRecord::fresh();
            for a in &mut self.tri_attrs[i as usize * self.eextras..(i as usize + 1) * self.eextras]
            {
                *a = 0.0;
            }
            if self.vararea {
                self.tri_area[i as usize] = -1.0;
            }
            if self.use_segments {
                self.tri_subs[i as usize] = [SubHandle::DUMMY; 3];
            }
            i
        } else {
            let i = self.tris.len() as u32;
            self.tris.push(TriRecord::fresh());
            if self.eextras > 0 {
                self.tri_attrs.resize(self.tri_attrs.len() + self.eextras, 0.0);
            }
            if self.vararea {
                self.tri_area.push(-1.0);
            }
            if self.use_segments {
                self.tri_subs.push([SubHandle::DUMMY; 3]);
            }
            i
        };
        self.tri_items += 1;
        TriHandle::new(idx, 0)
    }

    /// Create a new subsegment (orientation 0).
    pub fn make_subseg(&mut self) -> SubHandle {
        let idx = if let Some(i) = self.sub_free.pop() {
            self.subs[i as usize] = SubRecord::fresh();
            i
        } else {
            let i = self.subs.len() as u32;
            self.subs.push(SubRecord::fresh());
            i
        };
        self.sub_items += 1;
        SubHandle::new(idx, 0)
    }

    /// Deallocate a triangle (its slot becomes reusable).
    pub fn triangle_dealloc(&mut self, h: TriHandle) {
        let i = h.index();
        debug_assert!(i != 0 && !self.tris[i].dead);
        self.tris[i].dead = true;
        self.tri_free.push(i as u32);
        self.tri_items -= 1;
    }

    /// Deallocate a subsegment.
    pub fn subseg_dealloc(&mut self, h: SubHandle) {
        let i = h.index();
        debug_assert!(i != 0 && !self.subs[i].dead);
        self.subs[i].dead = true;
        self.sub_free.push(i as u32);
        self.sub_items -= 1;
    }

    /// Add a vertex, returning its id.
    pub fn add_vertex(&mut self, xy: [f64; 2], attrs: &[f64], mark: i32, kind: VertexKind) -> Vid {
        debug_assert_eq!(attrs.len(), self.nextras);
        let rec = VertRecord {
            xy,
            mark,
            kind,
            tri: TriHandle::DUMMY,
            dead: false,
        };
        let id = if let Some(i) = self.vert_free.pop() {
            self.verts[i as usize] = rec;
            let base = i as usize * self.nextras;
            self.vert_attrs[base..base + self.nextras].copy_from_slice(attrs);
            i
        } else {
            let i = self.verts.len() as u32;
            self.verts.push(rec);
            self.vert_attrs.extend_from_slice(attrs);
            i
        };
        self.vert_items += 1;
        id
    }

    /// Discard the vertex free-list so subsequent `add_vertex` calls allocate
    /// fresh slots (used before adding high-order nodes so corner vertices keep
    /// the lower output indices, matching Triangle).
    pub fn clear_vertex_freelist(&mut self) {
        self.vert_free.clear();
    }

    /// Quadratic edge-midpoint node side-array (set by `highorder`).
    pub fn set_tri_high(&mut self, high: Vec<[Vid; 3]>) {
        self.tri_high = high;
    }
    /// Whether quadratic high-order nodes have been generated.
    pub fn has_high_order(&self) -> bool {
        !self.tri_high.is_empty()
    }
    /// The edge-midpoint node on edge `orient` of triangle `h`.
    #[inline]
    pub fn high_node(&self, h: TriHandle) -> Vid {
        self.tri_high[h.index()][h.orient()]
    }

    /// Deallocate a vertex slot (e.g. a rejected Steiner point).
    pub fn vertex_dealloc(&mut self, v: Vid) {
        debug_assert!(!self.verts[v as usize].dead);
        self.verts[v as usize].dead = true;
        self.vert_free.push(v);
        self.vert_items -= 1;
    }

    /// Total vertex arena slots (live + dead).
    pub fn vert_arena_len(&self) -> usize {
        self.verts.len()
    }
    /// Whether vertex slot `v` is dead (deallocated).
    #[inline]
    pub fn vertex_is_dead(&self, v: Vid) -> bool {
        self.verts[v as usize].dead
    }

    // ----- vertex accessors ----------------------------------------------

    #[inline]
    pub fn point(&self, v: Vid) -> [f64; 2] {
        self.verts[v as usize].xy
    }
    #[inline]
    pub fn vertex(&self, v: Vid) -> &VertRecord {
        &self.verts[v as usize]
    }
    #[inline]
    pub fn vertex_mut(&mut self, v: Vid) -> &mut VertRecord {
        &mut self.verts[v as usize]
    }
    pub fn vertex_attrs(&self, v: Vid) -> &[f64] {
        let base = v as usize * self.nextras;
        &self.vert_attrs[base..base + self.nextras]
    }
    #[inline]
    pub fn vertex_kind(&self, v: Vid) -> VertexKind {
        self.verts[v as usize].kind
    }
    #[inline]
    pub fn set_vertex_kind(&mut self, v: Vid, kind: VertexKind) {
        self.verts[v as usize].kind = kind;
    }
    #[inline]
    pub fn vertex_mark(&self, v: Vid) -> i32 {
        self.verts[v as usize].mark
    }
    #[inline]
    pub fn set_vertex_mark(&mut self, v: Vid, mark: i32) {
        self.verts[v as usize].mark = mark;
    }

    /// One incident triangle recorded for vertex `v` (Triangle's `vertex2tri`);
    /// `DUMMY` if unset.
    #[inline]
    pub fn vertex2tri(&self, v: Vid) -> TriHandle {
        self.verts[v as usize].tri
    }
    #[inline]
    pub fn set_vertex2tri(&mut self, v: Vid, h: TriHandle) {
        self.verts[v as usize].tri = h;
    }

    // ----- robust predicates over vertex ids -----------------------------

    /// `counterclockwise(a, b, c)` — positive if CCW. Uses `self.noexact`.
    #[inline]
    pub fn ccw(&self, a: Vid, b: Vid, c: Vid) -> f64 {
        orient2d(self.point(a), self.point(b), self.point(c), self.noexact)
    }
    /// `incircle(a, b, c, d)` — positive if `d` is inside circle (a,b,c) (CCW).
    #[inline]
    pub fn in_circle(&self, a: Vid, b: Vid, c: Vid, d: Vid) -> f64 {
        crate::predicates::incircle(
            self.point(a),
            self.point(b),
            self.point(c),
            self.point(d),
            self.noexact,
        )
    }

    // ----- hull start edge (Triangle stores it in dummytri[0]) -----------

    /// Record an edge on the convex hull as the point-location start (the C code
    /// stores this in `dummytri[0]`).
    #[inline]
    pub fn set_hull_edge(&mut self, h: TriHandle) {
        self.tris[0].neigh[0] = h;
    }
    /// The recorded convex-hull start edge.
    #[inline]
    pub fn hull_edge(&self) -> TriHandle {
        self.tris[0].neigh[0]
    }

    // ----- triangle attribute / area accessors ---------------------------

    pub fn elem_attr(&self, h: TriHandle, i: usize) -> f64 {
        self.tri_attrs[h.index() * self.eextras + i]
    }
    pub fn set_elem_attr(&mut self, h: TriHandle, i: usize, val: f64) {
        self.tri_attrs[h.index() * self.eextras + i] = val;
    }
    pub fn area_bound(&self, h: TriHandle) -> f64 {
        // `tri_area` is only maintained when per-element area constraints are in
        // use; otherwise every triangle is unconstrained (`-1.0`).
        if self.vararea {
            self.tri_area[h.index()]
        } else {
            -1.0
        }
    }
    pub fn set_area_bound(&mut self, h: TriHandle, val: f64) {
        self.tri_area[h.index()] = val;
    }

    // ----- triangle navigation (triangle.cpp:947-1102) -------------------

    /// The adjoining triangle across `h`'s edge (`sym`).
    #[inline]
    pub fn sym(&self, h: TriHandle) -> TriHandle {
        self.tris[h.index()].neigh[h.orient()]
    }
    /// Rotate counterclockwise within the triangle (`lnext`).
    #[inline]
    pub fn lnext(&self, h: TriHandle) -> TriHandle {
        TriHandle::new(h.index() as u32, PLUS1MOD3[h.orient()])
    }
    /// Rotate clockwise within the triangle (`lprev`).
    #[inline]
    pub fn lprev(&self, h: TriHandle) -> TriHandle {
        TriHandle::new(h.index() as u32, MINUS1MOD3[h.orient()])
    }
    /// Next edge about the origin vertex (`onext = sym(lprev)`).
    #[inline]
    pub fn onext(&self, h: TriHandle) -> TriHandle {
        self.sym(self.lprev(h))
    }
    /// Previous edge about the origin vertex (`oprev = lnext(sym)`).
    #[inline]
    pub fn oprev(&self, h: TriHandle) -> TriHandle {
        self.lnext(self.sym(h))
    }
    /// Next edge about the destination vertex (`dnext = lprev(sym)`).
    #[inline]
    pub fn dnext(&self, h: TriHandle) -> TriHandle {
        self.lprev(self.sym(h))
    }
    /// Previous edge about the destination vertex (`dprev = sym(lnext)`).
    #[inline]
    pub fn dprev(&self, h: TriHandle) -> TriHandle {
        self.sym(self.lnext(h))
    }
    /// Next edge of the adjoining triangle (`rnext = sym(lnext(sym))`).
    #[inline]
    pub fn rnext(&self, h: TriHandle) -> TriHandle {
        self.sym(self.lnext(self.sym(h)))
    }
    /// Previous edge of the adjoining triangle (`rprev = sym(lprev(sym))`).
    #[inline]
    pub fn rprev(&self, h: TriHandle) -> TriHandle {
        self.sym(self.lprev(self.sym(h)))
    }

    #[inline]
    pub fn org(&self, h: TriHandle) -> Vid {
        self.tris[h.index()].verts[PLUS1MOD3[h.orient()]]
    }
    #[inline]
    pub fn dest(&self, h: TriHandle) -> Vid {
        self.tris[h.index()].verts[MINUS1MOD3[h.orient()]]
    }
    #[inline]
    pub fn apex(&self, h: TriHandle) -> Vid {
        self.tris[h.index()].verts[h.orient()]
    }
    #[inline]
    pub fn set_org(&mut self, h: TriHandle, v: Vid) {
        self.tris[h.index()].verts[PLUS1MOD3[h.orient()]] = v;
    }
    #[inline]
    pub fn set_dest(&mut self, h: TriHandle, v: Vid) {
        self.tris[h.index()].verts[MINUS1MOD3[h.orient()]] = v;
    }
    #[inline]
    pub fn set_apex(&mut self, h: TriHandle, v: Vid) {
        self.tris[h.index()].verts[h.orient()] = v;
    }

    /// Bond two triangle edges to each other (`bond`).
    #[inline]
    pub fn bond(&mut self, a: TriHandle, b: TriHandle) {
        self.tris[a.index()].neigh[a.orient()] = b;
        self.tris[b.index()].neigh[b.orient()] = a;
    }
    /// Detach a triangle edge so it faces outer space (`dissolve`).
    #[inline]
    pub fn dissolve(&mut self, h: TriHandle) {
        self.tris[h.index()].neigh[h.orient()] = TriHandle::DUMMY;
    }

    // ----- subsegment navigation (triangle.cpp:1167-1256) ----------------

    #[inline]
    pub fn ssym(&self, s: SubHandle) -> SubHandle {
        SubHandle::new(s.index() as u32, 1 - s.orient())
    }
    #[inline]
    pub fn spivot(&self, s: SubHandle) -> SubHandle {
        self.subs[s.index()].adj[s.orient()]
    }
    #[inline]
    pub fn snext(&self, s: SubHandle) -> SubHandle {
        self.subs[s.index()].adj[1 - s.orient()]
    }
    #[inline]
    pub fn sorg(&self, s: SubHandle) -> Vid {
        self.subs[s.index()].edge[s.orient()]
    }
    #[inline]
    pub fn sdest(&self, s: SubHandle) -> Vid {
        self.subs[s.index()].edge[1 - s.orient()]
    }
    #[inline]
    pub fn set_sorg(&mut self, s: SubHandle, v: Vid) {
        self.subs[s.index()].edge[s.orient()] = v;
    }
    #[inline]
    pub fn set_sdest(&mut self, s: SubHandle, v: Vid) {
        self.subs[s.index()].edge[1 - s.orient()] = v;
    }
    #[inline]
    pub fn seg_org(&self, s: SubHandle) -> Vid {
        self.subs[s.index()].seg[s.orient()]
    }
    #[inline]
    pub fn seg_dest(&self, s: SubHandle) -> Vid {
        self.subs[s.index()].seg[1 - s.orient()]
    }
    #[inline]
    pub fn set_seg_org(&mut self, s: SubHandle, v: Vid) {
        self.subs[s.index()].seg[s.orient()] = v;
    }
    #[inline]
    pub fn set_seg_dest(&mut self, s: SubHandle, v: Vid) {
        self.subs[s.index()].seg[1 - s.orient()] = v;
    }
    #[inline]
    pub fn smark(&self, s: SubHandle) -> i32 {
        self.subs[s.index()].marker
    }
    #[inline]
    pub fn set_smark(&mut self, s: SubHandle, v: i32) {
        self.subs[s.index()].marker = v;
    }
    #[inline]
    pub fn sbond(&mut self, a: SubHandle, b: SubHandle) {
        self.subs[a.index()].adj[a.orient()] = b;
        self.subs[b.index()].adj[b.orient()] = a;
    }
    #[inline]
    pub fn sdissolve(&mut self, s: SubHandle) {
        self.subs[s.index()].adj[s.orient()] = SubHandle::DUMMY;
    }

    // ----- triangle <-> subsegment bonds (triangle.cpp:1285-1312) --------

    #[inline]
    pub fn tspivot(&self, h: TriHandle) -> SubHandle {
        // With no segments there are no subsegments; `tri_subs` is then empty.
        if self.use_segments {
            self.tri_subs[h.index()][h.orient()]
        } else {
            SubHandle::DUMMY
        }
    }
    #[inline]
    pub fn stpivot(&self, s: SubHandle) -> TriHandle {
        self.subs[s.index()].tri[s.orient()]
    }
    #[inline]
    pub fn tsbond(&mut self, h: TriHandle, s: SubHandle) {
        self.tri_subs[h.index()][h.orient()] = s;
        self.subs[s.index()].tri[s.orient()] = h;
    }
    #[inline]
    pub fn tsdissolve(&mut self, h: TriHandle) {
        self.tri_subs[h.index()][h.orient()] = SubHandle::DUMMY;
    }
    #[inline]
    pub fn stdissolve(&mut self, s: SubHandle) {
        self.subs[s.index()].tri[s.orient()] = TriHandle::DUMMY;
    }

    // ----- infection flag (hole carving) ---------------------------------

    #[inline]
    pub fn infect(&mut self, h: TriHandle) {
        self.tris[h.index()].infected = true;
    }
    #[inline]
    pub fn uninfect(&mut self, h: TriHandle) {
        self.tris[h.index()].infected = false;
    }
    #[inline]
    pub fn infected(&self, h: TriHandle) -> bool {
        self.tris[h.index()].infected
    }

    // ----- traversal ------------------------------------------------------

    /// Indices of all live triangles (skips the dummy at 0 and dead slots).
    pub fn live_triangles(&self) -> impl Iterator<Item = usize> + '_ {
        (1..self.tris.len()).filter(move |&i| !self.tris[i].dead)
    }

    /// Indices of all live subsegments.
    pub fn live_subsegs(&self) -> impl Iterator<Item = usize> + '_ {
        (1..self.subs.len()).filter(move |&i| !self.subs[i].dead)
    }

    // ----- consistency check (port of checkmesh, triangle.cpp:6709) -------

    /// Verify topological consistency: no inverted triangles, neighbor bonds are
    /// reciprocal, and shared edges agree on their endpoints. Returns the number
    /// of problems found (0 = consistent). Always uses exact arithmetic.
    pub fn check_mesh(&self) -> usize {
        let mut horrors = 0usize;
        for i in self.live_triangles() {
            for orient in 0..3 {
                let h = TriHandle::new(i as u32, orient);
                let triorg = self.org(h);
                let tridest = self.dest(h);
                if orient == 0 {
                    let triapex = self.apex(h);
                    if self.is_real(triorg) && self.is_real(tridest) && self.is_real(triapex) {
                        let det = orient2d(
                            self.point(triorg),
                            self.point(tridest),
                            self.point(triapex),
                            false,
                        );
                        if det <= 0.0 {
                            horrors += 1; // inverted or degenerate
                        }
                    }
                }
                let oppotri = self.sym(h);
                if !oppotri.is_dummy() {
                    let oppooppotri = self.sym(oppotri);
                    if oppooppotri != h {
                        horrors += 1; // asymmetric bond
                    }
                    let oppoorg = self.org(oppotri);
                    let oppodest = self.dest(oppotri);
                    if triorg != oppodest || tridest != oppoorg {
                        horrors += 1; // mismatched shared edge
                    }
                }
            }
        }
        horrors
    }

    #[inline]
    fn is_real(&self, v: Vid) -> bool {
        v != NO_VERTEX
    }

    /// (debug) Audit subsegment↔triangle bonds for reciprocity. Returns a list
    /// of human-readable problems (empty = all bonds consistent).
    pub fn subseg_problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        for si in self.live_subsegs() {
            for sso in 0..2 {
                let s = SubHandle::new(si as u32, sso);
                let t = self.stpivot(s);
                if t.is_dummy() {
                    continue;
                }
                // The triangle this subseg points to should point back at it.
                let back = self.tspivot(t);
                if back.index() != si {
                    out.push(format!(
                        "subseg {si}.{sso} -> {t:?} but tspivot(tri) = {back:?} (not subseg {si})"
                    ));
                }
                // Endpoints should agree: sorg(s)==dest(t), sdest(s)==org(t).
                if self.sorg(s) != self.dest(t) || self.sdest(s) != self.org(t) {
                    out.push(format!(
                        "subseg {si}.{sso} endpoints ({},{}) != triangle {t:?} edge ({},{})",
                        self.sorg(s),
                        self.sdest(s),
                        self.dest(t),
                        self.org(t)
                    ));
                }
            }
        }
        out
    }

    /// (debug) Count interior edges that violate the empty-circumcircle property
    /// (ignoring subsegment edges). Zero for a true (constrained) Delaunay mesh.
    pub fn count_non_delaunay(&self) -> usize {
        let mut bad = 0;
        for i in self.live_triangles() {
            for orient in 0..3 {
                let h = TriHandle::new(i as u32, orient);
                let opp = self.sym(h);
                if opp.is_dummy() || opp.index() < i {
                    continue;
                }
                if !self.tspivot(h).is_dummy() {
                    continue; // subsegment edge: not required to be Delaunay
                }
                let (a, b, c) = (self.org(h), self.dest(h), self.apex(h));
                let d = self.apex(opp);
                if [a, b, c, d].contains(&NO_VERTEX) {
                    continue;
                }
                if crate::predicates::incircle(
                    self.point(a),
                    self.point(b),
                    self.point(c),
                    self.point(d),
                    false,
                ) > 0.0
                {
                    bad += 1;
                }
            }
        }
        bad
    }
}
