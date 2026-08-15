//! Arena record types. These replace Triangle's untyped `REAL**` blobs
//! (triangle.cpp tri layout: `[0..2]` neighbors, `[3..5]` vertices, `[6..8]`
//! subsegs; subseg layout: `[0,1]` adj, `[2,3]` edge verts, `[4,5]` segment
//! verts, `[6,7]` adjacent triangles, `[8]` marker). Per-element attribute and
//! area data live in side-arrays on the [`super::Mesh`] instead of being packed
//! into these records.

use super::handle::{SubHandle, TriHandle, Vid, NO_VERTEX};

/// A triangle: three neighbor edges, three corner vertices, three subsegment
/// edges, plus mesh-management flags.
#[derive(Clone, Debug)]
pub struct TriRecord {
    /// Adjoining triangles, one per edge (`DUMMY` = exterior / "outer space").
    pub neigh: [TriHandle; 3],
    /// Corner vertices. `NO_VERTEX` until set.
    pub verts: [Vid; 3],
    /// True if this slot has been deallocated (skipped during traversal).
    pub dead: bool,
    /// Flood-fill marker used by hole carving (Triangle's "infect" bit).
    pub infected: bool,
}

impl TriRecord {
    /// A freshly-made triangle: all edges face outer space, no vertices set.
    /// Adjoining subsegments live in the mesh's `tri_subs` side-array (only
    /// allocated when the triangulation actually uses segments), keeping this
    /// record small and cache-friendly on the pure-Delaunay path.
    pub fn fresh() -> Self {
        TriRecord {
            neigh: [TriHandle::DUMMY; 3],
            verts: [NO_VERTEX; 3],
            dead: false,
            infected: false,
        }
    }
}

/// A subsegment: the two oriented copies' adjacencies, the edge's two vertices,
/// the parent segment's two endpoints, the two adjoining triangles, and a marker.
#[derive(Clone, Debug)]
pub struct SubRecord {
    /// Adjacent subsegments (one per orientation).
    pub adj: [SubHandle; 2],
    /// This edge's endpoints (indexed by orientation).
    pub edge: [Vid; 2],
    /// The originating segment's endpoints (indexed by orientation).
    pub seg: [Vid; 2],
    /// Adjoining triangles (one per orientation).
    pub tri: [TriHandle; 2],
    /// Boundary marker.
    pub marker: i32,
    /// True if this slot has been deallocated.
    pub dead: bool,
}

impl SubRecord {
    pub fn fresh() -> Self {
        SubRecord {
            adj: [SubHandle::DUMMY; 2],
            edge: [NO_VERTEX; 2],
            seg: [NO_VERTEX; 2],
            tri: [TriHandle::DUMMY; 2],
            marker: 0,
            dead: false,
        }
    }
}

/// Vertex classification (triangle.cpp:305 `INPUTVERTEX`..`UNDEADVERTEX`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum VertexKind {
    Input = 0,
    Segment = 1,
    Free = 2,
    Dead = -32768,
    Undead = -32767,
}

/// A vertex: coordinates, boundary marker, classification, and a back-pointer to
/// one incident triangle (Triangle's `vertex2tri`). Attributes live in a
/// side-array on the [`super::Mesh`].
#[derive(Clone, Debug)]
pub struct VertRecord {
    pub xy: [f64; 2],
    pub mark: i32,
    pub kind: VertexKind,
    /// One incident triangle, or `DUMMY` if unset.
    pub tri: TriHandle,
    /// True if this vertex slot has been deallocated (e.g. a rejected Steiner
    /// point). Distinct from `VertexKind::Undead`, which stays in the output.
    pub dead: bool,
}
