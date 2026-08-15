//! Output types returned to callers.

/// The result of a triangulation / meshing run.
#[derive(Debug, Clone, Default)]
pub struct Triangulation {
    /// Output point coordinates (x, y interleaved per point: `points[i] = [x, y]`).
    pub points: Vec<[f64; 2]>,
    /// Per-point attributes, `num_point_attributes` per point, row-major.
    pub point_attributes: Vec<f64>,
    /// Boundary markers, one per point.
    pub point_markers: Vec<i32>,
    /// Triangle corner indices (into `points`), three per element.
    pub triangles: Vec<[usize; 3]>,
    /// Nodes per element: 3 for linear, 6 for quadratic (corners + `edge_nodes`).
    pub corners_per_triangle: usize,
    /// For quadratic (`-o2`) elements: the three edge-midpoint node indices per
    /// triangle, ordered as Triangle emits them (midpoints on edges 1, 2, 0).
    /// `None` for linear elements.
    pub edge_nodes: Option<Vec<[usize; 3]>>,
    /// Per-triangle attributes (e.g. region/material id), row-major.
    pub triangle_attributes: Vec<f64>,
    /// Triangle neighbor indices (`-1` = exterior), three per triangle, when requested.
    pub neighbors: Option<Vec<[i32; 3]>>,
    /// Output segments (subsegment endpoints), two indices each.
    pub segments: Vec<[usize; 2]>,
    /// Per-segment boundary markers.
    pub segment_markers: Vec<i32>,
    /// Convex-hull edge count.
    pub hull_size: usize,
}
