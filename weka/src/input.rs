//! Input types for the [`crate::Triangulator`] builder.

use crate::holes::RegionSpec;

/// A Planar Straight Line Graph: points plus constraining segments, holes, and
/// material regions.
#[derive(Clone, Debug, Default)]
pub struct Pslg {
    /// Point coordinates.
    pub points: Vec<[f64; 2]>,
    /// Per-point attributes (`num_point_attributes` per point, row-major).
    pub point_attributes: Vec<f64>,
    /// Number of attributes per point.
    pub num_point_attributes: usize,
    /// Optional per-point boundary markers.
    pub point_markers: Option<Vec<i32>>,
    /// Constraining segments (pairs of point indices).
    pub segments: Vec<[usize; 2]>,
    /// Optional per-segment boundary markers.
    pub segment_markers: Option<Vec<i32>>,
    /// Hole seed points (a point inside each region to remove).
    pub holes: Vec<[f64; 2]>,
    /// Region seed points carrying a material attribute / area constraint.
    pub regions: Vec<RegionSpec>,
}

impl Pslg {
    /// A PSLG with just points and no attributes.
    pub fn from_points(points: Vec<[f64; 2]>) -> Self {
        Pslg {
            points,
            ..Default::default()
        }
    }
}

/// An existing mesh to hand to [`Triangulator::refine`](crate::Triangulator::refine).
///
/// Only [`points`](InputMesh::points) and [`triangles`](InputMesh::triangles) are
/// required; everything else is optional and defaults to empty (use `..Default::default()`).
/// Include [`segments`](InputMesh::segments) to preserve boundaries/interfaces, and
/// [`triangle_area_constraints`](InputMesh::triangle_area_constraints) for
/// per-element target sizes (e.g. error-driven adaptive refinement).
#[derive(Clone, Debug, Default)]
pub struct InputMesh {
    /// Vertex coordinates.
    pub points: Vec<[f64; 2]>,
    /// Per-point attributes (`num_point_attributes` per point, row-major).
    pub point_attributes: Vec<f64>,
    /// Number of attributes per point.
    pub num_point_attributes: usize,
    /// Optional per-point boundary markers.
    pub point_markers: Option<Vec<i32>>,
    /// Element corner indices (three per triangle).
    pub triangles: Vec<[usize; 3]>,
    /// Per-element attributes (`num_triangle_attributes` per element).
    pub triangle_attributes: Vec<f64>,
    /// Number of attributes per element.
    pub num_triangle_attributes: usize,
    /// Optional per-element maximum-area constraint (one entry per triangle).
    pub triangle_area_constraints: Option<Vec<f64>>,
    /// Optional constraining segments to preserve during refinement.
    pub segments: Vec<[usize; 2]>,
    /// Optional per-segment boundary markers.
    pub segment_markers: Option<Vec<i32>>,
    /// Hole seed points.
    pub holes: Vec<[f64; 2]>,
    /// Region seed points (material attributes / area limits).
    pub regions: Vec<RegionSpec>,
}
