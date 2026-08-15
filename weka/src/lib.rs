//! `weka` is a pure-Rust 2D **Delaunay triangulator** and **quality mesh
//! generator**. Give it a cloud of points — or a polygonal domain with holes and
//! material regions — and it produces a triangular mesh suitable for finite
//! element analysis (FEA), interpolation, computational geometry, and rendering.
//!
//! It is a from-scratch, **100 % safe Rust** reimplementation of Jonathan
//! Shewchuk's widely-used *Triangle* library and produces equivalent meshes.
//!
//! # What it does
//!
//! * **Delaunay triangulation** of a point set.
//! * **Constrained Delaunay triangulation** of a *planar straight-line graph*
//!   (PSLG) — a set of points plus edges ("segments") that must appear in the
//!   output, such as domain boundaries or material interfaces.
//! * **Holes and concavities** — carve regions out of the domain.
//! * **Material regions** — tag elements with a per-region attribute.
//! * **Quality refinement** — add points so that no triangle has an angle
//!   smaller than a chosen bound and/or an area larger than a chosen bound
//!   (Ruppert's Delaunay refinement).
//! * **Quadratic (6-node) elements**, **element adjacency**, and **boundary
//!   markers** for finite-element workflows.
//! * **Refinement of an existing mesh** — rebuild a mesh you already have and
//!   refine it further (useful for adaptive analysis).
//!
//! # Getting started
//!
//! Add the crate to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! weka = "0.1"
//! ```
//!
//! Everything is driven through the [`Triangulator`] builder: create one,
//! chain the options you want, then call one of its `triangulate_*` methods.
//! Coordinates are plain `[f64; 2]` arrays and indices are `usize`.
//!
//! ## Triangulate a point set
//!
//! ```
//! use weka::Triangulator;
//!
//! let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]];
//! let mesh = Triangulator::new().triangulate_points(&points)?;
//!
//! for &[a, b, c] in &mesh.triangles {
//!     // `mesh.points[a]`, `[b]`, `[c]` are the corner coordinates.
//!     println!("triangle with corners {a}, {b}, {c}");
//! }
//! # Ok::<(), weka::TriangleError>(())
//! ```
//!
//! ## Mesh a polygon (a PSLG)
//!
//! A [`Pslg`] is a list of points plus the [`segments`](Pslg::segments) (index
//! pairs) that must appear as edges. Here we mesh the interior of a unit square:
//!
//! ```
//! use weka::{Pslg, Triangulator};
//!
//! let pslg = Pslg {
//!     points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
//!     segments: vec![[0, 1], [1, 2], [2, 3], [3, 0]], // the four boundary edges
//!     ..Default::default()
//! };
//! let mesh = Triangulator::new().triangulate_pslg(&pslg)?;
//! assert_eq!(mesh.triangles.len(), 2); // a square splits into two triangles
//! # Ok::<(), weka::TriangleError>(())
//! ```
//!
//! ## Generate a quality FEA mesh
//!
//! Ask for a minimum angle and a maximum element area, and the mesher inserts
//! extra ("Steiner") points until every element satisfies both bounds:
//!
//! ```
//! use weka::{Pslg, Triangulator};
//!
//! let pslg = Pslg {
//!     points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
//!     segments: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
//!     ..Default::default()
//! };
//! let mesh = Triangulator::new()
//!     .min_angle(28.0)   // no triangle narrower than 28°
//!     .max_area(0.02)    // no triangle bigger than 0.02
//!     .neighbors(true)   // also return element adjacency
//!     .triangulate_pslg(&pslg)?;
//!
//! assert!(mesh.triangles.len() > 50);
//! assert!(mesh.neighbors.is_some());
//! # Ok::<(), weka::TriangleError>(())
//! ```
//!
//! # Core concepts
//!
//! **Points** are `[f64; 2]`. In the output, a triangle refers to its corners by
//! their index into [`Triangulation::points`].
//!
//! **Segments** ([`Pslg::segments`]) are edges — given as pairs of point indices —
//! that are *guaranteed* to appear in the mesh (as a single edge, or as a chain
//! of shorter edges once refinement subdivides them). Use them for domain
//! boundaries and internal interfaces.
//!
//! **Holes** ([`Pslg::holes`]) are seed points: place one anywhere inside a
//! segment-bounded region you want removed, and every triangle reachable from it
//! (without crossing a segment) is deleted. Concavities are carved automatically.
//!
//! **Regions** ([`RegionSpec`]) tag a segment-bounded area with a material id.
//! Enable [`region_attributes`](Triangulator::region_attributes) and each output
//! element carries its region's [`attribute`](RegionSpec::attribute) in
//! [`Triangulation::triangle_attributes`]. A region may also impose its own local
//! [`area`](RegionSpec::area) limit.
//!
//! **Quality** is controlled by [`min_angle`](Triangulator::min_angle) (degrees)
//! and [`max_area`](Triangulator::max_area). Refinement is guaranteed to
//! terminate for minimum angles up to about 20–33° depending on the input;
//! very large minimum angles may not converge for domains with sharp input
//! angles. [`max_steiner`](Triangulator::max_steiner) caps how many points are
//! added.
//!
//! **Boundary markers** are integer tags. Supply them on input points/segments
//! ([`Pslg::point_markers`], [`Pslg::segment_markers`]); the mesh reports the
//! marker of every output point and segment
//! ([`Triangulation::point_markers`], [`Triangulation::segment_markers`]),
//! which is how FEA codes locate boundaries to apply conditions.
//!
//! # The output
//!
//! Every entry point returns a [`Triangulation`]. The always-present fields are
//! [`points`](Triangulation::points) and [`triangles`](Triangulation::triangles)
//! (triples of point indices, always counter-clockwise). Optional data is
//! produced on request:
//!
//! * [`neighbors`](Triangulation::neighbors) — set [`neighbors(true)`](Triangulator::neighbors);
//!   `-1` marks an edge on the domain boundary.
//! * [`edge_nodes`](Triangulation::edge_nodes) — set [`quadratic(true)`](Triangulator::quadratic)
//!   for 6-node elements; the three edge-midpoint node indices per triangle.
//! * [`triangle_attributes`](Triangulation::triangle_attributes) — region ids,
//!   with [`region_attributes(true)`](Triangulator::region_attributes).
//! * [`segments`](Triangulation::segments) and their markers — the recovered
//!   PSLG/boundary edges.
//!
//! # Refining an existing mesh
//!
//! [`Triangulator::refine`] takes an [`InputMesh`] (points + triangles you
//! already have) and re-meshes it under new quality constraints — the basis for
//! error-driven adaptive refinement:
//!
//! ```
//! use weka::{InputMesh, Pslg, Triangulator};
//!
//! // A coarse mesh of a square.
//! let square = Pslg {
//!     points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
//!     segments: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
//!     ..Default::default()
//! };
//! let coarse = Triangulator::new().triangulate_pslg(&square)?;
//!
//! let input = InputMesh {
//!     points: coarse.points.clone(),
//!     triangles: coarse.triangles.clone(),
//!     segments: square.segments.clone(),
//!     ..Default::default()
//! };
//! let fine = Triangulator::new().max_area(0.02).refine(&input)?;
//! assert!(fine.triangles.len() > coarse.triangles.len());
//! # Ok::<(), weka::TriangleError>(())
//! ```
//!
//! # Robustness and determinism
//!
//! weka uses exact-arithmetic geometric predicates, so it triangulates
//! collinear, cocircular, and near-degenerate inputs correctly rather than
//! producing inverted or missing triangles. Results are deterministic: the same
//! input always yields the same mesh.
//!
//! The robust predicates are available directly in the [`predicates`] module if
//! you need them (e.g. [`predicates::orient2d`], [`predicates::incircle`]).

pub mod builder;
#[doc(hidden)]
pub mod delaunay;
pub mod error;
#[doc(hidden)]
pub mod highorder;
#[doc(hidden)]
pub mod holes;
pub mod input;
#[doc(hidden)]
pub mod io_assembly;
#[doc(hidden)]
pub mod mesh;
pub mod output;
pub mod predicates;
#[doc(hidden)]
pub mod quality;
#[doc(hidden)]
pub mod reconstruct;
#[doc(hidden)]
pub mod rng;
#[doc(hidden)]
pub mod segments;

pub use builder::Triangulator;
pub use error::TriangleError;
pub use holes::RegionSpec;
pub use input::{InputMesh, Pslg};
pub use output::Triangulation;
#[doc(hidden)]
pub use quality::Quality;

use mesh::Mesh;
use rng::Rng;

/// Delaunay triangulation of a point set. Low-level helper; prefer
/// [`Triangulator::triangulate_points`]. `neighbors` requests the adjacency list.
#[doc(hidden)]
pub fn delaunay_points(points: &[[f64; 2]], neighbors: bool) -> Triangulation {
    let mut m = Mesh::new(0, 0, false, false);
    let mut rng = Rng::new();
    io_assembly::load_points(&mut m, points, &[], 0, None);
    let hull = delaunay::delaunay(&mut m, &mut rng, true, false);
    io_assembly::assemble(&m, hull, neighbors, false)
}

/// Constrained Delaunay triangulation of a PSLG. Low-level helper; prefer
/// [`Triangulator::triangulate_pslg`]. Recovers every input segment as mesh
/// edges; `convex` additionally encloses the convex hull with segments. Optional
/// markers default to zero when `None`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn cdt_pslg(
    points: &[[f64; 2]],
    point_markers: Option<&[i32]>,
    seg: &[[usize; 2]],
    segment_markers: Option<&[i32]>,
    convex: bool,
) -> Triangulation {
    mesh_pslg(
        points,
        &[],
        0,
        point_markers,
        seg,
        segment_markers,
        &[],
        &[],
        convex,
        false,
    )
}

/// Mesh a PSLG with quality refinement. Low-level helper; prefer
/// [`Triangulator`] with [`min_angle`](Triangulator::min_angle) /
/// [`max_area`](Triangulator::max_area). Runs constrained Delaunay, carves
/// holes/regions, then Ruppert refinement to a minimum angle (degrees) and an
/// optional maximum triangle area.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn mesh_pslg_quality(
    points: &[[f64; 2]],
    point_attrs: &[f64],
    num_point_attrs: usize,
    point_markers: Option<&[i32]>,
    seg: &[[usize; 2]],
    segment_markers: Option<&[i32]>,
    holes_list: &[[f64; 2]],
    regions: &[RegionSpec],
    convex: bool,
    region_attributes: bool,
    min_angle: f64,
    max_area: Option<f64>,
) -> Triangulation {
    let eextras = usize::from(region_attributes);
    let mut m = Mesh::new(num_point_attrs, eextras, true, false);
    let mut rng = Rng::new();
    io_assembly::load_points(&mut m, points, point_attrs, num_point_attrs, point_markers);
    let hull = delaunay::delaunay(&mut m, &mut rng, true, true);
    m.hullsize = hull as i64;
    m.checksegments = true;
    segments::form_skeleton(&mut m, &mut rng, seg, segment_markers, true, convex);
    if m.num_triangles() > 0 {
        holes::carve_holes(
            &mut m,
            &mut rng,
            holes_list,
            regions,
            convex,
            false,
            region_attributes,
            false,
            0,
        );
    }
    if m.num_triangles() > 0 {
        let mut q = Quality::new(min_angle, max_area);
        quality::enforce_quality(&mut m, &mut q, &mut rng);
    }
    io_assembly::assemble(&m, m.hullsize.max(0) as usize, true, true)
}

/// Mesh a full PSLG (constrained Delaunay + hole/region carving), without quality
/// refinement. Low-level helper; prefer [`Triangulator::triangulate_pslg`].
/// `region_attributes` appends a per-element region/material id column
/// (0 outside any seeded region).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn mesh_pslg(
    points: &[[f64; 2]],
    point_attrs: &[f64],
    num_point_attrs: usize,
    point_markers: Option<&[i32]>,
    seg: &[[usize; 2]],
    segment_markers: Option<&[i32]>,
    holes_list: &[[f64; 2]],
    regions: &[RegionSpec],
    convex: bool,
    region_attributes: bool,
) -> Triangulation {
    let eextras = usize::from(region_attributes);
    let mut m = Mesh::new(num_point_attrs, eextras, true, false);
    let mut rng = Rng::new();
    io_assembly::load_points(&mut m, points, point_attrs, num_point_attrs, point_markers);
    // poly = true: hull markers are applied by markhull/carving, not removeghosts.
    let hull = delaunay::delaunay(&mut m, &mut rng, true, true);
    m.hullsize = hull as i64;
    m.checksegments = true;
    segments::form_skeleton(&mut m, &mut rng, seg, segment_markers, true, convex);
    if m.num_triangles() > 0 {
        holes::carve_holes(
            &mut m,
            &mut rng,
            holes_list,
            regions,
            convex,
            false,
            region_attributes,
            false,
            0,
        );
    }
    io_assembly::assemble(&m, m.hullsize.max(0) as usize, true, true)
}
