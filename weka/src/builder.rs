//! The idiomatic public API: a [`Triangulator`] builder that configures and runs
//! the meshing pipeline (Delaunay → segment recovery → hole/region carving →
//! quality refinement → high-order nodes).

use crate::error::TriangleError;
use crate::input::{InputMesh, Pslg};
use crate::mesh::Mesh;
use crate::quality::Quality;
use crate::rng::Rng;
use crate::{delaunay, highorder, holes, io_assembly, quality, segments, Triangulation};

/// Builder for a triangulation / mesh-generation run. Construct with
/// [`Triangulator::new`], set options, then call one of `triangulate_points`,
/// `triangulate_pslg`, or `refine`.
#[derive(Clone, Default)]
pub struct Triangulator {
    min_angle: Option<f64>,
    max_area: Option<f64>,
    conforming: bool,
    convex: bool,
    region_attributes: bool,
    quadratic: bool,
    neighbors: bool,
    max_steiner: Option<usize>,
}

impl Triangulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enforce a minimum interior angle, in degrees (Triangle's `-q`).
    pub fn min_angle(mut self, degrees: f64) -> Self {
        self.min_angle = Some(degrees);
        self
    }
    /// Enforce a maximum triangle area (Triangle's `-a`).
    pub fn max_area(mut self, area: f64) -> Self {
        self.max_area = Some(area);
        self
    }
    /// Produce a conforming Delaunay mesh (Triangle's `-D`).
    pub fn conforming(mut self, yes: bool) -> Self {
        self.conforming = yes;
        self
    }
    /// Enclose the convex hull with segments (Triangle's `-c`).
    pub fn convex_hull(mut self, yes: bool) -> Self {
        self.convex = yes;
        self
    }
    /// Emit a per-element region/material attribute column (Triangle's `-A`).
    pub fn region_attributes(mut self, yes: bool) -> Self {
        self.region_attributes = yes;
        self
    }
    /// Emit quadratic (6-node) elements (Triangle's `-o2`).
    pub fn quadratic(mut self, yes: bool) -> Self {
        self.quadratic = yes;
        self
    }
    /// Include the triangle neighbor list in the output (Triangle's `-n`).
    pub fn neighbors(mut self, yes: bool) -> Self {
        self.neighbors = yes;
        self
    }
    /// Cap the number of Steiner points added during refinement (Triangle's `-S`).
    pub fn max_steiner(mut self, n: usize) -> Self {
        self.max_steiner = Some(n);
        self
    }

    fn quality_requested(&self) -> bool {
        self.min_angle.is_some() || self.max_area.is_some()
    }

    /// Triangulate a bare point set (optionally refined to quality).
    pub fn triangulate_points(&self, points: &[[f64; 2]]) -> Result<Triangulation, TriangleError> {
        if points.len() < 3 {
            return Err(TriangleError::TooFewPoints);
        }
        let pslg = Pslg::from_points(points.to_vec());
        self.run(&pslg, false)
    }

    /// Triangulate a PSLG (constrained Delaunay, holes, regions, quality).
    pub fn triangulate_pslg(&self, pslg: &Pslg) -> Result<Triangulation, TriangleError> {
        if pslg.points.len() < 3 {
            return Err(TriangleError::TooFewPoints);
        }
        let n = pslg.points.len();
        if pslg.segments.iter().any(|s| s[0] >= n || s[1] >= n) {
            return Err(TriangleError::InvalidIndex);
        }
        self.run(pslg, true)
    }

    /// Refine an existing mesh (Triangle's `-r`): rebuild it, then apply quality
    /// constraints (including any per-element area targets in `mesh`).
    pub fn refine(&self, mesh: &InputMesh) -> Result<Triangulation, TriangleError> {
        if mesh.points.len() < 3 || mesh.triangles.is_empty() {
            return Err(TriangleError::TooFewPoints);
        }
        let n = mesh.points.len();
        if mesh.triangles.iter().any(|t| t.iter().any(|&c| c >= n)) {
            return Err(TriangleError::InvalidIndex);
        }
        self.run_refine(mesh)
    }

    /// Shared pipeline for points (`poly=false`) and PSLGs (`poly=true`).
    fn run(&self, pslg: &Pslg, poly: bool) -> Result<Triangulation, TriangleError> {
        let region_attr = self.region_attributes;
        let eextras = usize::from(region_attr);
        let quality = self.quality_requested();
        let use_segments = poly || quality || self.convex;

        let mut m = Mesh::new(pslg.num_point_attributes, eextras, use_segments, false);
        let mut rng = Rng::new();
        io_assembly::load_points(
            &mut m,
            &pslg.points,
            &pslg.point_attributes,
            pslg.num_point_attributes,
            pslg.point_markers.as_deref(),
        );

        let hull = delaunay::delaunay(&mut m, &mut rng, true, poly);
        m.hullsize = hull as i64;

        if use_segments {
            m.checksegments = true;
            segments::form_skeleton(
                &mut m,
                &mut rng,
                &pslg.segments,
                pslg.segment_markers.as_deref(),
                poly,
                self.convex,
            );
        }

        // Hole/region carving applies to PSLGs.
        if poly && m.num_triangles() > 0 {
            holes::carve_holes(
                &mut m,
                &mut rng,
                &pslg.holes,
                &pslg.regions,
                self.convex,
                false,
                region_attr,
                false,
                0,
            );
        }

        self.finish(&mut m, &mut rng, quality, use_segments)
    }

    fn run_refine(&self, mesh: &InputMesh) -> Result<Triangulation, TriangleError> {
        let region_attr = self.region_attributes;
        // Output element-attribute columns = input columns (+ region column).
        let eextras = mesh.num_triangle_attributes + usize::from(region_attr);
        let poly = !mesh.segments.is_empty();
        let vararea = mesh.triangle_area_constraints.is_some();
        let mut m = Mesh::new(mesh.num_point_attributes, eextras, true, vararea);
        let mut rng = Rng::new();
        io_assembly::load_points(
            &mut m,
            &mesh.points,
            &mesh.point_attributes,
            mesh.num_point_attributes,
            mesh.point_markers.as_deref(),
        );
        m.checksegments = true;

        let hull = crate::reconstruct::reconstruct(
            &mut m,
            &mesh.triangles,
            &mesh.triangle_attributes,
            mesh.num_triangle_attributes,
            mesh.triangle_area_constraints.as_deref(),
            &mesh.segments,
            mesh.segment_markers.as_deref(),
            poly,
        );
        m.hullsize = hull as i64;

        if poly && m.num_triangles() > 0 {
            holes::carve_holes(
                &mut m,
                &mut rng,
                &mesh.holes,
                &mesh.regions,
                self.convex,
                false,
                region_attr,
                vararea,
                mesh.num_triangle_attributes,
            );
        }

        let quality = self.quality_requested() || vararea;
        self.finish(&mut m, &mut rng, quality, true)
    }

    fn finish(
        &self,
        m: &mut Mesh,
        rng: &mut Rng,
        quality: bool,
        use_segments: bool,
    ) -> Result<Triangulation, TriangleError> {
        if quality && m.num_triangles() > 0 {
            let mut q = Quality::new(self.min_angle.unwrap_or(0.0), self.max_area);
            q.conformdel = self.conforming;
            q.vararea = m.vararea;
            if let Some(s) = self.max_steiner {
                q.steinerleft = s as i64;
            }
            quality::enforce_quality(m, &mut q, rng);
        }
        if self.quadratic {
            highorder::high_order(m);
        }
        Ok(io_assembly::assemble(
            m,
            m.hullsize.max(0) as usize,
            self.neighbors,
            use_segments,
        ))
    }
}
