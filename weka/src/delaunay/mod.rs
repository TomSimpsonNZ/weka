//! Delaunay triangulation algorithms. For the FEA-scoped port, only the default
//! divide-and-conquer method is provided (Triangle's `delaunay()` dispatcher
//! otherwise also offers incremental/sweepline, which are out of scope).

pub mod divconq;

use crate::mesh::Mesh;
use crate::rng::Rng;

/// Triangulate all input vertices currently loaded in `m`. Returns the hull size.
pub fn delaunay(m: &mut Mesh, rng: &mut Rng, dwyer: bool, poly: bool) -> usize {
    divconq::divconq_delaunay(m, rng, dwyer, poly)
}
