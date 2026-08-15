# weka

**A pure-Rust 2D Delaunay triangulator and quality mesh generator.**

Give `weka` a cloud of points — or a polygonal domain with holes and material
regions — and it produces a triangular mesh suitable for finite element analysis
(FEA), interpolation, computational geometry, and rendering. It is a from-scratch,
**100 % safe Rust** reimplementation of Jonathan Shewchuk's widely-used
[*Triangle*](https://www.cs.cmu.edu/~quake/triangle.html) library and produces
equivalent meshes.

## Features

- **Delaunay triangulation** of a point set.
- **Constrained Delaunay triangulation** of a PSLG (points + required edges).
- **Holes and concavities** carved out of the domain.
- **Material regions** — per-element attributes for multi-material domains.
- **Quality refinement** — guaranteed minimum angle and/or maximum element area
  (Ruppert's Delaunay refinement).
- **Quadratic (6-node) elements**, **element adjacency**, and **boundary markers**
  for FEA.
- **Refinement of an existing mesh** — the basis for adaptive analysis.
- **Robust**: exact-arithmetic predicates handle collinear/cocircular/degenerate
  input correctly; results are deterministic.

## Install

```toml
[dependencies]
weka = "0.1"
```

## Quick start

Everything is driven through the [`Triangulator`] builder: create one, chain the
options you want, then call a `triangulate_*` method. Points are `[f64; 2]`,
indices are `usize`.

### Triangulate a point set

```rust
use weka::Triangulator;

let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]];
let mesh = Triangulator::new().triangulate_points(&points)?;

for &[a, b, c] in &mesh.triangles {
    // mesh.points[a], [b], [c] are the corner coordinates (always CCW).
    println!("triangle {a} {b} {c}");
}
# Ok::<(), weka::TriangleError>(())
```

### Mesh a polygon, with quality bounds

A `Pslg` is points plus the *segments* (edges, as index pairs) that must appear
in the mesh. Ask for a minimum angle and maximum area and the mesher inserts
points until every element complies:

```rust
use weka::{Pslg, Triangulator};

let pslg = Pslg {
    points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    segments: vec![[0, 1], [1, 2], [2, 3], [3, 0]], // the four boundary edges
    ..Default::default()
};

let mesh = Triangulator::new()
    .min_angle(28.0)   // no triangle narrower than 28°
    .max_area(0.02)    // no triangle larger than 0.02
    .neighbors(true)   // also return element adjacency
    .triangulate_pslg(&pslg)?;
# Ok::<(), weka::TriangleError>(())
```

## Concepts

| Concept | What it is |
|---|---|
| **Point** | `[f64; 2]`. Triangles refer to their corners by index into `Triangulation::points`. |
| **Segment** | An edge (`[usize; 2]`) guaranteed to appear in the mesh — a domain boundary or internal interface. Refinement may subdivide it into a chain of shorter edges. |
| **Hole** | A seed point placed inside a segment-bounded region to delete; everything reachable from it (without crossing a segment) is removed. |
| **Region** | A seed point (`RegionSpec`) tagging a segment-bounded area with a material attribute and an optional local area limit. |
| **Quality** | `min_angle` (degrees) and `max_area` bounds enforced by adding Steiner points. |
| **Boundary markers** | Integer tags on points/segments, carried through to the output so FEA codes can locate boundaries. |

## Builder options

| Method | Effect |
|---|---|
| `min_angle(deg)` | Enforce a minimum interior angle. |
| `max_area(a)` | Enforce a maximum element area. |
| `convex_hull(bool)` | Enclose the convex hull with segments. |
| `conforming(bool)` | Produce a conforming (rather than merely constrained) Delaunay mesh. |
| `region_attributes(bool)` | Emit a per-element region/material id column. |
| `quadratic(bool)` | Emit 6-node elements (edge midpoints in `edge_nodes`). |
| `neighbors(bool)` | Include the element adjacency list. |
| `max_steiner(n)` | Cap the number of inserted points. |

Entry points: `triangulate_points(&[[f64; 2]])`, `triangulate_pslg(&Pslg)`, and
`refine(&InputMesh)`.

## The output

A `Triangulation` always contains `points` and `triangles` (triples of point
indices, counter-clockwise). Requested extras: `neighbors` (`-1` = boundary),
`edge_nodes` (quadratic midpoint nodes), `triangle_attributes` (region ids),
`segments` + `segment_markers`, and `point_markers`.

## Example

A full FEA example — a rectangular plate with a square hole split into two
material regions, refined with quadratic elements and adjacency — is in
[`weka/examples/fea_mesh.rs`](weka/examples/fea_mesh.rs):

```sh
cargo run -p weka --release --example fea_mesh
```

## Performance

`weka` is written in 100 % safe Rust and is competitive with, and on the
refinement path faster than, the original C library. Measured on an Apple
M-series CPU (`cargo bench`); ratio is weka / C, so **< 1 is faster than C**.

Plain Delaunay of uniform-random points (near parity, converging with size):

| points | weka | C | ratio |
|---|------:|---:|------:|
| 10 000 | ~4.4 ms | ~4.0 ms | ~1.10× |
| 100 000 | ~48 ms | ~46 ms | ~1.05× |
| 1 000 000 | ~564 ms | ~552 ms | ~1.02× |

Quality meshing (`min_angle(20)` + a max area — the representative FEA workload)
is **strictly faster than C, with the lead growing as the mesh gets finer**:

| max area | weka | C | ratio |
|---|------:|---:|------:|
| 1e-3 | ~0.27 ms | ~0.29 ms | **0.94×** |
| 1e-4 | ~3.0 ms | ~3.4 ms | **0.90×** |
| 1e-5 | ~32 ms | ~39 ms | **0.83×** |

Constrained Delaunay is at parity (~0.97–1.15×).

## Correctness

`weka` is validated by differential tests that run the same inputs through the
original C library and compare results: exact match of the triangle set for
point sets up to a million points, exact segment/hole/region recovery, and — for
quality meshes — the min-angle and max-area guarantees plus the Delaunay property
on every interior edge.

## Building and testing

```sh
cargo test                       # unit + differential tests (needs a C++ compiler)
cargo bench                      # benchmarks against the C library
cargo clippy --workspace --tests
```

## License

MIT. The bundled C source under `triangle/`, used only as a testing/benchmark
reference, remains under Shewchuk's original license.
