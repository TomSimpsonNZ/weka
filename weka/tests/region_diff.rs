//! P5 differential test: multi-region material attributes (Triangle's `-A`).
//! A 2x1 rectangle split by an internal segment into two regions with distinct
//! attributes; weka vs C (`pzAQ`). Verifies each element gets the right region id.

use std::collections::BTreeMap;
use weka::RegionSpec;

fn tri_key(p: [[f64; 2]; 3]) -> [[u64; 2]; 3] {
    let mut k = [
        [p[0][0].to_bits(), p[0][1].to_bits()],
        [p[1][0].to_bits(), p[1][1].to_bits()],
        [p[2][0].to_bits(), p[2][1].to_bits()],
    ];
    k.sort_unstable();
    k
}

#[test]
fn regions_match_c() {
    // Vertices: rectangle corners + edge midpoints splitting it in half.
    let points = vec![
        [0.0, 0.0], // 0
        [2.0, 0.0], // 1
        [2.0, 1.0], // 2
        [0.0, 1.0], // 3
        [1.0, 0.0], // 4 (bottom mid)
        [1.0, 1.0], // 5 (top mid)
    ];
    let segments = vec![
        [0, 4],
        [4, 1],
        [1, 2],
        [2, 5],
        [5, 3],
        [3, 0],
        [4, 5], // internal divider
    ];
    let regions = vec![
        RegionSpec {
            point: [0.5, 0.5],
            attribute: 7.0,
            area: f64::NAN,
        },
        RegionSpec {
            point: [1.5, 0.5],
            attribute: 9.0,
            area: f64::NAN,
        },
    ];

    let t = weka::mesh_pslg(
        &points,
        &[],
        0,
        None,
        &segments,
        None,
        &[],
        &regions,
        false,
        true,
    );
    assert_eq!(
        t.triangle_attributes.len(),
        t.triangles.len(),
        "1 attr per triangle"
    );

    // C library, pzAQ.
    let mut pointlist = Vec::new();
    for p in &points {
        pointlist.push(p[0]);
        pointlist.push(p[1]);
    }
    let mut segmentlist = Vec::new();
    for s in &segments {
        segmentlist.push(s[0] as i32);
        segmentlist.push(s[1] as i32);
    }
    let regionlist = vec![0.5, 0.5, 7.0, 0.0, 1.5, 0.5, 9.0, 0.0];
    let input = ctriangle_sys::Input {
        pointlist,
        segmentlist,
        regionlist,
        ..Default::default()
    };
    let c = ctriangle_sys::triangulate_safe("pzAQ", &input);

    assert_eq!(t.triangles.len(), c.numberoftriangles, "triangle count");
    assert_eq!(
        c.numberoftriangleattributes, 1,
        "C should emit 1 region attribute"
    );

    // Map each triangle (by coords) to its region attribute, for both.
    let weka_map: BTreeMap<_, _> = t
        .triangles
        .iter()
        .enumerate()
        .map(|(i, &[a, b, cc])| {
            (
                tri_key([t.points[a], t.points[b], t.points[cc]]),
                t.triangle_attributes[i],
            )
        })
        .collect();

    let pt = |i: i32| {
        let i = i as usize;
        [c.pointlist[2 * i], c.pointlist[2 * i + 1]]
    };
    let c_map: BTreeMap<_, _> = c
        .trianglelist
        .chunks_exact(3)
        .enumerate()
        .map(|(i, w)| {
            (
                tri_key([pt(w[0]), pt(w[1]), pt(w[2])]),
                c.triangleattributelist[i],
            )
        })
        .collect();

    assert_eq!(
        weka_map, c_map,
        "per-triangle region attributes must match C"
    );

    // Sanity: both region ids actually appear.
    let vals: std::collections::BTreeSet<u64> = weka_map.values().map(|v| v.to_bits()).collect();
    assert!(vals.contains(&7.0f64.to_bits()) && vals.contains(&9.0f64.to_bits()));
}
