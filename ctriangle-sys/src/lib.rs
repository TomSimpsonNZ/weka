//! FFI bindings to Shewchuk's Triangle C library, used purely as a differential
//! testing / benchmarking oracle for the `weka` crate.
//!
//! Triangle relies on file-scope mutable statics (`randomseed`, the predicate
//! bounds computed by `exactinit`, etc.), so [`triangulate`] serializes all calls
//! through a global mutex.

use std::ffi::{c_char, c_int, c_void, CString};
use std::os::raw::c_double;
use std::ptr;
use std::sync::Mutex;

/// Mirror of `struct triangulateio` from `triangle.h` (double precision, no `SINGLE`).
#[repr(C)]
#[derive(Debug)]
pub struct TriangulateIo {
    pub pointlist: *mut c_double,
    pub pointattributelist: *mut c_double,
    pub pointmarkerlist: *mut c_int,
    pub numberofpoints: c_int,
    pub numberofpointattributes: c_int,

    pub trianglelist: *mut c_int,
    pub triangleattributelist: *mut c_double,
    pub trianglearealist: *mut c_double,
    pub neighborlist: *mut c_int,
    pub numberoftriangles: c_int,
    pub numberofcorners: c_int,
    pub numberoftriangleattributes: c_int,

    pub segmentlist: *mut c_int,
    pub segmentmarkerlist: *mut c_int,
    pub numberofsegments: c_int,

    pub holelist: *mut c_double,
    pub numberofholes: c_int,

    pub regionlist: *mut c_double,
    pub numberofregions: c_int,

    pub edgelist: *mut c_int,
    pub edgemarkerlist: *mut c_int,
    pub normlist: *mut c_double,
    pub numberofedges: c_int,
}

impl TriangulateIo {
    /// A fully-zeroed `triangulateio` (all pointers null, all counts zero).
    pub fn zeroed() -> Self {
        // SAFETY: an all-zero triangulateio is the documented "uninitialized"
        // state — null pointers tell Triangle to allocate output itself.
        unsafe { std::mem::zeroed() }
    }
}

extern "C" {
    // C-linkage shims defined in shim.cpp (Triangle itself is C++-mangled).
    fn weka_triangulate(
        triswitches: *const c_char,
        in_: *mut TriangulateIo,
        out: *mut TriangulateIo,
        vorout: *mut TriangulateIo,
    );
    fn weka_trifree(memptr: *mut c_void);
}

#[inline]
unsafe fn triangulate(
    triswitches: *const c_char,
    in_: *mut TriangulateIo,
    out: *mut TriangulateIo,
    vorout: *mut TriangulateIo,
) {
    weka_triangulate(triswitches, in_, out, vorout)
}

#[inline]
unsafe fn trifree(memptr: *mut c_void) {
    weka_trifree(memptr)
}

static C_LOCK: Mutex<()> = Mutex::new(());

/// Logical input to the C triangulator (owns its arrays; C never frees inputs).
#[derive(Default, Clone)]
pub struct Input {
    pub pointlist: Vec<f64>,
    pub pointattributelist: Vec<f64>,
    pub pointmarkerlist: Vec<i32>,
    pub numberofpointattributes: i32,

    pub trianglelist: Vec<i32>,
    pub triangleattributelist: Vec<f64>,
    pub trianglearealist: Vec<f64>,
    pub numberofcorners: i32,
    pub numberoftriangleattributes: i32,

    pub segmentlist: Vec<i32>,
    pub segmentmarkerlist: Vec<i32>,

    pub holelist: Vec<f64>,
    pub regionlist: Vec<f64>,
}

/// Owned copy of the C output (all C-allocated arrays are copied then freed).
#[derive(Default, Debug, Clone)]
pub struct Output {
    pub pointlist: Vec<f64>,
    pub pointattributelist: Vec<f64>,
    pub pointmarkerlist: Vec<i32>,
    pub numberofpoints: usize,
    pub numberofpointattributes: usize,

    pub trianglelist: Vec<i32>,
    pub triangleattributelist: Vec<f64>,
    pub neighborlist: Vec<i32>,
    pub numberoftriangles: usize,
    pub numberofcorners: usize,
    pub numberoftriangleattributes: usize,

    pub segmentlist: Vec<i32>,
    pub segmentmarkerlist: Vec<i32>,
    pub numberofsegments: usize,

    pub edgelist: Vec<i32>,
    pub edgemarkerlist: Vec<i32>,
    pub numberofedges: usize,
}

unsafe fn copy_i32(ptr: *const c_int, len: usize) -> Vec<i32> {
    if ptr.is_null() || len == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(ptr, len).to_vec()
    }
}

unsafe fn copy_f64(ptr: *const c_double, len: usize) -> Vec<f64> {
    if ptr.is_null() || len == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(ptr, len).to_vec()
    }
}

unsafe fn free_if(ptr: *mut c_void) {
    if !ptr.is_null() {
        trifree(ptr);
    }
}

/// Run the C `triangulate()` with the given switch string (no leading dash).
///
/// Output arrays are read according to the counts Triangle reports, copied into
/// owned `Vec`s, and the C-allocated memory is freed before returning.
pub fn triangulate_safe(switches: &str, input: &Input) -> Output {
    let cswitches = CString::new(switches).expect("switch string contains NUL");
    let numberofpoints = (input.pointlist.len() / 2) as c_int;

    // Build the C input struct. Empty Vecs map to null pointers.
    let mut input = input.clone(); // local mutable copy so we can take *mut pointers
    let mut cin = TriangulateIo::zeroed();
    cin.numberofpoints = numberofpoints;
    cin.numberofpointattributes = input.numberofpointattributes;
    cin.pointlist = vec_ptr_f64(&mut input.pointlist);
    cin.pointattributelist = vec_ptr_f64(&mut input.pointattributelist);
    cin.pointmarkerlist = vec_ptr_i32(&mut input.pointmarkerlist);

    cin.numberofcorners = input.numberofcorners;
    cin.numberoftriangleattributes = input.numberoftriangleattributes;
    if !input.trianglelist.is_empty() {
        let corners = if input.numberofcorners > 0 {
            input.numberofcorners as usize
        } else {
            3
        };
        cin.numberoftriangles = (input.trianglelist.len() / corners) as c_int;
    }
    cin.trianglelist = vec_ptr_i32(&mut input.trianglelist);
    cin.triangleattributelist = vec_ptr_f64(&mut input.triangleattributelist);
    cin.trianglearealist = vec_ptr_f64(&mut input.trianglearealist);

    cin.numberofsegments = (input.segmentlist.len() / 2) as c_int;
    cin.segmentlist = vec_ptr_i32(&mut input.segmentlist);
    cin.segmentmarkerlist = vec_ptr_i32(&mut input.segmentmarkerlist);

    cin.numberofholes = (input.holelist.len() / 2) as c_int;
    cin.holelist = vec_ptr_f64(&mut input.holelist);
    cin.numberofregions = (input.regionlist.len() / 4) as c_int;
    cin.regionlist = vec_ptr_f64(&mut input.regionlist);

    let mut cout = TriangulateIo::zeroed();

    let _guard = C_LOCK.lock().unwrap();
    // SAFETY: cin is fully initialized; cout is zeroed so Triangle allocates all
    // requested outputs itself. We hold the global lock for the duration.
    unsafe {
        triangulate(
            cswitches.as_ptr(),
            &mut cin as *mut _,
            &mut cout as *mut _,
            ptr::null_mut(),
        );
    }

    // SAFETY: read counts/arrays Triangle set, copy them out, then free.
    unsafe {
        let np = cout.numberofpoints.max(0) as usize;
        let npa = cout.numberofpointattributes.max(0) as usize;
        let nt = cout.numberoftriangles.max(0) as usize;
        let nc = cout.numberofcorners.max(0) as usize;
        let nta = cout.numberoftriangleattributes.max(0) as usize;
        let nseg = cout.numberofsegments.max(0) as usize;
        let nedge = cout.numberofedges.max(0) as usize;

        let out = Output {
            pointlist: copy_f64(cout.pointlist, np * 2),
            pointattributelist: copy_f64(cout.pointattributelist, np * npa),
            pointmarkerlist: copy_i32(cout.pointmarkerlist, np),
            numberofpoints: np,
            numberofpointattributes: npa,

            trianglelist: copy_i32(cout.trianglelist, nt * nc),
            triangleattributelist: copy_f64(cout.triangleattributelist, nt * nta),
            neighborlist: copy_i32(cout.neighborlist, nt * 3),
            numberoftriangles: nt,
            numberofcorners: nc,
            numberoftriangleattributes: nta,

            segmentlist: copy_i32(cout.segmentlist, nseg * 2),
            segmentmarkerlist: copy_i32(cout.segmentmarkerlist, nseg),
            numberofsegments: nseg,

            edgelist: copy_i32(cout.edgelist, nedge * 2),
            edgemarkerlist: copy_i32(cout.edgemarkerlist, nedge),
            numberofedges: nedge,
        };

        // Free everything Triangle allocated. holelist/regionlist in `cout` alias
        // the input pointers (Triangle copies the pointer, not the data) so we must
        // NOT free those.
        free_if(cout.pointlist as *mut c_void);
        free_if(cout.pointattributelist as *mut c_void);
        free_if(cout.pointmarkerlist as *mut c_void);
        free_if(cout.trianglelist as *mut c_void);
        free_if(cout.triangleattributelist as *mut c_void);
        free_if(cout.neighborlist as *mut c_void);
        free_if(cout.segmentlist as *mut c_void);
        free_if(cout.segmentmarkerlist as *mut c_void);
        free_if(cout.edgelist as *mut c_void);
        free_if(cout.edgemarkerlist as *mut c_void);

        out
    }
}

fn vec_ptr_f64(v: &mut Vec<f64>) -> *mut c_double {
    if v.is_empty() {
        ptr::null_mut()
    } else {
        v.as_mut_ptr()
    }
}

fn vec_ptr_i32(v: &mut Vec<i32>) -> *mut c_int {
    if v.is_empty() {
        ptr::null_mut()
    } else {
        v.as_mut_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_square() {
        // Four corners of a unit square; zero-based numbering, quiet.
        let input = Input {
            pointlist: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            ..Default::default()
        };
        let out = triangulate_safe("zQ", &input);
        assert_eq!(out.numberofpoints, 4);
        // A convex quad triangulates into exactly 2 triangles.
        assert_eq!(out.numberoftriangles, 2);
        assert_eq!(out.numberofcorners, 3);
        assert_eq!(out.trianglelist.len(), 6);
        // Every emitted index is a valid vertex.
        assert!(out.trianglelist.iter().all(|&i| (0..4).contains(&i)));
    }
}
