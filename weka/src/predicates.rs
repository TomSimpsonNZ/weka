//! Robust geometric predicates, ported from Shewchuk's adaptive-precision
//! routines in triangle.cpp (`exactinit`, `counterclockwise`, `incircle`).
//!
//! `orient2d` (`counterclockwise`) is a full port and returns values bit-identical
//! to the C library. `incircle` ports the fast path and the first two refinement
//! stages verbatim (covering the common case exactly); the rare deepest
//! exact-arithmetic stage is delegated to the `robust` crate for a guaranteed
//! correct *sign* (the only thing the mesh code relies on in that regime).
//!
//! Rust performs no automatic FMA contraction or floating-point reassociation, so
//! the two-sum / two-product building blocks reproduce the C arithmetic faithfully.

/// A 2D point. Matches the `vertex` coordinate pair used throughout the mesh.
pub type Point = [f64; 2];

/// Exact-arithmetic constants (machine epsilon, splitter, and the orientation /
/// incircle error bounds) used by the adaptive-precision predicates.
#[derive(Debug, Clone, Copy)]
pub struct Consts {
    pub epsilon: f64,
    pub splitter: f64,
    pub resulterrbound: f64,
    pub ccwerrbound_a: f64,
    pub ccwerrbound_b: f64,
    pub ccwerrbound_c: f64,
    pub iccerrbound_a: f64,
    pub iccerrbound_b: f64,
    pub iccerrbound_c: f64,
}

/// The exact-arithmetic constants for IEEE-754 double precision.
///
/// Triangle computes these at runtime in `exactinit()` (triangle.cpp:4894), but
/// for `f64` they are fixed: machine epsilon is `2^-53` and the splitter is
/// `2^27 + 1`. Computing them as a `const` (Rust permits float arithmetic in
/// const context) removes a per-predicate-call atomic load from the hot path.
/// `exactinit_runtime` reproduces the loop and is checked against `CONSTS` in
/// the tests, guaranteeing the hardcoded epsilon/splitter are correct.
const CONSTS: Consts = {
    let epsilon = 1.1102230246251565e-16_f64; // 2^-53
    let splitter = 134217729.0_f64; // 2^27 + 1
    Consts {
        epsilon,
        splitter,
        resulterrbound: (3.0 + 8.0 * epsilon) * epsilon,
        ccwerrbound_a: (3.0 + 16.0 * epsilon) * epsilon,
        ccwerrbound_b: (2.0 + 12.0 * epsilon) * epsilon,
        ccwerrbound_c: (9.0 + 64.0 * epsilon) * epsilon * epsilon,
        iccerrbound_a: (10.0 + 96.0 * epsilon) * epsilon,
        iccerrbound_b: (4.0 + 48.0 * epsilon) * epsilon,
        iccerrbound_c: (44.0 + 576.0 * epsilon) * epsilon * epsilon,
    }
};

/// Returns the exact-arithmetic constants (a compile-time constant).
#[inline(always)]
pub fn consts() -> &'static Consts {
    &CONSTS
}

/// Runtime port of Triangle's `exactinit()`, retained to validate [`CONSTS`].
#[cfg(test)]
fn exactinit_runtime() -> Consts {
    let half = 0.5f64;
    let mut epsilon = 1.0f64;
    let mut splitter = 1.0f64;
    let mut check = 1.0f64;
    let mut every_other = true;
    loop {
        let lastcheck = check;
        epsilon *= half;
        if every_other {
            splitter *= 2.0;
        }
        every_other = !every_other;
        check = 1.0 + epsilon;
        if check == 1.0 || check == lastcheck {
            break;
        }
    }
    splitter += 1.0;
    Consts {
        epsilon,
        splitter,
        resulterrbound: (3.0 + 8.0 * epsilon) * epsilon,
        ccwerrbound_a: (3.0 + 16.0 * epsilon) * epsilon,
        ccwerrbound_b: (2.0 + 12.0 * epsilon) * epsilon,
        ccwerrbound_c: (9.0 + 64.0 * epsilon) * epsilon * epsilon,
        iccerrbound_a: (10.0 + 96.0 * epsilon) * epsilon,
        iccerrbound_b: (4.0 + 48.0 * epsilon) * epsilon,
        iccerrbound_c: (44.0 + 576.0 * epsilon) * epsilon * epsilon,
    }
}

// ---------------------------------------------------------------------------
// Two-sum / two-product building blocks (Shewchuk's error-free transforms).
// Each returns (high, low) where high is the rounded result and low is the
// exact roundoff error.
// ---------------------------------------------------------------------------

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bvirt = x - a;
    let avirt = x - bvirt;
    let bround = b - bvirt;
    let around = a - avirt;
    (x, around + bround)
}

#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bvirt = x - a;
    (x, b - bvirt)
}

#[inline]
fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let x = a - b;
    let (_, y) = two_diff_tail(a, b, x);
    (x, y)
}

#[inline]
fn two_diff_tail(a: f64, b: f64, x: f64) -> (f64, f64) {
    let bvirt = a - x;
    let avirt = x + bvirt;
    let bround = bvirt - b;
    let around = a - avirt;
    (x, around + bround)
}

#[inline]
fn split(a: f64, splitter: f64) -> (f64, f64) {
    let c = splitter * a;
    let abig = c - a;
    let ahi = c - abig;
    (ahi, a - ahi)
}

#[inline]
fn two_product(a: f64, b: f64, splitter: f64) -> (f64, f64) {
    let x = a * b;
    let (ahi, alo) = split(a, splitter);
    let (bhi, blo) = split(b, splitter);
    let err1 = x - (ahi * bhi);
    let err2 = err1 - (alo * bhi);
    let err3 = err2 - (ahi * blo);
    (x, (alo * blo) - err3)
}

#[inline]
fn two_one_diff(a1: f64, a0: f64, b: f64) -> (f64, f64, f64) {
    let (i, x0) = two_diff(a0, b);
    let (x2, x1) = two_sum(a1, i);
    (x2, x1, x0)
}

#[inline]
fn two_two_diff(a1: f64, a0: f64, b1: f64, b0: f64) -> (f64, f64, f64, f64) {
    let (j, n0, x0) = two_one_diff(a1, a0, b0);
    let (x3, x2, x1) = two_one_diff(j, n0, b1);
    (x3, x2, x1, x0)
}

/// Port of `estimate()` (triangle.cpp:5121): sum of an expansion's components.
#[inline]
fn estimate(e: &[f64]) -> f64 {
    e.iter().sum()
}

/// Port of `fast_expansion_sum_zeroelim()` (triangle.cpp:4969). Writes `h = e + f`
/// into the caller-provided buffer `h` (must be ≥ `e.len() + f.len()`) and returns
/// the number of components written. Uses no heap allocation.
fn fast_expansion_sum_zeroelim(e: &[f64], f: &[f64], h: &mut [f64]) -> usize {
    let (elen, flen) = (e.len(), f.len());
    let mut eindex = 0usize;
    let mut findex = 0usize;
    let mut hindex = 0usize;
    let mut enow = e[0];
    let mut fnow = f[0];
    let mut q;
    if (fnow > enow) == (fnow > -enow) {
        q = enow;
        eindex += 1;
        enow = if eindex < elen { e[eindex] } else { enow };
    } else {
        q = fnow;
        findex += 1;
        fnow = if findex < flen { f[findex] } else { fnow };
    }
    if eindex < elen && findex < flen {
        let (qnew, hh);
        if (fnow > enow) == (fnow > -enow) {
            let r = fast_two_sum(enow, q);
            qnew = r.0;
            hh = r.1;
            eindex += 1;
            enow = if eindex < elen { e[eindex] } else { enow };
        } else {
            let r = fast_two_sum(fnow, q);
            qnew = r.0;
            hh = r.1;
            findex += 1;
            fnow = if findex < flen { f[findex] } else { fnow };
        }
        q = qnew;
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1;
        }
        while eindex < elen && findex < flen {
            let (qn, hh2);
            if (fnow > enow) == (fnow > -enow) {
                let r = two_sum(q, enow);
                qn = r.0;
                hh2 = r.1;
                eindex += 1;
                enow = if eindex < elen { e[eindex] } else { enow };
            } else {
                let r = two_sum(q, fnow);
                qn = r.0;
                hh2 = r.1;
                findex += 1;
                fnow = if findex < flen { f[findex] } else { fnow };
            }
            q = qn;
            if hh2 != 0.0 {
                h[hindex] = hh2;
                hindex += 1;
            }
        }
    }
    while eindex < elen {
        let (qn, hh2) = two_sum(q, enow);
        eindex += 1;
        enow = if eindex < elen { e[eindex] } else { enow };
        q = qn;
        if hh2 != 0.0 {
            h[hindex] = hh2;
            hindex += 1;
        }
    }
    while findex < flen {
        let (qn, hh2) = two_sum(q, fnow);
        findex += 1;
        fnow = if findex < flen { f[findex] } else { fnow };
        q = qn;
        if hh2 != 0.0 {
            h[hindex] = hh2;
            hindex += 1;
        }
    }
    if q != 0.0 || hindex == 0 {
        h[hindex] = q;
        hindex += 1;
    }
    hindex
}

/// Port of `scale_expansion_zeroelim()` (triangle.cpp:5064). Writes `h = b * e`
/// into `h` (must be ≥ `2 * e.len()`) and returns the number of components.
fn scale_expansion_zeroelim(e: &[f64], b: f64, splitter: f64, h: &mut [f64]) -> usize {
    let (bhi, blo) = split(b, splitter);
    let mut hindex = 0usize;
    // Two_Product_Presplit(e[0], b, bhi, blo)
    let mut q;
    {
        let x = e[0] * b;
        let (ahi, alo) = split(e[0], splitter);
        let err1 = x - (ahi * bhi);
        let err2 = err1 - (alo * bhi);
        let err3 = err2 - (ahi * blo);
        let hh = (alo * blo) - err3;
        q = x;
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1;
        }
    }
    for &enow in &e[1..] {
        let product1 = enow * b;
        let (ahi, alo) = split(enow, splitter);
        let err1 = product1 - (ahi * bhi);
        let err2 = err1 - (alo * bhi);
        let err3 = err2 - (ahi * blo);
        let product0 = (alo * blo) - err3;
        let (sum, hh) = two_sum(q, product0);
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1;
        }
        let (qn, hh2) = fast_two_sum(product1, sum);
        q = qn;
        if hh2 != 0.0 {
            h[hindex] = hh2;
            hindex += 1;
        }
    }
    if q != 0.0 || hindex == 0 {
        h[hindex] = q;
        hindex += 1;
    }
    hindex
}

// ---------------------------------------------------------------------------
// orient2d / counterclockwise — full port of triangle.cpp:5249.
// ---------------------------------------------------------------------------

/// Returns a positive value if `pa`, `pb`, `pc` are in counterclockwise order,
/// negative if clockwise, zero if collinear. Robust (exact sign).
///
/// `noexact = true` reproduces Triangle's `-X` switch (fast, non-robust path).
#[inline]
pub fn orient2d(pa: Point, pb: Point, pc: Point, noexact: bool) -> f64 {
    let detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    let detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
    let det = detleft - detright;

    if noexact {
        return det;
    }

    let detsum;
    if detleft > 0.0 {
        if detright <= 0.0 {
            return det;
        }
        detsum = detleft + detright;
    } else if detleft < 0.0 {
        if detright >= 0.0 {
            return det;
        }
        detsum = -detleft - detright;
    } else {
        return det;
    }

    let c = consts();
    let errbound = c.ccwerrbound_a * detsum;
    if det >= errbound || -det >= errbound {
        return det;
    }
    counterclockwiseadapt(pa, pb, pc, detsum, c)
}

/// Port of `counterclockwiseadapt()` (triangle.cpp:5160).
fn counterclockwiseadapt(pa: Point, pb: Point, pc: Point, detsum: f64, c: &Consts) -> f64 {
    let s = c.splitter;
    let acx = pa[0] - pc[0];
    let bcx = pb[0] - pc[0];
    let acy = pa[1] - pc[1];
    let bcy = pb[1] - pc[1];

    let (detleft, detlefttail) = two_product(acx, bcy, s);
    let (detright, detrighttail) = two_product(acy, bcx, s);

    let (b3, b2, b1, b0) = two_two_diff(detleft, detlefttail, detright, detrighttail);
    let bb = [b0, b1, b2, b3];

    let mut det = estimate(&bb);
    let mut errbound = c.ccwerrbound_b * detsum;
    if det >= errbound || -det >= errbound {
        return det;
    }

    let (_, acxtail) = two_diff_tail(pa[0], pc[0], acx);
    let (_, bcxtail) = two_diff_tail(pb[0], pc[0], bcx);
    let (_, acytail) = two_diff_tail(pa[1], pc[1], acy);
    let (_, bcytail) = two_diff_tail(pb[1], pc[1], bcy);

    if acxtail == 0.0 && acytail == 0.0 && bcxtail == 0.0 && bcytail == 0.0 {
        return det;
    }

    errbound = c.ccwerrbound_c * detsum + c.resulterrbound * det.abs();
    det += (acx * bcytail + bcy * acxtail) - (acy * bcxtail + bcx * acytail);
    if det >= errbound || -det >= errbound {
        return det;
    }

    let (s1, s0) = two_product(acxtail, bcy, s);
    let (t1, t0) = two_product(acytail, bcx, s);
    let (u3, u2, u1, u0) = two_two_diff(s1, s0, t1, t0);
    let u = [u0, u1, u2, u3];
    let mut c1 = [0.0f64; 8];
    let c1len = fast_expansion_sum_zeroelim(&bb, &u, &mut c1);

    let (s1, s0) = two_product(acx, bcytail, s);
    let (t1, t0) = two_product(acy, bcxtail, s);
    let (u3, u2, u1, u0) = two_two_diff(s1, s0, t1, t0);
    let u = [u0, u1, u2, u3];
    let mut c2 = [0.0f64; 12];
    let c2len = fast_expansion_sum_zeroelim(&c1[..c1len], &u, &mut c2);

    let (s1, s0) = two_product(acxtail, bcytail, s);
    let (t1, t0) = two_product(acytail, bcxtail, s);
    let (u3, u2, u1, u0) = two_two_diff(s1, s0, t1, t0);
    let u = [u0, u1, u2, u3];
    let mut d = [0.0f64; 16];
    let dlen = fast_expansion_sum_zeroelim(&c2[..c2len], &u, &mut d);

    d[dlen - 1]
}

// ---------------------------------------------------------------------------
// incircle — fast path + first two refinement stages (triangle.cpp:5899, 5319),
// deepest exact stage delegated to the `robust` crate for a correct sign.
// ---------------------------------------------------------------------------

/// Returns a positive value if `pd` lies inside the circle through `pa`, `pb`,
/// `pc` (which must be counterclockwise), negative if outside, zero if cocircular.
/// Robust (exact sign).
#[inline]
pub fn incircle(pa: Point, pb: Point, pc: Point, pd: Point, noexact: bool) -> f64 {
    let adx = pa[0] - pd[0];
    let bdx = pb[0] - pd[0];
    let cdx = pc[0] - pd[0];
    let ady = pa[1] - pd[1];
    let bdy = pb[1] - pd[1];
    let cdy = pc[1] - pd[1];

    let bdxcdy = bdx * cdy;
    let cdxbdy = cdx * bdy;
    let alift = adx * adx + ady * ady;

    let cdxady = cdx * ady;
    let adxcdy = adx * cdy;
    let blift = bdx * bdx + bdy * bdy;

    let adxbdy = adx * bdy;
    let bdxady = bdx * ady;
    let clift = cdx * cdx + cdy * cdy;

    let det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady);

    if noexact {
        return det;
    }

    let c = consts();
    let permanent = (bdxcdy.abs() + cdxbdy.abs()) * alift
        + (cdxady.abs() + adxcdy.abs()) * blift
        + (adxbdy.abs() + bdxady.abs()) * clift;
    let errbound = c.iccerrbound_a * permanent;
    if det > errbound || -det > errbound {
        return det;
    }

    match incircle_adapt(pa, pb, pc, pd, permanent, c) {
        Some(refined) => refined,
        // Deepest exact stage: delegate sign to the `robust` crate. Magnitude is
        // irrelevant here — the mesh only branches on the sign of incircle.
        None => robust::incircle(
            robust::Coord { x: pa[0], y: pa[1] },
            robust::Coord { x: pb[0], y: pb[1] },
            robust::Coord { x: pc[0], y: pc[1] },
            robust::Coord { x: pd[0], y: pd[1] },
        ),
    }
}

/// Stages B and C of `incircleadapt()` (triangle.cpp:5398-5460). Returns `Some`
/// when the sign is resolved by these stages, `None` when the deepest exact
/// expansion is required.
fn incircle_adapt(pa: Point, pb: Point, pc: Point, pd: Point, permanent: f64, c: &Consts) -> Option<f64> {
    let s = c.splitter;
    let adx = pa[0] - pd[0];
    let bdx = pb[0] - pd[0];
    let cdx = pc[0] - pd[0];
    let ady = pa[1] - pd[1];
    let bdy = pb[1] - pd[1];
    let cdy = pc[1] - pd[1];

    // bc, ca, ab cross-term expansions.
    let bc = {
        let (bdxcdy1, bdxcdy0) = two_product(bdx, cdy, s);
        let (cdxbdy1, cdxbdy0) = two_product(cdx, bdy, s);
        let (bc3, bc2, bc1, bc0) = two_two_diff(bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0);
        [bc0, bc1, bc2, bc3]
    };
    let mut adet = [0.0f64; 32];
    let alen = lift(&bc, adx, ady, s, &mut adet);

    let ca = {
        let (cdxady1, cdxady0) = two_product(cdx, ady, s);
        let (adxcdy1, adxcdy0) = two_product(adx, cdy, s);
        let (ca3, ca2, ca1, ca0) = two_two_diff(cdxady1, cdxady0, adxcdy1, adxcdy0);
        [ca0, ca1, ca2, ca3]
    };
    let mut bdet = [0.0f64; 32];
    let blen = lift(&ca, bdx, bdy, s, &mut bdet);

    let ab = {
        let (adxbdy1, adxbdy0) = two_product(adx, bdy, s);
        let (bdxady1, bdxady0) = two_product(bdx, ady, s);
        let (ab3, ab2, ab1, ab0) = two_two_diff(adxbdy1, adxbdy0, bdxady1, bdxady0);
        [ab0, ab1, ab2, ab3]
    };
    let mut cdet = [0.0f64; 32];
    let clen = lift(&ab, cdx, cdy, s, &mut cdet);

    let mut abdet = [0.0f64; 64];
    let ablen = fast_expansion_sum_zeroelim(&adet[..alen], &bdet[..blen], &mut abdet);
    let mut fin1 = [0.0f64; 96];
    let finlen = fast_expansion_sum_zeroelim(&abdet[..ablen], &cdet[..clen], &mut fin1);
    let fin1 = &fin1[..finlen];

    let mut det = estimate(fin1);
    let mut errbound = c.iccerrbound_b * permanent;
    if det >= errbound || -det >= errbound {
        return Some(det);
    }

    // Stage C: floating-point correction term.
    let (_, adxtail) = two_diff_tail(pa[0], pd[0], adx);
    let (_, adytail) = two_diff_tail(pa[1], pd[1], ady);
    let (_, bdxtail) = two_diff_tail(pb[0], pd[0], bdx);
    let (_, bdytail) = two_diff_tail(pb[1], pd[1], bdy);
    let (_, cdxtail) = two_diff_tail(pc[0], pd[0], cdx);
    let (_, cdytail) = two_diff_tail(pc[1], pd[1], cdy);
    if adxtail == 0.0
        && bdxtail == 0.0
        && cdxtail == 0.0
        && adytail == 0.0
        && bdytail == 0.0
        && cdytail == 0.0
    {
        return Some(det);
    }

    errbound = c.iccerrbound_c * permanent + c.resulterrbound * det.abs();
    det += ((adx * adx + ady * ady)
        * ((bdx * cdytail + cdy * bdxtail) - (bdy * cdxtail + cdx * bdytail))
        + 2.0 * (adx * adxtail + ady * adytail) * (bdx * cdy - bdy * cdx))
        + ((bdx * bdx + bdy * bdy)
            * ((cdx * adytail + ady * cdxtail) - (cdy * adxtail + adx * cdytail))
            + 2.0 * (bdx * bdxtail + bdy * bdytail) * (cdx * ady - cdy * adx))
        + ((cdx * cdx + cdy * cdy)
            * ((adx * bdytail + bdy * adxtail) - (ady * bdxtail + bdx * adytail))
            + 2.0 * (cdx * cdxtail + cdy * cdytail) * (adx * bdy - ady * bdx));
    if det >= errbound || -det >= errbound {
        return Some(det);
    }

    None
}

/// Computes `x*x*e + y*y*e` (the per-vertex "lift" expansion used by incircle)
/// into `out` (must be ≥ 32), returning the number of components. No heap use.
#[inline]
fn lift(e: &[f64], x: f64, y: f64, s: f64, out: &mut [f64]) -> usize {
    let mut xe = [0.0f64; 8];
    let xelen = scale_expansion_zeroelim(e, x, s, &mut xe);
    let mut xxe = [0.0f64; 16];
    let xxelen = scale_expansion_zeroelim(&xe[..xelen], x, s, &mut xxe);
    let mut ye = [0.0f64; 8];
    let yelen = scale_expansion_zeroelim(e, y, s, &mut ye);
    let mut yye = [0.0f64; 16];
    let yyelen = scale_expansion_zeroelim(&ye[..yelen], y, s, &mut yye);
    fast_expansion_sum_zeroelim(&xxe[..xxelen], &yye[..yyelen], out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_consts_match_runtime_exactinit() {
        let r = exactinit_runtime();
        let c = consts();
        assert_eq!(c.epsilon, r.epsilon, "epsilon");
        assert_eq!(c.splitter, r.splitter, "splitter");
        assert_eq!(c.resulterrbound, r.resulterrbound, "resulterrbound");
        assert_eq!(c.ccwerrbound_a, r.ccwerrbound_a, "ccwA");
        assert_eq!(c.ccwerrbound_b, r.ccwerrbound_b, "ccwB");
        assert_eq!(c.ccwerrbound_c, r.ccwerrbound_c, "ccwC");
        assert_eq!(c.iccerrbound_a, r.iccerrbound_a, "iccA");
        assert_eq!(c.iccerrbound_b, r.iccerrbound_b, "iccB");
        assert_eq!(c.iccerrbound_c, r.iccerrbound_c, "iccC");
    }

    #[test]
    fn orient2d_basic_signs() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c_ccw = [0.0, 1.0];
        let c_cw = [0.0, -1.0];
        let c_col = [2.0, 0.0];
        assert!(orient2d(a, b, c_ccw, false) > 0.0);
        assert!(orient2d(a, b, c_cw, false) < 0.0);
        assert_eq!(orient2d(a, b, c_col, false), 0.0);
    }

    #[test]
    fn incircle_basic_signs() {
        // Triangle (0,0),(1,0),(0,1) counterclockwise; circumcircle passes through
        // these three. (0.25,0.25) is inside, (2,2) is outside, (1,1) is on it.
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        assert!(incircle(a, b, c, [0.25, 0.25], false) > 0.0);
        assert!(incircle(a, b, c, [2.0, 2.0], false) < 0.0);
        assert_eq!(incircle(a, b, c, [1.0, 1.0], false), 0.0);
    }

    #[test]
    fn orient2d_matches_robust_sign() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        for _ in 0..200_000 {
            // Mix of generic and near-degenerate coordinates.
            let scale = if rng.gen_bool(0.5) { 1.0 } else { 1e-9 };
            let a = [rng.gen::<f64>() * scale, rng.gen::<f64>() * scale];
            let b = [rng.gen::<f64>() * scale, rng.gen::<f64>() * scale];
            let c = [rng.gen::<f64>() * scale, rng.gen::<f64>() * scale];
            let ours = orient2d(a, b, c, false).signum() as i64 * (orient2d(a, b, c, false) != 0.0) as i64;
            let theirs = {
                let v = robust::orient2d(
                    robust::Coord { x: a[0], y: a[1] },
                    robust::Coord { x: b[0], y: b[1] },
                    robust::Coord { x: c[0], y: c[1] },
                );
                v.signum() as i64 * (v != 0.0) as i64
            };
            assert_eq!(ours, theirs, "orient2d sign mismatch for {a:?} {b:?} {c:?}");
        }
    }

    #[test]
    fn incircle_matches_robust_sign() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let mut checked = 0u64;
        for _ in 0..200_000 {
            let scale = if rng.gen_bool(0.5) { 1.0 } else { 1e-7 };
            let a = [rng.gen::<f64>() * scale, rng.gen::<f64>() * scale];
            let b = [rng.gen::<f64>() * scale, rng.gen::<f64>() * scale];
            let c = [rng.gen::<f64>() * scale, rng.gen::<f64>() * scale];
            let d = [rng.gen::<f64>() * scale, rng.gen::<f64>() * scale];
            // incircle assumes a,b,c CCW; orient to satisfy that.
            let (b, c) = if orient2d(a, b, c, false) < 0.0 { (c, b) } else { (b, c) };
            let ours = incircle(a, b, c, d, false);
            let theirs = robust::incircle(
                robust::Coord { x: a[0], y: a[1] },
                robust::Coord { x: b[0], y: b[1] },
                robust::Coord { x: c[0], y: c[1] },
                robust::Coord { x: d[0], y: d[1] },
            );
            let so = ours.partial_cmp(&0.0).unwrap();
            let st = theirs.partial_cmp(&0.0).unwrap();
            assert_eq!(so, st, "incircle sign mismatch for {a:?} {b:?} {c:?} {d:?}: {ours} vs {theirs}");
            checked += 1;
        }
        assert_eq!(checked, 200_000);
    }
}
