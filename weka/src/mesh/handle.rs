//! Packed mesh handles — the safe-Rust equivalent of Triangle's pointer-with-
//! orientation encoding (triangle.cpp:947 `encode`/`decode`, :1167 `sencode`).
//!
//! A [`TriHandle`] packs a triangle arena index with an orientation in `0..3`;
//! a [`SubHandle`] packs a subsegment index with an orientation in `0..2`. Arena
//! index `0` is reserved for the `dummytri`/`dummysub` sentinel ("outer space"),
//! so the all-zero handle is the canonical "no neighbor".

/// Vertex identifier (index into the vertex arena).
pub type Vid = u32;

/// Sentinel meaning "no vertex" (Triangle stores a NULL vertex pointer here).
pub const NO_VERTEX: Vid = u32::MAX;

/// `(orient + 1) mod 3`, as a lookup table (triangle.cpp:937 `plus1mod3`).
pub const PLUS1MOD3: [usize; 3] = [1, 2, 0];
/// `(orient + 2) mod 3` i.e. `(orient - 1) mod 3` (triangle.cpp:938 `minus1mod3`).
pub const MINUS1MOD3: [usize; 3] = [2, 0, 1];

/// An oriented triangle handle: arena index in the high bits, edge orientation
/// (`0..3`) in the low 2 bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TriHandle(pub u32);

impl TriHandle {
    /// The "outer space" sentinel triangle (arena index 0, orientation 0).
    pub const DUMMY: TriHandle = TriHandle(0);

    #[inline]
    pub fn new(index: u32, orient: usize) -> Self {
        debug_assert!(orient < 3);
        TriHandle((index << 2) | orient as u32)
    }

    #[inline]
    pub fn index(self) -> usize {
        (self.0 >> 2) as usize
    }

    #[inline]
    pub fn orient(self) -> usize {
        (self.0 & 3) as usize
    }

    /// True if this handle refers to the `dummytri` sentinel.
    #[inline]
    pub fn is_dummy(self) -> bool {
        self.index() == 0
    }

    /// Same triangle, orientation set to `orient`.
    #[inline]
    pub fn with_orient(self, orient: usize) -> Self {
        TriHandle::new(self.index() as u32, orient)
    }
}

impl std::fmt::Debug for TriHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tri#{}.{}", self.index(), self.orient())
    }
}

/// An oriented subsegment handle: arena index in the high bits, orientation
/// (`0..2`) in the low bit.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubHandle(pub u32);

impl SubHandle {
    /// The omnipresent `dummysub` sentinel (arena index 0, orientation 0).
    pub const DUMMY: SubHandle = SubHandle(0);

    #[inline]
    pub fn new(index: u32, ssorient: usize) -> Self {
        debug_assert!(ssorient < 2);
        SubHandle((index << 1) | ssorient as u32)
    }

    #[inline]
    pub fn index(self) -> usize {
        (self.0 >> 1) as usize
    }

    #[inline]
    pub fn orient(self) -> usize {
        (self.0 & 1) as usize
    }

    /// True if this handle refers to the `dummysub` sentinel.
    #[inline]
    pub fn is_dummy(self) -> bool {
        self.index() == 0
    }
}

impl std::fmt::Debug for SubHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sub#{}.{}", self.index(), self.orient())
    }
}
