//! Error type for triangulation requests.

/// Errors returned by the [`crate::Triangulator`] entry points.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriangleError {
    /// Fewer than three input points were supplied.
    TooFewPoints,
    /// A segment or triangle references a point index that is out of range.
    InvalidIndex,
}

impl std::fmt::Display for TriangleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TriangleError::TooFewPoints => write!(f, "at least 3 input points are required"),
            TriangleError::InvalidIndex => write!(f, "input references an out-of-range point index"),
        }
    }
}

impl std::error::Error for TriangleError {}
