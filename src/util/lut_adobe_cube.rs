//! Adobe / DaVinci Resolve ASCII `.cube` LUT parsing (CPU).

/// In-memory RGB samples for a 3D LUT of edge length [`Self::size`].

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LutRgbF32Cube3d {
    /// Cube dimension (`N` in `LUT_3D_SIZE N`).
    pub(crate) size: u32,
    /// Flattened RGB triplets; length should equal `size³` after parsing.
    pub(crate) rgb: Vec<[f32; 3]>,
}

/// Errors emitted while parsing or validating `.cube` LUT files.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LutCubeParseError {
    /// size outside the supported range.
    InvalidLutSize { size: u32 },
    /// number of RGB samples not match `size^3`.
    WrongRgbCount { expected: usize, actual: usize },
}

impl std::fmt::Display for LutCubeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLutSize { size } => {
                write!(f, "LUT size {size} is outside supported bounds")
            }
            Self::WrongRgbCount { expected, actual } => write!(
                f,
                "LUT cube has {actual} RGB samples but expected {expected}"
            ),
        }
    }
}

impl std::error::Error for LutCubeParseError {}
