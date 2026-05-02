//! Adobe / DaVinci Resolve ASCII `.cube` LUT parsing (CPU).

use crate::VisoError;

/// In-memory RGB samples for a 3D LUT of edge length [`Self::size`].
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LutRgbF32Cube3d {
    /// Cube dimension (`N` in `LUT_3D_SIZE N`).
    pub(crate) size: u32,
    /// Flattened RGB triplets; length should equal `size³` after parsing.
    pub(crate) rgb: Vec<[f32; 3]>,
}

impl LutRgbF32Cube3d {
    /// Maximum supported size.
    pub(crate) const MAX_SIZE: u32 = 256;

    /// Build a LUT after validating `size` and `rgb.len() == size³`.
    ///
    /// # Errors
    ///
    /// Returns [`LutCubeParseError`] when `size` is outside `2..=MAX_SIZE`, when
    /// `size³` does not fit in [`usize`], or when the RGB sample count is
    /// wrong.
    #[allow(dead_code)] // No call sites until parsing / tests land.
    pub(crate) fn new(
        size: u32,
        rgb: Vec<[f32; 3]>,
    ) -> Result<Self, LutCubeParseError> {
        if !(2..=Self::MAX_SIZE).contains(&size) {
            return Err(LutCubeParseError::InvalidLutSize { size });
        }

        let expected = expected_lut_sample_count(size)
            .ok_or(LutCubeParseError::InvalidLutSize { size })?;

        let actual = rgb.len();
        if actual != expected {
            return Err(LutCubeParseError::WrongRgbCount { expected, actual });
        }

        Ok(Self { size, rgb })
    }
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

/// Map to `VisoError::OptionsParse`.
impl From<LutCubeParseError> for VisoError {
    fn from(value: LutCubeParseError) -> Self {
        Self::OptionsParse(value.to_string())
    }
}

/// Returns `size³` as [`usize`] if fits; otherwise [`None`]
#[must_use]
pub(crate) fn expected_lut_sample_count(size: u32) -> Option<usize> {
    let n = usize::try_from(size).ok()?;
    Some(n.checked_mul(n)?.checked_mul(n)?)
}
