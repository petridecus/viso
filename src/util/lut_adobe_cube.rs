//! Adobe / DaVinci Resolve ASCII `.cube` LUT parsing (CPU).
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LutRgbF32Cube3d {
    /// Cube dimension (`N` in `LUT_3D_SIZE N`).
    pub(crate) size: u32,
    /// Flattened RGB triplets; length must equal `size³` once parsing lands.
    pub(crate) rgb: Vec<[f32; 3]>,
}
