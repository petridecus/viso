//! GPU-based object picking and selection management.
//!
//! Renders residue indices to an offscreen buffer and reads back the pixel
//! under the cursor to determine which residue was clicked or hovered.

mod pipeline;
pub(crate) mod state;
pub(crate) mod utils;

pub use pipeline::{Picking, PickingGeometry, SelectionBuffer};
