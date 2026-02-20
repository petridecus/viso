//! GPU-based object picking and selection management.
//!
//! Renders residue indices to an offscreen buffer and reads back the pixel
//! under the cursor to determine which residue was clicked or hovered.

#[allow(clippy::module_inception)]
mod picking;
pub(crate) mod picking_state;

pub use picking::{Picking, PickingGeometry, SelectionBuffer};
