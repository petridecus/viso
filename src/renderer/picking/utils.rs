//! Shared picking utilities for ball-and-stick and other capsule-based
//! renderers.

use crate::renderer::impostor::capsule::CapsuleInstance;

/// Pick-ID offset for small-molecule atoms so they don't collide with protein
/// residue indices.
pub const SMALL_MOLECULE_PICK_OFFSET: u32 = 100_000;

/// Build a degenerate capsule (sphere) for picking at `pos` with `radius`.
///
/// Both endpoints are identical so the GPU draws a sphere impostor; color
/// channels are zeroed because picking only uses the residue-index channel.
pub fn picking_sphere(
    pos: [f32; 3],
    radius: f32,
    pick_id: u32,
) -> CapsuleInstance {
    CapsuleInstance {
        endpoint_a: [pos[0], pos[1], pos[2], radius],
        endpoint_b: [pos[0], pos[1], pos[2], pick_id as f32],
        color_a: [0.0; 4],
        color_b: [0.0; 4],
    }
}

/// Build a zero-color bond capsule for picking between two endpoints.
pub fn picking_bond(
    pos_a: [f32; 3],
    pos_b: [f32; 3],
    radius: f32,
    pick_id: u32,
) -> CapsuleInstance {
    CapsuleInstance {
        endpoint_a: [pos_a[0], pos_a[1], pos_a[2], radius],
        endpoint_b: [pos_b[0], pos_b[1], pos_b[2], pick_id as f32],
        color_a: [0.0; 4],
        color_b: [0.0; 4],
    }
}
