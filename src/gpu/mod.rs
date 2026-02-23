//! GPU resource management utilities.
//!
//! Provides wgpu device/surface initialization, dynamic buffer management,
//! per-residue color storage, and shader composition.

/// Growable GPU buffers with automatic reallocation.
pub mod dynamic_buffer;
/// wgpu device, surface, and queue initialization.
pub mod render_context;
/// Per-residue color storage buffer for GPU shaders.
pub mod residue_color;
/// WGSL shader composition with `#import` support via naga-oil.
pub mod shader_composer;
/// Framework-agnostic render-target texture abstraction.
pub mod texture;
