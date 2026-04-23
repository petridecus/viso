//! Camera system for 3D scene viewing.
//!
//! Provides an orbital camera with rotation, panning, zoom, animation,
//! frustum culling, and input handling.

/// Orbital camera controller managing rotation, pan, zoom, and GPU resources.
pub(crate) mod controller;
/// Core camera struct and GPU uniform types.
pub(crate) mod core;
/// Fit-to-molecule-data helpers.
pub(crate) mod fit;
/// View frustum extraction and intersection tests.
pub(crate) mod frustum;
