//! Camera system for 3D scene viewing.
//!
//! Provides an orbital camera with rotation, panning, zoom, animation,
//! frustum culling, and input handling.

/// Orbital camera controller managing rotation, pan, zoom, and GPU resources.
pub mod controller;
/// Core camera struct and GPU uniform types.
pub mod core;
/// View frustum extraction and intersection tests.
pub mod frustum;
/// Window-event-based camera input handler.
pub mod input;
