//! Shared utilities for the rendering engine.
//!
//! Helpers for frame timing, trajectory playback, hashing, score-to-color
//! mapping, sheet-surface adjustments, easing curves, and bond topology
//! lookups.

/// Residue-level bond and element lookups from `foldit_conv`.
pub mod bond_topology;
/// Per-frame timing, FPS capping, and delta-time tracking.
pub mod frame_timing;
/// Fast hashing helpers for change detection on Vec3 data.
pub mod hash;
/// Score-to-color gradient mapping.
pub mod score_color;
/// Sheet-surface sidechain position adjustment.
pub mod sheet_adjust;
