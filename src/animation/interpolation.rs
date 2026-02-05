//! Centralized interpolation utilities for animation.
//!
//! This module provides a single source of truth for progress values
//! used in animation interpolation. All animation code (backbone, sidechain,
//! bonds) should use `InterpolationContext` to ensure consistent timing.

use glam::Vec3;

/// Single source of truth for progress values in a single animation frame.
///
/// This context is computed once per frame from the behavior and raw progress,
/// then passed to all interpolation functions to ensure consistency.
///
/// # Why This Exists
///
/// Different animation behaviors may apply different easing curves or have
/// multiple phases (e.g., CollapseExpand). Without a centralized context,
/// different parts of the code (backbone vs sidechain interpolation) could
/// compute different progress values, causing visual desync.
///
/// # Usage
///
/// ```ignore
/// // In animation update loop:
/// let ctx = behavior.compute_context(raw_t);
///
/// // For all interpolations in this frame:
/// let backbone_pos = lerp_position(&ctx, start_backbone, end_backbone);
/// let sidechain_pos = lerp_position(&ctx, start_sidechain, end_sidechain);
/// // Both use the same eased_t value
/// ```
#[derive(Debug, Clone, Copy)]
pub struct InterpolationContext {
    /// Raw progress (0.0 to 1.0), unmodified from animation timer.
    pub raw_t: f32,
    /// Eased progress for smooth animations. This is the primary value
    /// that all interpolation should use unless behavior-specific.
    pub eased_t: f32,
    /// For multi-phase behaviors: progress within current phase (0.0 to 1.0).
    pub phase_t: Option<f32>,
    /// For multi-phase behaviors: eased progress within current phase.
    pub phase_eased_t: Option<f32>,
}

impl InterpolationContext {
    /// Create a simple context with just raw and eased values.
    pub fn simple(raw_t: f32, eased_t: f32) -> Self {
        Self {
            raw_t,
            eased_t,
            phase_t: None,
            phase_eased_t: None,
        }
    }

    /// Create a context with phase information (for CollapseExpand, etc).
    pub fn with_phase(raw_t: f32, eased_t: f32, phase_t: f32, phase_eased_t: f32) -> Self {
        Self {
            raw_t,
            eased_t,
            phase_t: Some(phase_t),
            phase_eased_t: Some(phase_eased_t),
        }
    }

    /// Create an "identity" context representing animation complete (t=1.0).
    /// Use this when no animation is running.
    pub fn identity() -> Self {
        Self {
            raw_t: 1.0,
            eased_t: 1.0,
            phase_t: None,
            phase_eased_t: None,
        }
    }

    /// Create a linear context (no easing).
    pub fn linear(raw_t: f32) -> Self {
        Self::simple(raw_t, raw_t)
    }

    /// Get the unified progress value that all interpolation should use.
    ///
    /// For most behaviors, this returns `eased_t`. Override if needed.
    #[inline]
    pub fn unified_t(&self) -> f32 {
        self.eased_t
    }
}

impl Default for InterpolationContext {
    fn default() -> Self {
        Self::identity()
    }
}

/// Interpolate between two positions using the unified progress from context.
///
/// This is the primary interpolation function for positions.
/// All position interpolation should use this to ensure consistency.
#[inline]
pub fn lerp_position(ctx: &InterpolationContext, start: Vec3, end: Vec3) -> Vec3 {
    let t = ctx.unified_t();
    start + (end - start) * t
}

/// Interpolate between two f32 values using the unified progress from context.
#[inline]
pub fn lerp_f32(ctx: &InterpolationContext, start: f32, end: f32) -> f32 {
    let t = ctx.unified_t();
    start + (end - start) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_context() {
        let ctx = InterpolationContext::identity();
        assert_eq!(ctx.raw_t, 1.0);
        assert_eq!(ctx.eased_t, 1.0);
        assert_eq!(ctx.unified_t(), 1.0);
    }

    #[test]
    fn test_simple_context() {
        let ctx = InterpolationContext::simple(0.5, 0.7);
        assert_eq!(ctx.raw_t, 0.5);
        assert_eq!(ctx.eased_t, 0.7);
        assert_eq!(ctx.unified_t(), 0.7);
    }

    #[test]
    fn test_lerp_position() {
        let ctx = InterpolationContext::simple(0.5, 0.5);
        let start = Vec3::ZERO;
        let end = Vec3::new(10.0, 20.0, 30.0);
        let result = lerp_position(&ctx, start, end);
        assert!((result - Vec3::new(5.0, 10.0, 15.0)).length() < 0.001);
    }

    #[test]
    fn test_lerp_f32() {
        let ctx = InterpolationContext::simple(0.25, 0.25);
        let result = lerp_f32(&ctx, 0.0, 100.0);
        assert!((result - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_with_phase() {
        let ctx = InterpolationContext::with_phase(0.3, 0.4, 0.6, 0.8);
        assert_eq!(ctx.raw_t, 0.3);
        assert_eq!(ctx.eased_t, 0.4);
        assert_eq!(ctx.phase_t, Some(0.6));
        assert_eq!(ctx.phase_eased_t, Some(0.8));
    }
}
