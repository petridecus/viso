//! Core trait for animation behaviors.

use std::sync::Arc;
use std::time::Duration;

use super::super::interpolation::InterpolationContext;
use super::state::ResidueVisualState;

/// How to handle a new target arriving while an animation is in progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreemptionStrategy {
    /// Start new animation from current visual position to new target.
    /// Animation time resets to 0.
    #[default]
    Restart,

    /// Ignore new targets until current animation completes.
    Ignore,

    /// Blend toward new target while maintaining current velocity.
    /// Keeps animation continuity without resetting time.
    Blend,
}

/// Defines how a structural change should be animated.
///
/// This trait is the core abstraction for animation behaviors. Implementations
/// control the interpolation curve, timing, and visual effects.
///
/// # Examples
///
/// Simple interpolation:
/// ```ignore
/// impl AnimationBehavior for MyBehavior {
///     fn compute_state(&self, t: f32, start: &ResidueVisualState, end: &ResidueVisualState) -> ResidueVisualState {
///         start.lerp(end, t)  // Linear interpolation
///     }
///     fn duration(&self) -> Duration { Duration::from_millis(300) }
/// }
/// ```
///
/// Multi-phase animation:
/// ```ignore
/// fn compute_state(&self, t: f32, start: &ResidueVisualState, end: &ResidueVisualState) -> ResidueVisualState {
///     if t < 0.5 {
///         // Phase 1: collapse
///         let phase_t = t * 2.0;
///         collapse(start, phase_t)
///     } else {
///         // Phase 2: expand
///         let phase_t = (t - 0.5) * 2.0;
///         expand(end, phase_t)
///     }
/// }
/// ```
pub trait AnimationBehavior: Send + Sync {
    /// Get the eased time value for a given raw progress t (0.0 to 1.0).
    ///
    /// This is the single source of truth for easing. Override this to apply
    /// custom easing curves. Used by both backbone and sidechain interpolation.
    ///
    /// Default implementation: linear (no easing).
    fn eased_t(&self, t: f32) -> f32 {
        t
    }

    /// Interpolate a position, optionally collapsing toward a point.
    ///
    /// Used for sidechain animation where atoms should collapse toward
    /// the backbone (CA) during mutations.
    ///
    /// Default implementation: lerp using eased_t (ignores collapse_point).
    /// CollapseExpand overrides to do two-phase collapse/expand.
    fn interpolate_position(
        &self,
        t: f32,
        start: glam::Vec3,
        end: glam::Vec3,
        _collapse_point: glam::Vec3,
    ) -> glam::Vec3 {
        // Use eased_t for consistent easing with backbone
        let eased = self.eased_t(t);
        start + (end - start) * eased
    }
    /// Compute the visual state at time t (0.0 to 1.0).
    ///
    /// This is the core method that defines the animation curve.
    /// Simple implementations just lerp with easing; complex ones
    /// (like CollapseExpand) can have multiple phases.
    fn compute_state(
        &self,
        t: f32,
        start: &ResidueVisualState,
        end: &ResidueVisualState,
    ) -> ResidueVisualState;

    /// Total duration of the animation.
    fn duration(&self) -> Duration;

    /// How to handle preemption when a new target arrives mid-animation.
    fn preemption(&self) -> PreemptionStrategy {
        PreemptionStrategy::Restart
    }

    /// Optional name for debugging/logging.
    fn name(&self) -> &'static str {
        "unnamed"
    }

    /// Compute the interpolation context for this behavior at raw progress t.
    ///
    /// This provides a single source of truth for all progress values in a frame.
    /// Both backbone and sidechain interpolation should use the same context
    /// to ensure they move in sync.
    ///
    /// Default implementation: applies eased_t for simple behaviors.
    /// Complex behaviors (CollapseExpand) should override with phase-aware logic.
    fn compute_context(&self, raw_t: f32) -> InterpolationContext {
        InterpolationContext::simple(raw_t, self.eased_t(raw_t))
    }
}

/// Type alias for shared behavior references.
pub type SharedBehavior = Arc<dyn AnimationBehavior>;

/// Create a shared behavior from any AnimationBehavior implementation.
pub fn shared<B: AnimationBehavior + 'static>(behavior: B) -> SharedBehavior {
    Arc::new(behavior)
}
