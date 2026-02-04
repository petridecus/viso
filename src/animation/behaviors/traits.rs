//! Core trait for animation behaviors.

use std::sync::Arc;
use std::time::Duration;

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
    /// Interpolate a position, optionally collapsing toward a point.
    ///
    /// Used for sidechain animation where atoms should collapse toward
    /// the backbone (CA) during mutations.
    ///
    /// Default implementation: linear interpolation (ignores collapse_point).
    /// CollapseExpand overrides to do two-phase collapse/expand.
    fn interpolate_position(
        &self,
        t: f32,
        start: glam::Vec3,
        end: glam::Vec3,
        _collapse_point: glam::Vec3,
    ) -> glam::Vec3 {
        // Default: linear interpolation, ignore collapse_point
        start + (end - start) * t
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
}

/// Type alias for shared behavior references.
pub type SharedBehavior = Arc<dyn AnimationBehavior>;

/// Create a shared behavior from any AnimationBehavior implementation.
pub fn shared<B: AnimationBehavior + 'static>(behavior: B) -> SharedBehavior {
    Arc::new(behavior)
}
