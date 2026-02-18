//! Core trait for animation behaviors.

use std::{sync::Arc, time::Duration};

use super::{
    super::interpolation::InterpolationContext, state::ResidueVisualState,
};

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
/// Implementations control the interpolation curve, timing, and visual effects.
/// See [`SmoothInterpolation`], [`CollapseExpand`], [`Cascade`] for examples.
pub trait AnimationBehavior: Send + Sync {
    /// Eased time for a given raw progress. Override for custom easing.
    /// Default: linear (no easing).
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

    /// Interpolation context for this behavior at raw progress t.
    /// Complex behaviors (CollapseExpand) should override with phase-aware
    /// logic.
    fn compute_context(&self, raw_t: f32) -> InterpolationContext {
        InterpolationContext::simple(raw_t, self.eased_t(raw_t))
    }

    /// Whether sidechain atoms should be included in animation frames at this
    /// progress.
    ///
    /// Multi-phase behaviors (like BackboneThenExpand) can return false during
    /// phases where sidechains should be hidden, preventing visual
    /// artifacts when new sidechain atoms appear before the backbone has
    /// finished easing.
    ///
    /// Default: always include sidechains.
    fn should_include_sidechains(&self, _raw_t: f32) -> bool {
        true
    }
}

/// Type alias for shared behavior references.
pub type SharedBehavior = Arc<dyn AnimationBehavior>;

/// Create a shared behavior from any AnimationBehavior implementation.
pub fn shared<B: AnimationBehavior + 'static>(behavior: B) -> SharedBehavior {
    Arc::new(behavior)
}
