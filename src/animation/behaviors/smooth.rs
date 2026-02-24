//! Smooth interpolation behavior with configurable easing.

use std::time::Duration;

use super::{
    state::ResidueVisualState,
    traits::{AnimationBehavior, PreemptionStrategy},
};
use crate::util::easing::EasingFunction;

/// Smooth interpolation with configurable easing.
///
/// This is the default behavior for most animations.
/// Uses 300ms cubic hermite ease-out by default.
#[derive(Debug, Clone)]
pub struct SmoothInterpolation {
    /// Total animation duration.
    pub duration: Duration,
    /// Easing curve for interpolation.
    pub easing: EasingFunction,
    /// Strategy when a new target arrives mid-animation.
    pub preemption: PreemptionStrategy,
}

impl SmoothInterpolation {
    /// Create with custom parameters.
    pub fn new(duration: Duration, easing: EasingFunction) -> Self {
        Self {
            duration,
            easing,
            preemption: PreemptionStrategy::Restart,
        }
    }

    /// Standard defaults: 300ms, cubic hermite ease-out.
    pub fn standard() -> Self {
        Self {
            duration: Duration::from_millis(300),
            easing: EasingFunction::DEFAULT,
            preemption: PreemptionStrategy::Restart,
        }
    }

    /// Fast interpolation (100ms, quadratic ease-out).
    pub fn fast() -> Self {
        Self {
            duration: Duration::from_millis(100),
            easing: EasingFunction::QuadraticOut,
            preemption: PreemptionStrategy::Restart,
        }
    }

    /// Linear interpolation (no easing distortion).
    pub fn linear(duration: Duration) -> Self {
        Self {
            duration,
            easing: EasingFunction::Linear,
            preemption: PreemptionStrategy::Restart,
        }
    }

    /// Set preemption strategy.
    pub fn with_preemption(mut self, strategy: PreemptionStrategy) -> Self {
        self.preemption = strategy;
        self
    }
}

impl Default for SmoothInterpolation {
    fn default() -> Self {
        Self::standard()
    }
}

impl AnimationBehavior for SmoothInterpolation {
    fn eased_t(&self, t: f32) -> f32 {
        self.easing.evaluate(t)
    }

    fn compute_state(
        &self,
        t: f32,
        start: &ResidueVisualState,
        end: &ResidueVisualState,
    ) -> ResidueVisualState {
        let eased_t = self.eased_t(t);
        start.lerp(end, eased_t)
    }

    fn duration(&self) -> Duration {
        self.duration
    }

    fn preemption(&self) -> PreemptionStrategy {
        self.preemption
    }

    fn name(&self) -> &'static str {
        "smooth"
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    fn test_state_a() -> ResidueVisualState {
        ResidueVisualState::new(
            [Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)],
            [0.0, 0.0, 0.0, 0.0],
            1,
        )
    }

    fn test_state_b() -> ResidueVisualState {
        ResidueVisualState::new(
            [Vec3::Y, Vec3::new(1.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.0)],
            [90.0, 0.0, 0.0, 0.0],
            1,
        )
    }

    #[test]
    fn test_smooth_interpolation_endpoints() {
        let behavior = SmoothInterpolation::linear(Duration::from_millis(100));
        let a = test_state_a();
        let b = test_state_b();

        let at_start = behavior.compute_state(0.0, &a, &b);
        assert!((at_start.backbone[0] - a.backbone[0]).length() < 0.001);

        let at_end = behavior.compute_state(1.0, &a, &b);
        assert!((at_end.backbone[0] - b.backbone[0]).length() < 0.001);
    }

    #[test]
    fn test_smooth_interpolation_midpoint() {
        let behavior = SmoothInterpolation::linear(Duration::from_millis(100));
        let a = test_state_a();
        let b = test_state_b();

        let mid = behavior.compute_state(0.5, &a, &b);
        let expected_n = Vec3::new(0.0, 0.5, 0.0);
        assert!((mid.backbone[0] - expected_n).length() < 0.001);
    }

    #[test]
    fn test_standard_values() {
        let behavior = SmoothInterpolation::standard();
        assert_eq!(behavior.duration, Duration::from_millis(300));
        assert_eq!(behavior.easing, EasingFunction::DEFAULT);
    }
}
