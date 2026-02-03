//! Instant snap behavior with no interpolation.

use std::time::Duration;

use super::state::ResidueVisualState;
use super::traits::{AnimationBehavior, PreemptionStrategy};

/// Instant snap to target with no interpolation.
///
/// Useful when you want immediate visual feedback or when
/// animation would be inappropriate (e.g., loading a new structure).
#[derive(Debug, Clone, Copy, Default)]
pub struct Snap;

impl AnimationBehavior for Snap {
    fn compute_state(
        &self,
        _t: f32,
        _start: &ResidueVisualState,
        end: &ResidueVisualState,
    ) -> ResidueVisualState {
        *end
    }

    fn duration(&self) -> Duration {
        Duration::ZERO
    }

    fn preemption(&self) -> PreemptionStrategy {
        PreemptionStrategy::Restart
    }

    fn name(&self) -> &'static str {
        "snap"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_snap_behavior() {
        let behavior = Snap;
        let a = ResidueVisualState::new(
            [Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)],
            [0.0, 0.0, 0.0, 0.0],
            1,
        );
        let b = ResidueVisualState::new(
            [Vec3::Y, Vec3::new(1.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.0)],
            [90.0, 0.0, 0.0, 0.0],
            1,
        );

        // Snap should always return end state regardless of t
        let at_start = behavior.compute_state(0.0, &a, &b);
        assert!((at_start.backbone[0] - b.backbone[0]).length() < 0.001);

        let at_mid = behavior.compute_state(0.5, &a, &b);
        assert!((at_mid.backbone[0] - b.backbone[0]).length() < 0.001);

        assert_eq!(behavior.duration(), Duration::ZERO);
    }
}
