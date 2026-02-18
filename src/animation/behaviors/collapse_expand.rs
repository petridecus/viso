//! Two-phase collapse/expand animation for mutations.

use std::time::Duration;

use crate::util::easing::EasingFunction;

use super::super::interpolation::InterpolationContext;
use super::state::ResidueVisualState;
use super::traits::{AnimationBehavior, PreemptionStrategy};

/// Two-phase animation: collapse to backbone, then expand to new state.
///
/// Useful for mutations where the old sidechain retracts and the new one grows.
///
/// # Phases
///
/// 1. **Collapse** (0.0 to midpoint): Old sidechain chi angles interpolate toward 0,
///    creating a visual "retraction" toward the backbone.
/// 2. **Expand** (midpoint to 1.0): New sidechain chi angles interpolate from 0
///    to their final values, creating a visual "growth" effect.
///
/// The backbone positions interpolate smoothly throughout both phases.
#[derive(Debug, Clone)]
pub struct CollapseExpand {
    pub collapse_duration: Duration,
    pub expand_duration: Duration,
    pub collapse_easing: EasingFunction,
    pub expand_easing: EasingFunction,
}

impl CollapseExpand {
    /// Create with custom durations.
    pub fn new(collapse_duration: Duration, expand_duration: Duration) -> Self {
        Self {
            collapse_duration,
            expand_duration,
            collapse_easing: EasingFunction::QuadraticIn,
            expand_easing: EasingFunction::QuadraticOut,
        }
    }

    /// Create with symmetric durations.
    pub fn symmetric(duration: Duration) -> Self {
        let half = duration / 2;
        Self::new(half, half)
    }

    /// Set custom easing functions.
    pub fn with_easing(
        mut self,
        collapse_easing: EasingFunction,
        expand_easing: EasingFunction,
    ) -> Self {
        self.collapse_easing = collapse_easing;
        self.expand_easing = expand_easing;
        self
    }

    /// Total duration of both phases.
    fn total_duration(&self) -> Duration {
        self.collapse_duration + self.expand_duration
    }

    /// Fraction of total time spent in collapse phase (0.0 to 1.0).
    fn collapse_fraction(&self) -> f32 {
        let total = self.total_duration().as_secs_f32();
        if total == 0.0 {
            0.5
        } else {
            self.collapse_duration.as_secs_f32() / total
        }
    }
}

impl Default for CollapseExpand {
    fn default() -> Self {
        Self::new(Duration::from_millis(150), Duration::from_millis(150))
    }
}

impl AnimationBehavior for CollapseExpand {
    fn compute_context(&self, raw_t: f32) -> InterpolationContext {
        let collapse_frac = self.collapse_fraction();

        if raw_t < collapse_frac {
            // Collapse phase
            let phase_t = if collapse_frac > 0.0 {
                raw_t / collapse_frac
            } else {
                1.0
            };
            let phase_eased = self.collapse_easing.evaluate(phase_t);

            // Global eased progress: during collapse, we go from 0 to collapse_frac
            // using the collapse easing curve
            let eased_t = phase_eased * collapse_frac;

            InterpolationContext::with_phase(
                raw_t,
                eased_t,
                phase_t,
                phase_eased,
            )
        } else {
            // Expand phase
            let phase_t = if collapse_frac < 1.0 {
                (raw_t - collapse_frac) / (1.0 - collapse_frac)
            } else {
                1.0
            };
            let phase_eased = self.expand_easing.evaluate(phase_t);

            // Global eased progress: during expand, we go from collapse_frac to 1.0
            // using the expand easing curve
            let eased_t = collapse_frac + phase_eased * (1.0 - collapse_frac);

            InterpolationContext::with_phase(
                raw_t,
                eased_t,
                phase_t,
                phase_eased,
            )
        }
    }

    fn compute_state(
        &self,
        t: f32,
        start: &ResidueVisualState,
        end: &ResidueVisualState,
    ) -> ResidueVisualState {
        let ctx = self.compute_context(t);
        let collapse_frac = self.collapse_fraction();

        let backbone_t = ctx.eased_t;
        let backbone = [
            start.backbone[0]
                + (end.backbone[0] - start.backbone[0]) * backbone_t,
            start.backbone[1]
                + (end.backbone[1] - start.backbone[1]) * backbone_t,
            start.backbone[2]
                + (end.backbone[2] - start.backbone[2]) * backbone_t,
        ];

        // Get phase-eased progress for chi interpolation
        let phase_eased = ctx.phase_eased_t.unwrap_or(1.0);

        let chis = if t < collapse_frac {
            // Phase 1: Collapse old sidechain toward backbone (chi -> 0)
            let mut chis = [0.0f32; 4];
            for (i, chi) in chis.iter_mut().enumerate().take(start.num_chis) {
                *chi = start.chis[i] * (1.0 - phase_eased);
            }
            chis
        } else {
            // Phase 2: Expand new sidechain from backbone (0 -> chi)
            let mut chis = [0.0f32; 4];
            for (i, chi) in chis.iter_mut().enumerate().take(end.num_chis) {
                *chi = end.chis[i] * phase_eased;
            }
            chis
        };

        ResidueVisualState {
            backbone,
            chis,
            num_chis: start.num_chis.max(end.num_chis),
        }
    }

    fn duration(&self) -> Duration {
        self.total_duration()
    }

    fn preemption(&self) -> PreemptionStrategy {
        PreemptionStrategy::Restart
    }

    fn name(&self) -> &'static str {
        "collapse-expand"
    }

    fn interpolate_position(
        &self,
        t: f32,
        start: glam::Vec3,
        end: glam::Vec3,
        collapse_point: glam::Vec3,
    ) -> glam::Vec3 {
        let collapse_frac = self.collapse_fraction();

        if t < collapse_frac {
            // Phase 1: Collapse from start toward collapse_point (backbone CA)
            let phase_t = if collapse_frac > 0.0 {
                t / collapse_frac
            } else {
                1.0
            };
            let eased_t = self.collapse_easing.evaluate(phase_t);
            start + (collapse_point - start) * eased_t
        } else {
            // Phase 2: Expand from collapse_point toward end
            let phase_t = if collapse_frac < 1.0 {
                (t - collapse_frac) / (1.0 - collapse_frac)
            } else {
                1.0
            };
            let eased_t = self.expand_easing.evaluate(phase_t);
            collapse_point + (end - collapse_point) * eased_t
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn test_state_a() -> ResidueVisualState {
        ResidueVisualState::new(
            [Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)],
            [60.0, 0.0, 0.0, 0.0],
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
    fn test_collapse_expand_endpoints() {
        let behavior = CollapseExpand::default();
        let a = test_state_a();
        let b = test_state_b();

        // At t=0, should be at start
        let at_start = behavior.compute_state(0.0, &a, &b);
        assert!((at_start.chis[0] - a.chis[0]).abs() < 0.001);
        assert!((at_start.backbone[0] - a.backbone[0]).length() < 0.001);

        // At t=1, should be at end
        let at_end = behavior.compute_state(1.0, &a, &b);
        assert!((at_end.chis[0] - b.chis[0]).abs() < 0.001);
        assert!((at_end.backbone[0] - b.backbone[0]).length() < 0.001);
    }

    #[test]
    fn test_collapse_expand_midpoint() {
        let behavior = CollapseExpand::new(
            Duration::from_millis(100),
            Duration::from_millis(100),
        );
        let a = test_state_a();
        let b = test_state_b();

        // At t=0.5 (transition point), chi should be collapsed (near 0)
        let at_mid = behavior.compute_state(0.5, &a, &b);
        assert!(
            at_mid.chis[0].abs() < 1.0,
            "At collapse point, chi should be near 0, got {}",
            at_mid.chis[0]
        );

        // Backbone should be halfway
        let expected_backbone_0 = (a.backbone[0] + b.backbone[0]) * 0.5;
        assert!((at_mid.backbone[0] - expected_backbone_0).length() < 0.001);
    }

    #[test]
    fn test_collapse_expand_duration() {
        let behavior = CollapseExpand::new(
            Duration::from_millis(100),
            Duration::from_millis(200),
        );
        assert_eq!(behavior.duration(), Duration::from_millis(300));
    }
}
