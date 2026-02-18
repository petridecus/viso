//! Sequenced animation: backbone lerp completes first, then sidechains expand.
//!
//! Designed for the diffusion finalize transition where streaming backbone-only
//! frames transition to full-atom results. The backbone smoothly lerps to its
//! final position, and only after it arrives do sidechains expand outward.

use std::time::Duration;

use crate::util::easing::EasingFunction;

use super::super::interpolation::InterpolationContext;
use super::state::ResidueVisualState;
use super::traits::{AnimationBehavior, PreemptionStrategy};

/// Two-phase sequenced animation: backbone lerp, then sidechain expand.
///
/// # Phases
///
/// 1. **Backbone lerp** (0.0 to backbone_fraction): Backbone positions interpolate
///    from start to end. Sidechains stay frozen at their collapsed (CA) positions.
/// 2. **Sidechain expand** (backbone_fraction to 1.0): Backbone is at its final
///    position. Sidechain chi angles interpolate from 0 to final values, and
///    sidechain atom positions expand from CA toward their final positions.
#[derive(Debug, Clone)]
pub struct BackboneThenExpand {
    pub backbone_duration: Duration,
    pub expand_duration: Duration,
    pub backbone_easing: EasingFunction,
    pub expand_easing: EasingFunction,
}

impl BackboneThenExpand {
    /// Create with custom durations for each phase.
    pub fn new(backbone_duration: Duration, expand_duration: Duration) -> Self {
        Self {
            backbone_duration,
            expand_duration,
            backbone_easing: EasingFunction::QuadraticOut,
            expand_easing: EasingFunction::QuadraticOut,
        }
    }

    /// Set custom easing functions.
    pub fn with_easing(
        mut self,
        backbone_easing: EasingFunction,
        expand_easing: EasingFunction,
    ) -> Self {
        self.backbone_easing = backbone_easing;
        self.expand_easing = expand_easing;
        self
    }

    /// Total duration of both phases.
    fn total_duration(&self) -> Duration {
        self.backbone_duration + self.expand_duration
    }

    /// Fraction of total time spent in backbone phase (0.0 to 1.0).
    fn backbone_fraction(&self) -> f32 {
        let total = self.total_duration().as_secs_f32();
        if total == 0.0 {
            0.5
        } else {
            self.backbone_duration.as_secs_f32() / total
        }
    }
}

impl Default for BackboneThenExpand {
    fn default() -> Self {
        Self::new(Duration::from_millis(200), Duration::from_millis(300))
    }
}

impl AnimationBehavior for BackboneThenExpand {
    fn compute_context(&self, raw_t: f32) -> InterpolationContext {
        let bb_frac = self.backbone_fraction();

        if raw_t < bb_frac {
            // Phase 1: Backbone lerp
            let phase_t = if bb_frac > 0.0 {
                raw_t / bb_frac
            } else {
                1.0
            };
            let phase_eased = self.backbone_easing.evaluate(phase_t);

            // eased_t goes 0→1 during this phase (backbone fully completes)
            let eased_t = phase_eased;

            InterpolationContext::with_phase(raw_t, eased_t, phase_t, phase_eased)
        } else {
            // Phase 2: Sidechain expand — backbone is done
            let phase_t = if bb_frac < 1.0 {
                (raw_t - bb_frac) / (1.0 - bb_frac)
            } else {
                1.0
            };
            let phase_eased = self.expand_easing.evaluate(phase_t);

            // eased_t stays at 1.0 — backbone is fully at end position
            let eased_t = 1.0;

            InterpolationContext::with_phase(raw_t, eased_t, phase_t, phase_eased)
        }
    }

    fn compute_state(
        &self,
        t: f32,
        start: &ResidueVisualState,
        end: &ResidueVisualState,
    ) -> ResidueVisualState {
        let ctx = self.compute_context(t);
        let bb_frac = self.backbone_fraction();

        // Backbone uses eased_t: goes 0→1 during phase 1, stays 1.0 during phase 2
        let backbone_t = ctx.eased_t;
        let backbone = [
            start.backbone[0] + (end.backbone[0] - start.backbone[0]) * backbone_t,
            start.backbone[1] + (end.backbone[1] - start.backbone[1]) * backbone_t,
            start.backbone[2] + (end.backbone[2] - start.backbone[2]) * backbone_t,
        ];

        let phase_eased = ctx.phase_eased_t.unwrap_or(1.0);

        let chis = if t < bb_frac {
            // Phase 1: Keep start chi values (sidechains frozen)
            start.chis
        } else {
            // Phase 2: Expand new sidechain from 0 → final chi values
            let mut chis = [0.0f32; 4];
            for i in 0..end.num_chis {
                chis[i] = end.chis[i] * phase_eased;
            }
            chis
        };

        let num_chis = if t < bb_frac {
            start.num_chis
        } else {
            end.num_chis
        };

        ResidueVisualState {
            backbone,
            chis,
            num_chis,
        }
    }

    fn duration(&self) -> Duration {
        self.total_duration()
    }

    fn preemption(&self) -> PreemptionStrategy {
        PreemptionStrategy::Restart
    }

    fn name(&self) -> &'static str {
        "backbone-then-expand"
    }

    fn should_include_sidechains(&self, raw_t: f32) -> bool {
        // Hide sidechains during phase 1 (backbone lerp) so they don't flash
        // at their final positions before the expand phase.
        raw_t >= self.backbone_fraction()
    }

    fn interpolate_position(
        &self,
        t: f32,
        _start: glam::Vec3,
        end: glam::Vec3,
        collapse_point: glam::Vec3,
    ) -> glam::Vec3 {
        let bb_frac = self.backbone_fraction();

        if t < bb_frac {
            // Phase 1: Pin sidechains to the moving backbone CA so they stay
            // hidden behind the ribbon. collapse_point is interpolated by
            // the caller using eased_t (0→1 during this phase), tracking
            // the backbone CA as it lerps to its final position.
            collapse_point
        } else {
            // Phase 2: Expand from collapse_point (CA at final position) → end
            let phase_t = if bb_frac < 1.0 {
                (t - bb_frac) / (1.0 - bb_frac)
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
    fn test_endpoints() {
        let behavior = BackboneThenExpand::default();
        let a = test_state_a();
        let b = test_state_b();

        // At t=0, should be at start
        let at_start = behavior.compute_state(0.0, &a, &b);
        assert!((at_start.backbone[0] - a.backbone[0]).length() < 0.001);
        assert!((at_start.chis[0] - a.chis[0]).abs() < 0.001);

        // At t=1, should be at end
        let at_end = behavior.compute_state(1.0, &a, &b);
        assert!((at_end.backbone[0] - b.backbone[0]).length() < 0.001);
        assert!((at_end.chis[0] - b.chis[0]).abs() < 0.001);
    }

    #[test]
    fn test_backbone_completes_before_expand() {
        let behavior = BackboneThenExpand::new(
            Duration::from_millis(200),
            Duration::from_millis(300),
        );
        let a = test_state_a();
        let b = test_state_b();

        let bb_frac = behavior.backbone_fraction(); // 0.4

        // Just before phase boundary: backbone should be nearly at end
        let near_boundary = behavior.compute_state(bb_frac - 0.01, &a, &b);
        assert!(
            (near_boundary.backbone[0] - b.backbone[0]).length() < 0.1,
            "Backbone should be nearly at end before phase boundary"
        );
        // But chis should still be at start
        assert!(
            (near_boundary.chis[0] - a.chis[0]).abs() < 1.0,
            "Chis should still be at start during backbone phase"
        );

        // At phase boundary: backbone at end, chis just starting to expand
        let at_boundary = behavior.compute_state(bb_frac, &a, &b);
        assert!(
            (at_boundary.backbone[0] - b.backbone[0]).length() < 0.001,
            "Backbone should be at end at phase boundary"
        );

        // During expand phase: backbone stays at end
        let mid_expand = behavior.compute_state(bb_frac + (1.0 - bb_frac) * 0.5, &a, &b);
        assert!(
            (mid_expand.backbone[0] - b.backbone[0]).length() < 0.001,
            "Backbone should stay at end during expand phase"
        );
        // Chis should be partially expanded
        assert!(
            mid_expand.chis[0] > 0.0 && mid_expand.chis[0] < b.chis[0],
            "Chis should be partially expanded, got {}",
            mid_expand.chis[0]
        );
    }

    #[test]
    fn test_sidechain_pinned_to_ca_during_backbone() {
        let behavior = BackboneThenExpand::new(
            Duration::from_millis(200),
            Duration::from_millis(300),
        );
        let start = Vec3::new(1.0, 0.0, 0.0);
        let end = Vec3::new(1.0, 5.0, 0.0);
        let collapse = Vec3::new(1.0, 2.0, 0.0);

        // During backbone phase: sidechain tracks the collapse point (moving CA)
        let pos = behavior.interpolate_position(0.2, start, end, collapse);
        assert!(
            (pos - collapse).length() < 0.001,
            "Sidechain should be pinned to collapse point (CA) during backbone phase"
        );
    }

    #[test]
    fn test_sidechain_expands_after_backbone() {
        let behavior = BackboneThenExpand::new(
            Duration::from_millis(200),
            Duration::from_millis(300),
        );
        let start = Vec3::new(1.0, 0.0, 0.0);
        let end = Vec3::new(1.0, 5.0, 0.0);
        let collapse = Vec3::new(1.0, 2.0, 0.0);

        let bb_frac = behavior.backbone_fraction();

        // At phase boundary: sidechain at collapse point
        let pos = behavior.interpolate_position(bb_frac, start, end, collapse);
        assert!(
            (pos - collapse).length() < 0.001,
            "Sidechain should be at collapse point at phase boundary"
        );

        // At end: sidechain at final position
        let pos = behavior.interpolate_position(1.0, start, end, collapse);
        assert!(
            (pos - end).length() < 0.001,
            "Sidechain should be at end at t=1"
        );
    }

    #[test]
    fn test_duration() {
        let behavior = BackboneThenExpand::new(
            Duration::from_millis(200),
            Duration::from_millis(300),
        );
        assert_eq!(behavior.duration(), Duration::from_millis(500));
    }
}
