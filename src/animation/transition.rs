//! Transition describes how to animate from the current state to a new target.

use std::time::Duration;

use super::easing::EasingFunction;

/// A single phase of an animation sequence.
///
/// Each phase has its own easing, duration, and lerp range. The runner
/// evaluates phases sequentially — when one phase's duration expires,
/// the next begins.
#[derive(Debug, Clone)]
pub struct AnimationPhase {
    /// Easing curve for this phase.
    pub easing: EasingFunction,
    /// Duration of this phase.
    pub duration: Duration,
    /// Start of the global lerp range (0.0–1.0) this phase covers.
    pub lerp_start: f32,
    /// End of the global lerp range (0.0–1.0) this phase covers.
    pub lerp_end: f32,
    /// Whether sidechains should be visible during this phase.
    pub include_sidechains: bool,
}

/// Describes how to animate from current state to a new target.
///
/// Consumers construct transitions via preset constructors:
/// [`snap()`](Self::snap), [`smooth()`](Self::smooth),
/// [`collapse_expand()`](Self::collapse_expand),
/// [`backbone_then_expand()`](Self::backbone_then_expand),
/// or [`cascade()`](Self::cascade).
#[derive(Clone)]
pub struct Transition {
    /// Animation phases (single or multi-phase).
    pub(crate) phases: Vec<AnimationPhase>,
    /// Debug name.
    pub(crate) name: &'static str,
    /// Whether the animator should allow backbone size changes.
    /// When false, size mismatches cause an instant snap.
    pub allows_size_change: bool,
    /// Whether to suppress initial sidechain GPU uploads.
    /// Used by multi-phase behaviors that hide sidechains in phase 1.
    pub suppress_initial_sidechains: bool,
}

impl Transition {
    /// Total duration across all phases.
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.phases.iter().map(|p| p.duration).sum()
    }

    /// Instant snap with no animation. Allows size changes.
    #[must_use]
    pub fn snap() -> Self {
        Self {
            phases: vec![AnimationPhase {
                easing: EasingFunction::Linear,
                duration: Duration::ZERO,
                lerp_start: 0.0,
                lerp_end: 1.0,
                include_sidechains: true,
            }],

            name: "snap",
            allows_size_change: true,
            suppress_initial_sidechains: false,
        }
    }

    /// Standard smooth interpolation (300ms, cubic hermite ease-out).
    #[must_use]
    pub fn smooth() -> Self {
        Self {
            phases: vec![AnimationPhase {
                easing: EasingFunction::DEFAULT,
                duration: Duration::from_millis(300),
                lerp_start: 0.0,
                lerp_end: 1.0,
                include_sidechains: true,
            }],

            name: "smooth",
            allows_size_change: false,
            suppress_initial_sidechains: false,
        }
    }

    /// Sidechains collapse to CA, backbone moves, sidechains expand.
    /// Used for mutations.
    #[must_use]
    pub fn collapse_expand(collapse: Duration, expand: Duration) -> Self {
        let total_secs = (collapse + expand).as_secs_f32();
        let frac = if total_secs == 0.0 {
            0.5
        } else {
            collapse.as_secs_f32() / total_secs
        };
        Self {
            phases: vec![
                AnimationPhase {
                    easing: EasingFunction::QuadraticIn,
                    duration: collapse,
                    lerp_start: 0.0,
                    lerp_end: frac,
                    include_sidechains: true,
                },
                AnimationPhase {
                    easing: EasingFunction::QuadraticOut,
                    duration: expand,
                    lerp_start: frac,
                    lerp_end: 1.0,
                    include_sidechains: true,
                },
            ],

            name: "collapse-expand",
            allows_size_change: true,
            suppress_initial_sidechains: true,
        }
    }

    /// Backbone animates first, then sidechains expand.
    /// Two-phase with configurable durations.
    #[must_use]
    pub fn backbone_then_expand(backbone: Duration, expand: Duration) -> Self {
        Self {
            phases: vec![
                AnimationPhase {
                    easing: EasingFunction::QuadraticOut,
                    duration: backbone,
                    lerp_start: 0.0,
                    lerp_end: 1.0,
                    include_sidechains: false,
                },
                AnimationPhase {
                    easing: EasingFunction::Linear,
                    duration: expand,
                    lerp_start: 1.0,
                    lerp_end: 1.0,
                    include_sidechains: true,
                },
            ],

            name: "backbone-then-expand",
            allows_size_change: false,
            suppress_initial_sidechains: true,
        }
    }

    /// Staggered per-residue delays for wave-like effects.
    ///
    /// Staggered per-residue delays for wave-like effects.
    ///
    /// Per-residue staggering is not yet integrated into the runner.
    #[must_use]
    pub fn cascade(base: Duration, _delay_per_residue: Duration) -> Self {
        Self {
            phases: vec![AnimationPhase {
                easing: EasingFunction::QuadraticOut,
                duration: base,
                lerp_start: 0.0,
                lerp_end: 1.0,
                include_sidechains: true,
            }],
            name: "cascade",
            allows_size_change: false,
            suppress_initial_sidechains: false,
        }
    }

    /// Allow backbone size changes during animation.
    #[must_use]
    pub fn allowing_size_change(mut self) -> Self {
        self.allows_size_change = true;
        self
    }

    /// Suppress initial sidechain GPU uploads.
    #[must_use]
    pub fn suppressing_initial_sidechains(mut self) -> Self {
        self.suppress_initial_sidechains = true;
        self
    }

    /// Linear easing for testing.
    #[cfg(test)]
    pub(crate) fn linear(duration: Duration) -> Self {
        Self {
            phases: vec![AnimationPhase {
                easing: EasingFunction::Linear,
                duration,
                lerp_start: 0.0,
                lerp_end: 1.0,
                include_sidechains: true,
            }],

            name: "linear",
            allows_size_change: false,
            suppress_initial_sidechains: false,
        }
    }
}

impl Default for Transition {
    fn default() -> Self {
        Self::smooth()
    }
}

impl std::fmt::Debug for Transition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transition")
            .field("name", &self.name)
            .field("phases", &self.phases.len())
            .field("allows_size_change", &self.allows_size_change)
            .field(
                "suppress_initial_sidechains",
                &self.suppress_initial_sidechains,
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snap_transition() {
        let t = Transition::snap();
        assert_eq!(t.name, "snap");
        assert!(t.allows_size_change);
        assert!(!t.suppress_initial_sidechains);
        assert_eq!(t.total_duration(), Duration::ZERO);
    }

    #[test]
    fn test_smooth_transition() {
        let t = Transition::smooth();
        assert_eq!(t.name, "smooth");
        assert!(!t.allows_size_change);
        assert!(!t.suppress_initial_sidechains);
        assert_eq!(t.total_duration(), Duration::from_millis(300));
    }

    #[test]
    fn test_default_is_smooth() {
        let t = Transition::default();
        assert_eq!(t.name, "smooth");
    }

    #[test]
    fn test_builder_methods() {
        let t = Transition::smooth()
            .allowing_size_change()
            .suppressing_initial_sidechains();
        assert!(t.allows_size_change);
        assert!(t.suppress_initial_sidechains);
    }

    #[test]
    fn test_collapse_expand_phases() {
        let t = Transition::collapse_expand(
            Duration::from_millis(200),
            Duration::from_millis(300),
        );
        assert_eq!(t.phases.len(), 2);
        assert_eq!(t.total_duration(), Duration::from_millis(500));
        assert!((t.phases[0].lerp_end - 0.4).abs() < 0.01);
        assert!((t.phases[1].lerp_start - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_backbone_then_expand_sidechains() {
        let t = Transition::backbone_then_expand(
            Duration::from_millis(300),
            Duration::from_millis(200),
        );
        assert!(!t.phases[0].include_sidechains);
        assert!(t.phases[1].include_sidechains);
    }
}
