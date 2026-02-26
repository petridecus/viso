//! Transition describes how to animate from the current state to a new target.

use std::sync::Arc;
use std::time::Duration;

use super::behaviors::{
    shared, AnimationBehavior, BackboneThenExpand, Cascade, CollapseExpand,
    SharedBehavior, SmoothInterpolation, Snap,
};

/// Describes how to animate from current state to a new target.
///
/// Consumers construct transitions via preset constructors:
/// [`snap()`](Self::snap), [`smooth()`](Self::smooth),
/// [`collapse_expand()`](Self::collapse_expand),
/// [`backbone_then_expand()`](Self::backbone_then_expand),
/// [`cascade()`](Self::cascade), or [`with_behavior()`](Self::with_behavior)
/// for custom strategies.
#[derive(Clone)]
pub struct Transition {
    /// The interpolation behavior (easing, phasing, duration).
    pub behavior: SharedBehavior,
    /// Whether the animator should allow backbone size changes.
    /// When false, size mismatches cause an instant snap.
    pub allows_size_change: bool,
    /// Whether to suppress initial sidechain GPU uploads.
    /// Used by multi-phase behaviors that hide sidechains in phase 1.
    pub suppress_initial_sidechains: bool,
}

impl Transition {
    /// Instant snap with no animation. Allows size changes.
    #[must_use]
    pub fn snap() -> Self {
        Self {
            behavior: Arc::new(Snap),
            allows_size_change: true,
            suppress_initial_sidechains: false,
        }
    }

    /// Standard smooth interpolation (300ms, cubic hermite ease-out).
    #[must_use]
    pub fn smooth() -> Self {
        Self {
            behavior: Arc::new(SmoothInterpolation::standard()),
            allows_size_change: false,
            suppress_initial_sidechains: false,
        }
    }

    /// Sidechains collapse to CA, backbone moves, sidechains expand.
    /// Used for mutations.
    #[must_use]
    pub fn collapse_expand(collapse: Duration, expand: Duration) -> Self {
        Self {
            behavior: shared(CollapseExpand::new(collapse, expand)),
            allows_size_change: true,
            suppress_initial_sidechains: true,
        }
    }

    /// Backbone animates first, then sidechains expand.
    /// Two-phase with configurable durations.
    #[must_use]
    pub fn backbone_then_expand(backbone: Duration, expand: Duration) -> Self {
        Self {
            behavior: shared(BackboneThenExpand::new(backbone, expand)),
            allows_size_change: false,
            suppress_initial_sidechains: true,
        }
    }

    /// Staggered per-residue delays for wave-like effects.
    #[must_use]
    pub fn cascade(base: Duration, delay_per_residue: Duration) -> Self {
        Self {
            behavior: shared(Cascade::new(base, delay_per_residue)),
            allows_size_change: false,
            suppress_initial_sidechains: false,
        }
    }

    /// Create a transition with a custom behavior.
    pub fn with_behavior(behavior: impl AnimationBehavior + 'static) -> Self {
        Self {
            behavior: Arc::new(behavior),
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
}

impl Default for Transition {
    fn default() -> Self {
        Self::smooth()
    }
}

impl std::fmt::Debug for Transition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transition")
            .field("behavior", &self.behavior.name())
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
        assert_eq!(t.behavior.name(), "snap");
        assert!(t.allows_size_change);
        assert!(!t.suppress_initial_sidechains);
    }

    #[test]
    fn test_smooth_transition() {
        let t = Transition::smooth();
        assert_eq!(t.behavior.name(), "smooth");
        assert!(!t.allows_size_change);
        assert!(!t.suppress_initial_sidechains);
    }

    #[test]
    fn test_default_is_smooth() {
        let t = Transition::default();
        assert_eq!(t.behavior.name(), "smooth");
    }

    #[test]
    fn test_builder_methods() {
        let t = Transition::smooth()
            .allowing_size_change()
            .suppressing_initial_sidechains();
        assert!(t.allows_size_change);
        assert!(t.suppress_initial_sidechains);
    }
}
