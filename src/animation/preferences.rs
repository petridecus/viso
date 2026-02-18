//! Animation preferences map actions to behaviors.
//!
//! Users can customize which animation behavior is used for each action type.

use super::behaviors::{
    shared, BackboneThenExpand, Cascade, CollapseExpand, SharedBehavior, SmoothInterpolation, Snap,
};

/// Actions that can trigger animations.
///
/// Each action can be configured to use a different animation behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnimationAction {
    /// Rosetta energy minimization (backbone + sidechains).
    Wiggle,
    /// Rosetta rotamer optimization (primarily sidechains).
    Shake,
    /// User-triggered residue mutation.
    Mutation,
    /// ML diffusion process (RFD3, SimpleFold).
    Diffusion,
    /// Final transition from diffusion streaming to full-atom result.
    DiffusionFinalize,
    /// Instant prediction result reveal.
    Reveal,
    /// Loading a new structure.
    Load,
}

impl AnimationAction {
    /// Whether this action supports animating across backbone size changes.
    ///
    /// Most actions (Wiggle, Shake, Mutation) operate on the same structure
    /// and always have the same residue count. But some transitions — like
    /// `DiffusionFinalize` — change the residue count (e.g., backbone-only
    /// streaming frames → full-atom result). For these, the animator will
    /// resize the current state to match the new target and animate the
    /// residues that overlap.
    pub fn allows_size_change(self) -> bool {
        matches!(
            self,
            AnimationAction::DiffusionFinalize | AnimationAction::Load
        )
    }
}

/// User-configurable mapping from actions to animation behaviors.
#[derive(Clone)]
pub struct AnimationPreferences {
    /// Animation for wiggle (Rosetta minimize).
    pub wiggle: SharedBehavior,
    /// Animation for shake (Rosetta rotamer packing).
    pub shake: SharedBehavior,
    /// Animation for residue mutations.
    pub mutation: SharedBehavior,
    /// Animation for diffusion intermediates (RFD3, SimpleFold).
    pub diffusion: SharedBehavior,
    /// Animation for final diffusion result (backbone lerp + sidechain expand).
    pub diffusion_finalize: SharedBehavior,
    /// Animation for revealing instant prediction results.
    pub reveal: SharedBehavior,
    /// Animation for loading new structures.
    pub load: SharedBehavior,
}

impl AnimationPreferences {
    /// Behavior for a given action.
    pub fn get(&self, action: AnimationAction) -> &SharedBehavior {
        match action {
            AnimationAction::Wiggle => &self.wiggle,
            AnimationAction::Shake => &self.shake,
            AnimationAction::Mutation => &self.mutation,
            AnimationAction::Diffusion => &self.diffusion,
            AnimationAction::DiffusionFinalize => &self.diffusion_finalize,
            AnimationAction::Reveal => &self.reveal,
            AnimationAction::Load => &self.load,
        }
    }

    /// Set the behavior for a given action.
    pub fn set(&mut self, action: AnimationAction, behavior: SharedBehavior) {
        match action {
            AnimationAction::Wiggle => self.wiggle = behavior,
            AnimationAction::Shake => self.shake = behavior,
            AnimationAction::Mutation => self.mutation = behavior,
            AnimationAction::Diffusion => self.diffusion = behavior,
            AnimationAction::DiffusionFinalize => self.diffusion_finalize = behavior,
            AnimationAction::Reveal => self.reveal = behavior,
            AnimationAction::Load => self.load = behavior,
        }
    }

    /// Create preferences with all animations disabled (instant snap).
    pub fn disabled() -> Self {
        let snap = shared(Snap);
        Self {
            wiggle: snap.clone(),
            shake: snap.clone(),
            mutation: snap.clone(),
            diffusion: snap.clone(),
            diffusion_finalize: snap.clone(),
            reveal: snap.clone(),
            load: snap,
        }
    }
}

impl Default for AnimationPreferences {
    fn default() -> Self {
        use std::time::Duration;

        Self {
            // Rosetta operations use standard smooth interpolation
            wiggle: shared(SmoothInterpolation::rosetta_default()),
            shake: shared(SmoothInterpolation::rosetta_default()),

            // Mutations use collapse/expand effect
            mutation: shared(CollapseExpand::default()),

            // Diffusion uses linear interpolation to not distort ML intermediates
            diffusion: shared(SmoothInterpolation::linear(Duration::from_millis(100))),

            // Final diffusion result: backbone lerps to end FIRST, then sidechains expand
            diffusion_finalize: shared(BackboneThenExpand::new(
                Duration::from_millis(400),
                Duration::from_millis(600),
            )),

            // Reveals use cascade for dramatic effect
            reveal: shared(Cascade::default()),

            // Loading snaps immediately (no animation)
            load: shared(Snap),
        }
    }
}

impl std::fmt::Debug for AnimationPreferences {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationPreferences")
            .field("wiggle", &self.wiggle.name())
            .field("shake", &self.shake.name())
            .field("mutation", &self.mutation.name())
            .field("diffusion", &self.diffusion.name())
            .field("diffusion_finalize", &self.diffusion_finalize.name())
            .field("reveal", &self.reveal.name())
            .field("load", &self.load.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_preferences() {
        let prefs = AnimationPreferences::default();

        assert_eq!(prefs.wiggle.name(), "smooth");
        assert_eq!(prefs.shake.name(), "smooth");
        assert_eq!(prefs.mutation.name(), "collapse-expand");
        assert_eq!(prefs.diffusion.name(), "smooth");
        assert_eq!(prefs.diffusion_finalize.name(), "backbone-then-expand");
        assert_eq!(prefs.reveal.name(), "cascade");
        assert_eq!(prefs.load.name(), "snap");
    }

    #[test]
    fn test_disabled_preferences() {
        let prefs = AnimationPreferences::disabled();

        assert_eq!(prefs.wiggle.name(), "snap");
        assert_eq!(prefs.shake.name(), "snap");
        assert_eq!(prefs.mutation.name(), "snap");
    }

    #[test]
    fn test_get_and_set() {
        let mut prefs = AnimationPreferences::default();

        // Override wiggle with snap
        prefs.set(AnimationAction::Wiggle, shared(Snap));
        assert_eq!(prefs.get(AnimationAction::Wiggle).name(), "snap");

        // Other actions unchanged
        assert_eq!(prefs.get(AnimationAction::Shake).name(), "smooth");
    }
}
