//! Structure animator composed from smaller components.
//!
//! - `StructureState`: Holds current and target visual states
//! - `AnimationRunner`: Executes a single animation
//! - `AnimationController`: Handles preemption and transition-based behavior

mod controller;
mod runner;
mod sidechain;
mod state;

use std::collections::HashMap;
use std::time::Instant;

pub use controller::AnimationController;
use glam::Vec3;
pub use runner::AnimationRunner;
use runner::ResidueAnimationData;
use sidechain::{AnimationContext, SidechainAnimationData, SidechainTarget};
pub use state::StructureState;

use super::interpolation::InterpolationContext;
use super::transition::Transition;
use crate::scene::EntityResidueRange;

/// Animation state for a single entity.
///
/// Pairs an [`AnimationRunner`] with the residue range it owns in the
/// flat structure array so each entity can animate independently.
struct EntityAnimationState {
    /// The active runner for this entity.
    runner: AnimationRunner,
    /// Residue range this entity owns in the flat array.
    range: EntityResidueRange,
}

/// Composes [`StructureState`], [`AnimationRunner`], and
/// [`AnimationController`] to animate backbone/sidechain transitions.
pub struct StructureAnimator {
    state: StructureState,
    runner: Option<AnimationRunner>,
    controller: AnimationController,
    /// Sidechain animation state (positions, residue indices, CA, snapped
    /// ranges, change tracking).
    sc: SidechainAnimationData,
    /// Current frame's animation progress (0.0 to 1.0), set by update().
    current_frame_progress: f32,
    /// Per-entity animation runners. When populated, these take precedence
    /// over the global `runner` for their entity's residues.
    entity_runners: HashMap<u32, EntityAnimationState>,
    /// Per-entity behavior overrides. When an entity has an entry here,
    /// `animate_entity` will use this transition instead of the default.
    entity_behaviors: HashMap<u32, Transition>,
}

impl StructureAnimator {
    /// Animator with default settings.
    pub fn new() -> Self {
        Self {
            state: StructureState::new(),
            runner: None,
            controller: AnimationController::new(),
            sc: SidechainAnimationData::new(),
            current_frame_progress: 1.0,
            entity_runners: HashMap::new(),
            entity_behaviors: HashMap::new(),
        }
    }

    /// Enable or disable animations.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn set_enabled(&mut self, enabled: bool) {
        self.controller.set_enabled(enabled);
        if !enabled {
            self.runner = None;
            self.entity_runners.clear();
        }
    }

    /// Whether animations are enabled.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn is_enabled(&self) -> bool {
        self.controller.is_enabled()
    }

    /// Whether any animation (global or per-entity) is currently in progress.
    pub fn is_animating(&self) -> bool {
        self.runner.is_some() || !self.entity_runners.is_empty()
    }

    /// Animation progress (0.0 to 1.0), computed in the last `update()` call.
    pub fn progress(&self) -> f32 {
        self.current_frame_progress
    }

    /// Build an `AnimationContext` from current animator state.
    fn animation_context(&self) -> AnimationContext<'_> {
        AnimationContext {
            raw_t: self.progress(),
            runner: self.runner.as_ref(),
            is_animating: self.is_animating(),
        }
    }

    /// Set a new target state, potentially triggering an animation.
    pub fn set_target(
        &mut self,
        backbone_chains: &[Vec<Vec3>],
        transition: &Transition,
    ) {
        let new_target = StructureState::from_backbone(backbone_chains);

        // Use the pending sidechain transition if sidechains changed,
        // otherwise use the provided transition
        let effective_transition = if self.sc.changed {
            self.sc
                .take_pending_transition()
                .unwrap_or_else(|| transition.clone())
        } else {
            transition.clone()
        };

        // Let controller decide about backbone animation
        let maybe_runner = self.controller.handle_new_target(
            &mut self.state,
            &new_target,
            self.runner.as_ref(),
            &effective_transition,
        );

        if let Some(runner) = maybe_runner {
            self.runner = Some(runner);
        } else if self.sc.changed {
            // Sidechain-only change: create a timing-only runner
            // (empty residue data — just provides progress/behavior for
            // sidechain interpolation)
            self.runner = Some(AnimationRunner::new(
                effective_transition.behavior.clone(),
                vec![],
            ));
        }

        // Reset sidechain change flag after processing
        self.sc.changed = false;

        self.state.set_target(new_target);
    }

    /// Update animations for the current frame.
    ///
    /// Returns `true` if animations are still active.
    pub fn update(&mut self, now: Instant) -> bool {
        let mut any_active = false;

        // 1. Poll the global runner (backward compat path).
        if let Some(ref runner) = self.runner {
            self.current_frame_progress = runner.progress(now);
            runner.apply_to_state(&mut self.state, self.current_frame_progress);

            if self.current_frame_progress >= 1.0 {
                self.current_frame_progress = 1.0;
                self.state.snap_to_target();
                self.runner = None;
            } else {
                any_active = true;
            }
        } else {
            self.current_frame_progress = 1.0;
        }

        // 2. Poll per-entity runners. Each runner interpolates only its own
        //    residue range and writes the result into `self.state`.
        let mut completed_entities = Vec::new();
        for (&entity_id, entity_state) in &self.entity_runners {
            let t = entity_state.runner.progress(now);

            if t >= 1.0 {
                Self::snap_range_to_target(
                    &mut self.state,
                    &entity_state.range,
                );
                completed_entities.push(entity_id);
            } else {
                entity_state.runner.apply_to_state(&mut self.state, t);
                any_active = true;
            }
        }

        // 3. Remove completed entity runners.
        for id in completed_entities {
            let _ = self.entity_runners.remove(&id);
        }

        any_active
    }

    /// Skip all animations (global and per-entity) to end state.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn skip(&mut self) {
        self.current_frame_progress = 1.0;
        self.state.snap_to_target();
        self.runner = None;
        // Snap each entity's residues to target before clearing.
        for entity_state in self.entity_runners.values() {
            Self::snap_range_to_target(&mut self.state, &entity_state.range);
        }
        self.entity_runners.clear();
    }

    /// Cancel all animations (global and per-entity), staying at current
    /// visual position.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn cancel(&mut self) {
        self.runner = None;
        self.entity_runners.clear();
    }

    /// Get the current visual backbone state as chains.
    pub fn get_backbone(&self) -> Vec<Vec<Vec3>> {
        self.state.to_backbone_chains()
    }

    /// Total number of residues in the structure.
    pub fn residue_count(&self) -> usize {
        self.state.residue_count()
    }

    /// Get the current structure state.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn state(&self) -> &StructureState {
        &self.state
    }

    /// Get the active animation runner, if any.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn runner(&self) -> Option<&AnimationRunner> {
        self.runner.as_ref()
    }

    /// Current interpolation context (unified across backbone, sidechains,
    /// bonds).
    pub fn interpolation_context(&self) -> InterpolationContext {
        let raw_t = self.progress();
        match self.runner.as_ref() {
            Some(runner) if raw_t < 1.0 => {
                runner.behavior().compute_context(raw_t)
            }
            _ => InterpolationContext::identity(),
        }
    }

    /// Set sidechain target positions with an explicit transition for
    /// sidechain-only animations.
    ///
    /// If sidechains change but backbone doesn't, this transition will be used
    /// to trigger an animation. Call this BEFORE `set_target()` for proper
    /// animation triggering.
    pub fn set_sidechain_target_with_transition(
        &mut self,
        positions: &[Vec3],
        residue_indices: &[u32],
        ca_positions: &[Vec3],
        transition: Option<&Transition>,
    ) {
        let anim = AnimationContext {
            raw_t: self.current_frame_progress,
            runner: self.runner.as_ref(),
            is_animating: self.is_animating(),
        };
        let target = SidechainTarget {
            positions,
            residue_indices,
            ca_positions,
        };
        self.sc.set_target(&target, transition, anim);
    }

    /// Snap an entity's residue range so `current = target`.
    fn snap_range_to_target(
        state: &mut StructureState,
        range: &EntityResidueRange,
    ) {
        let start = range.start as usize;
        let end = range.end() as usize;
        for r in start..end {
            if let Some(target) = state.get_target(r).copied() {
                state.set_current(r, target);
            }
        }
    }

    /// Get interpolated sidechain positions using the current animation
    /// behavior.
    ///
    /// This applies the same interpolation logic as backbone animation,
    /// including collapse/expand for mutations.
    pub fn get_sidechain_positions(&self) -> Vec<Vec3> {
        self.sc.get_positions(self.animation_context())
    }

    /// Snap sidechain and CA positions for entities NOT covered by any
    /// transition.
    ///
    /// For each entity range whose id is NOT in `active_entities`, sets
    /// `start = target` so those residues produce zero displacement during
    /// interpolation (they snap instantly).
    ///
    /// Call this AFTER `set_sidechain_target_with_transition` and `set_target`
    /// when per-entity transitions are in use.
    pub fn snap_entities_without_action(
        &mut self,
        entity_residue_ranges: &[EntityResidueRange],
        active_entities: &std::collections::HashSet<u32>,
    ) {
        self.sc
            .snap_non_targeted(entity_residue_ranges, active_entities);
    }

    /// Remove non-targeted entity residues from the active AnimationRunner.
    ///
    /// For entities NOT in `active_entities`, removes their residues from
    /// the runner's list so `apply_to_state` never touches them. Also snaps
    /// their backbone `current = target` so they show no visual motion.
    ///
    /// Call this AFTER `set_target` (which creates the runner) and AFTER
    /// `snap_entities_without_action` when per-entity transitions are in use.
    pub fn remove_non_targeted_from_runner(
        &mut self,
        entity_residue_ranges: &[EntityResidueRange],
        active_entities: &std::collections::HashSet<u32>,
    ) {
        // Collect residue ranges for non-targeted entities
        let snap_ranges: Vec<(usize, usize)> = entity_residue_ranges
            .iter()
            .filter(|r| !active_entities.contains(&r.entity_id))
            .map(|r| (r.start as usize, r.end() as usize))
            .collect();

        if snap_ranges.is_empty() {
            return;
        }

        // Snap backbone state so current = target for those residues
        for &(start, end) in &snap_ranges {
            for r in start..end {
                if let Some(target) = self.state.get_target(r).copied() {
                    self.state.set_current(r, target);
                }
            }
        }

        // Remove from runner so apply_to_state never overwrites them
        if let Some(ref mut runner) = self.runner {
            runner.remove_residue_ranges(&snap_ranges);
        }

        // Store snapped ranges so get_sidechain_positions() can skip them
        self.sc.set_snapped_ranges(snap_ranges);
    }

    /// Check if sidechain animation state is valid (has data).
    pub fn has_sidechain_data(&self) -> bool {
        self.sc.has_data()
    }

    /// Whether sidechains should be included in animation frames right now.
    ///
    /// Multi-phase behaviors (BackboneThenExpand) hide sidechains during
    /// the backbone-lerp phase so new atoms don't flash at their final
    /// positions.
    pub fn should_include_sidechains(&self) -> bool {
        self.runner.as_ref().is_none_or(|r| {
            r.behavior().should_include_sidechains(self.progress())
        })
    }

    // ── Per-entity animation API ──────────────────────────────────────

    /// Set the animation behavior that will be used for this entity's next
    /// update via [`animate_entity`](Self::animate_entity).
    ///
    /// Overrides the default smooth transition for the given entity.
    pub fn set_entity_behavior(
        &mut self,
        entity_id: u32,
        transition: Transition,
    ) {
        let _ = self.entity_behaviors.insert(entity_id, transition);
    }

    /// Clear a per-entity behavior override, reverting to the default
    /// (smooth) transition.
    pub fn clear_entity_behavior(&mut self, entity_id: u32) {
        let _ = self.entity_behaviors.remove(&entity_id);
    }

    /// Start a per-entity animation for the given entity.
    ///
    /// The entity's residues (identified by `range`) will be animated
    /// independently from the rest of the structure. If the entity already
    /// has an active runner, it is replaced (the current visual state is
    /// synced first so the new animation starts from the visible position).
    ///
    /// `new_backbone` is the full-structure backbone chains — only the
    /// residues within `range` are extracted for animation.
    #[allow(dead_code)] // API for future engine integration
    pub fn animate_entity(
        &mut self,
        entity_id: u32,
        range: &EntityResidueRange,
        new_backbone: &[Vec<Vec3>],
        transition: &Transition,
    ) {
        if !self.controller.is_enabled() {
            return;
        }

        let new_target = StructureState::from_backbone(new_backbone);
        let start_idx = range.start as usize;
        let end_idx = range.end() as usize;

        // If this entity is already animating, sync its residues to the
        // current interpolated position so the new animation picks up
        // smoothly.
        if let Some(prev) = self.entity_runners.get(&entity_id) {
            let t = prev.runner.progress(Instant::now());
            prev.runner.apply_to_state(&mut self.state, t);
        }

        // Build per-residue animation data for this entity's range.
        let mut residue_data = Vec::new();
        for global_idx in start_idx..end_idx {
            let start = match self.state.get_current(global_idx) {
                Some(s) => *s,
                None => continue,
            };
            let target = match new_target.get_target(global_idx) {
                Some(t) => *t,
                None => continue,
            };

            if StructureState::states_differ(&start, &target) {
                residue_data.push(ResidueAnimationData {
                    residue_idx: global_idx,
                    start,
                    target,
                });
            }
        }

        // Update the structure's target state for this entity's residues.
        for global_idx in start_idx..end_idx {
            if let Some(target) = new_target.get_target(global_idx).copied() {
                self.state.set_target_residue(global_idx, target);
            }
        }

        // Only create a runner if there are residues that actually changed.
        if residue_data.is_empty() {
            // No visual change — snap and remove any stale runner.
            Self::snap_range_to_target(&mut self.state, range);
            let _ = self.entity_runners.remove(&entity_id);
            return;
        }

        let runner =
            AnimationRunner::new(transition.behavior.clone(), residue_data);
        let _ = self.entity_runners.insert(
            entity_id,
            EntityAnimationState {
                runner,
                range: *range,
            },
        );
    }

    /// Whether a specific entity is currently animating.
    #[allow(dead_code)] // API for future engine integration
    pub fn is_entity_animating(&self, entity_id: u32) -> bool {
        self.entity_runners.contains_key(&entity_id)
    }

    /// Whether any animation (global or per-entity) is active.
    ///
    /// Alias for [`is_animating`](Self::is_animating) — provided for
    /// symmetry with [`is_entity_animating`](Self::is_entity_animating).
    #[allow(dead_code)] // API for future engine integration
    pub fn is_any_animating(&self) -> bool {
        self.is_animating()
    }

    // ── End per-entity animation API ────────────────────────────────

    /// Get the CA position of a residue by index.
    /// Returns the interpolated position during animation.
    pub fn get_ca_position(&self, residue_idx: usize) -> Option<Vec3> {
        self.sc.get_ca(residue_idx, self.animation_context())
    }
}

impl Default for StructureAnimator {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for StructureAnimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructureAnimator")
            .field("residue_count", &self.state.residue_count())
            .field("is_animating", &self.is_animating())
            .field("entity_runners", &self.entity_runners.len())
            .field("controller", &self.controller)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn make_backbone(y: f32) -> Vec<Vec<Vec3>> {
        vec![vec![
            Vec3::new(0.0, y, 0.0),
            Vec3::new(1.0, y, 0.0),
            Vec3::new(2.0, y, 0.0),
            Vec3::new(3.0, y, 0.0),
            Vec3::new(4.0, y, 0.0),
            Vec3::new(5.0, y, 0.0),
        ]]
    }

    #[test]
    fn test_animator_initial_state() {
        let animator = StructureAnimator::new();
        assert!(animator.is_enabled());
        assert!(!animator.is_animating());
        assert_eq!(animator.residue_count(), 0);
    }

    #[test]
    fn test_animator_first_target_snaps() {
        let mut animator = StructureAnimator::new();
        animator.set_target(&make_backbone(0.0), &Transition::snap());

        assert!(!animator.is_animating());
        assert_eq!(animator.residue_count(), 2);
    }

    #[test]
    fn test_animator_animates_on_change() {
        let mut animator = StructureAnimator::new();
        animator.set_target(&make_backbone(0.0), &Transition::smooth());
        animator.set_target(&make_backbone(10.0), &Transition::smooth());

        assert!(animator.is_animating());
    }

    #[test]
    fn test_animator_skip() {
        let mut animator = StructureAnimator::new();
        animator.set_target(&make_backbone(0.0), &Transition::smooth());
        animator.set_target(&make_backbone(10.0), &Transition::smooth());

        animator.skip();

        assert!(!animator.is_animating());
        let backbone = animator.get_backbone();
        assert!((backbone[0][0].y - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_animator_completes() {
        let mut animator = StructureAnimator::new();
        animator.set_target(&make_backbone(0.0), &Transition::snap());
        animator.set_target(&make_backbone(10.0), &Transition::smooth());

        let future = Instant::now() + Duration::from_secs(1);
        let still_animating = animator.update(future);

        assert!(!still_animating);
        assert!(!animator.is_animating());
    }

    #[test]
    fn test_animator_sidechain_only_change_triggers_animation() {
        let mut animator = StructureAnimator::new();

        // Set initial backbone
        animator.set_target(&make_backbone(5.0), &Transition::snap());
        assert!(!animator.is_animating(), "Initial backbone should snap");

        // Set initial sidechains
        let sidechain_pos =
            vec![Vec3::new(1.0, 5.0, 1.0), Vec3::new(2.0, 5.0, 1.0)];
        let residue_indices = vec![0, 1];
        let ca_positions =
            vec![Vec3::new(1.0, 5.0, 0.0), Vec3::new(4.0, 5.0, 0.0)];
        animator.set_sidechain_target_with_transition(
            &sidechain_pos,
            &residue_indices,
            &ca_positions,
            Some(&Transition::smooth()),
        );

        // Set same backbone again - should trigger animation due to sidechain
        // change
        animator.set_target(&make_backbone(5.0), &Transition::snap());

        // Now change sidechains with same backbone
        let new_sidechain_pos =
            vec![Vec3::new(1.0, 5.0, 2.0), Vec3::new(2.0, 5.0, 2.0)];
        animator.set_sidechain_target_with_transition(
            &new_sidechain_pos,
            &residue_indices,
            &ca_positions,
            Some(&Transition::smooth()),
        );

        // Set same backbone - should animate because sidechains changed
        animator.set_target(&make_backbone(5.0), &Transition::snap());
        assert!(
            animator.is_animating(),
            "Sidechain-only change should trigger animation"
        );
    }
}
