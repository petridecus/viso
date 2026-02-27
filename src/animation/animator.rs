//! Structure animator for per-entity backbone/sidechain transitions.

use std::collections::HashMap;
use std::time::Instant;

use glam::Vec3;

use super::runner::{
    AnimationRunner, ResidueAnimationData, ResidueVisualState,
    SidechainAnimPositions,
};
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
    /// Current interpolated sidechain positions (computed each frame in
    /// `update()`). Pre-computed so queries can read without recomputing.
    sidechain_current: Option<Vec<Vec3>>,
}

/// Manages per-entity animation runners that interpolate backbone and
/// sidechain transitions independently.
pub struct StructureAnimator {
    state: StructureState,
    enabled: bool,
    /// Per-entity animation runners. Each entity interpolates its own
    /// residue range with its own behavior and timing.
    entity_runners: HashMap<u32, EntityAnimationState>,
}

impl StructureAnimator {
    /// Animator with default settings.
    pub fn new() -> Self {
        Self {
            state: StructureState::new(),
            enabled: true,
            entity_runners: HashMap::new(),
        }
    }

    /// Enable or disable animations.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.entity_runners.clear();
        }
    }

    /// Whether animations are enabled.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Whether any per-entity animation is currently in progress.
    pub fn is_animating(&self) -> bool {
        !self.entity_runners.is_empty()
    }

    /// Update animations for the current frame.
    ///
    /// Interpolates both backbone and sidechain positions using the same
    /// `eased_t` from each entity's behavior. Returns `true` if animations
    /// are still active.
    pub fn update(&mut self, now: Instant) -> bool {
        let mut any_active = false;
        let mut completed_entities = Vec::new();

        for (&entity_id, entity_state) in &mut self.entity_runners {
            let t = entity_state.runner.progress(now);

            if t >= 1.0 {
                Self::snap_range_to_target(
                    &mut self.state,
                    &entity_state.range,
                );
                completed_entities.push(entity_id);
            } else {
                for (idx, visual) in entity_state.runner.interpolate_residues(t)
                {
                    self.state.set_current(idx, visual);
                }
                entity_state.sidechain_current =
                    entity_state.runner.interpolate_sidechain(t);
                any_active = true;
            }
        }

        for id in completed_entities {
            let _ = self.entity_runners.remove(&id);
        }

        any_active
    }

    /// Skip all per-entity animations to end state.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn skip(&mut self) {
        for entity_state in self.entity_runners.values() {
            Self::snap_range_to_target(&mut self.state, &entity_state.range);
        }
        self.entity_runners.clear();
    }

    /// Cancel all per-entity animations, staying at current visual
    /// position.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn cancel(&mut self) {
        self.entity_runners.clear();
    }

    /// Get the current visual backbone state as chains.
    pub fn get_backbone(&self) -> Vec<Vec<Vec3>> {
        self.state.to_backbone_chains()
    }

    /// Total number of residues in the structure.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn residue_count(&self) -> usize {
        self.state.residue_count()
    }

    /// Get the CA position of a residue from current visual state.
    pub fn get_ca_position(&self, residue_idx: usize) -> Option<Vec3> {
        Some(self.state.get_current(residue_idx)?.backbone[1])
    }

    /// Whether sidechains should be included in animation frames.
    ///
    /// Multi-phase behaviors (BackboneThenExpand) hide sidechains during
    /// the backbone-lerp phase so new atoms don't flash at their final
    /// positions.
    pub fn should_include_sidechains(&self) -> bool {
        let now = Instant::now();
        for es in self.entity_runners.values() {
            let t = es.runner.progress(now);
            if !es.runner.should_include_sidechains(t) {
                return false;
            }
        }
        true
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

    // ── Per-entity animation API ──────────────────────────────────────

    /// Start a per-entity animation for the given entity.
    ///
    /// The entity's residues (identified by `range`) will be animated
    /// independently from the rest of the structure. If the entity already
    /// has an active runner, it is replaced (the current visual state is
    /// synced first so the new animation starts from the visible position).
    ///
    /// `new_backbone` is the full-structure backbone chains — only the
    /// residues within `range` are extracted for animation.
    ///
    /// Optional `sidechain` positions are lerped with the same `eased_t`
    /// as backbone.
    pub fn animate_entity(
        &mut self,
        range: &EntityResidueRange,
        new_backbone: &[Vec<Vec3>],
        transition: &Transition,
        sidechain: Option<SidechainAnimPositions>,
    ) {
        if !self.enabled {
            return;
        }

        let entity_id = range.entity_id;
        let prev_sidechain = self.sync_preempted_entity(entity_id);

        let new_target = StructureState::from_backbone(new_backbone);
        let start_idx = range.start as usize;
        let end_idx = range.end() as usize;

        // Build per-residue animation data for this entity's range.
        let residue_data =
            self.build_residue_data(start_idx, end_idx, &new_target);

        // Update the structure's target state for this entity's residues.
        for global_idx in start_idx..end_idx {
            if let Some(target) = new_target.get_target(global_idx).copied() {
                self.state.set_target_residue(global_idx, target);
            }
        }

        // Check if sidechains changed (even if backbone didn't)
        let sc_changed = sidechain
            .as_ref()
            .is_some_and(|sc| !sc.start.is_empty() || !sc.target.is_empty());

        // Only create a runner if there are residues or sidechains that
        // actually changed.
        if residue_data.is_empty() && !sc_changed {
            // No visual change — snap and remove any stale runner.
            Self::snap_range_to_target(&mut self.state, range);
            let _ = self.entity_runners.remove(&entity_id);
            return;
        }

        // If preempting a previous animation, use its current sidechain
        // positions as the new start (smooth handoff). Only valid when
        // atom counts match.
        let sidechain = match (sidechain, prev_sidechain) {
            (Some(mut sc), Some(prev_positions))
                if sc.start.len() == prev_positions.len() =>
            {
                sc.start = prev_positions;
                Some(sc)
            }
            (sc, _) => sc,
        };

        let runner = AnimationRunner::new(transition, residue_data, sidechain);

        // Compute initial sidechain positions so queries before the
        // first update() call return correct values.
        let initial_t = runner.progress(Instant::now());
        let initial_sc = runner.interpolate_sidechain(initial_t);

        let _ = self.entity_runners.insert(
            entity_id,
            EntityAnimationState {
                runner,
                range: *range,
                sidechain_current: initial_sc,
            },
        );
    }

    /// If the entity is already animating, sync its residues to the
    /// current interpolated position and return its sidechain state.
    fn sync_preempted_entity(&mut self, entity_id: u32) -> Option<Vec<Vec3>> {
        let prev = self.entity_runners.get(&entity_id)?;
        let t = prev.runner.progress(Instant::now());
        for (idx, visual) in prev.runner.interpolate_residues(t) {
            self.state.set_current(idx, visual);
        }
        prev.sidechain_current.clone()
    }

    /// Build per-residue animation data comparing current vs target.
    fn build_residue_data(
        &self,
        start_idx: usize,
        end_idx: usize,
        new_target: &StructureState,
    ) -> Vec<ResidueAnimationData> {
        let mut data = Vec::new();
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
                data.push(ResidueAnimationData {
                    residue_idx: global_idx,
                    start,
                    target,
                });
            }
        }
        data
    }

    /// Get aggregated interpolated sidechain positions across all
    /// per-entity runners.
    ///
    /// Reads pre-computed positions from the last `update()` call (or
    /// initial values set by `animate_entity()`). Returns `None` if no
    /// entity runner has sidechain data.
    pub fn get_aggregated_sidechain_positions(&self) -> Option<Vec<Vec3>> {
        let has_any = self
            .entity_runners
            .values()
            .any(|es| es.sidechain_current.is_some());

        if !has_any {
            return None;
        }

        let mut all_positions = Vec::new();
        for entity_state in self.entity_runners.values() {
            if let Some(ref positions) = entity_state.sidechain_current {
                all_positions.extend_from_slice(positions);
            }
        }

        Some(all_positions)
    }

    /// Whether a specific entity is currently animating.
    #[allow(dead_code)] // API for future engine integration
    pub fn is_entity_animating(&self, entity_id: u32) -> bool {
        self.entity_runners.contains_key(&entity_id)
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
            .field("enabled", &self.enabled)
            .finish_non_exhaustive()
    }
}

// ── StructureState ──────────────────────────────────────────────────────────

/// Holds the current and target visual state for a structure.
///
/// Responsible for:
/// - Converting between backbone chain format and per-residue states
/// - Tracking current visual state vs target state
/// - Detecting when states differ
#[derive(Debug, Clone)]
pub(crate) struct StructureState {
    /// Current visual state per residue (what we render).
    current: Vec<ResidueVisualState>,
    /// Target state per residue (animation end state).
    target: Vec<ResidueVisualState>,
    /// Number of residues per chain (for preserving chain boundaries).
    chain_lengths: Vec<usize>,
}

impl StructureState {
    /// Create empty state.
    pub fn new() -> Self {
        Self {
            current: Vec::new(),
            target: Vec::new(),
            chain_lengths: Vec::new(),
        }
    }

    /// Create state from backbone chains.
    ///
    /// Backbone chains are organized as `[chain][atoms]` where atoms are
    /// `[N, CA, C, N, CA, C, ...]` (3 atoms per residue).
    pub fn from_backbone(backbone_chains: &[Vec<Vec3>]) -> Self {
        let states = Self::backbone_to_states(backbone_chains);
        // Track how many residues in each chain
        let chain_lengths: Vec<usize> = backbone_chains
            .iter()
            .map(|chain| chain.len() / 3)
            .collect();
        Self {
            current: states.clone(),
            target: states,
            chain_lengths,
        }
    }

    /// Get current visual state for a residue.
    pub fn get_current(&self, idx: usize) -> Option<&ResidueVisualState> {
        self.current.get(idx)
    }

    /// Get target state for a residue.
    pub fn get_target(&self, idx: usize) -> Option<&ResidueVisualState> {
        self.target.get(idx)
    }

    /// Number of residues.
    pub fn residue_count(&self) -> usize {
        self.current.len()
    }

    /// Set current state for a specific residue.
    pub fn set_current(&mut self, idx: usize, state: ResidueVisualState) {
        if let Some(current) = self.current.get_mut(idx) {
            *current = state;
        }
    }

    /// Set target state for a specific residue.
    ///
    /// Used by per-entity animation to update only one entity's residues
    /// without replacing the entire target array.
    pub fn set_target_residue(
        &mut self,
        idx: usize,
        state: ResidueVisualState,
    ) {
        if let Some(target) = self.target.get_mut(idx) {
            *target = state;
        }
    }

    /// Convert current state back to backbone chains format.
    pub fn to_backbone_chains(&self) -> Vec<Vec<Vec3>> {
        if self.current.is_empty() {
            return Vec::new();
        }

        // Use chain_lengths to preserve chain boundaries
        let mut chains = Vec::new();
        let mut residue_idx = 0;

        for &chain_len in &self.chain_lengths {
            let mut chain_positions = Vec::with_capacity(chain_len * 3);
            for _ in 0..chain_len {
                if let Some(state) = self.current.get(residue_idx) {
                    chain_positions.extend(state.backbone.iter().copied());
                }
                residue_idx += 1;
            }
            if !chain_positions.is_empty() {
                chains.push(chain_positions);
            }
        }

        // Fallback: if chain_lengths is empty but we have residues, return as
        // single chain
        if chains.is_empty() && !self.current.is_empty() {
            let positions: Vec<Vec3> = self
                .current
                .iter()
                .flat_map(|s| s.backbone.iter().copied())
                .collect();
            chains.push(positions);
        }

        chains
    }

    /// Whether two states differ above a small epsilon threshold.
    pub fn states_differ(
        a: &ResidueVisualState,
        b: &ResidueVisualState,
    ) -> bool {
        const EPSILON: f32 = 0.0001;

        for i in 0..3 {
            if (a.backbone[i] - b.backbone[i]).length() > EPSILON {
                return true;
            }
        }

        false
    }

    /// Convert backbone chains to per-residue visual states.
    fn backbone_to_states(
        backbone_chains: &[Vec<Vec3>],
    ) -> Vec<ResidueVisualState> {
        let mut states = Vec::new();

        for chain in backbone_chains {
            for chunk in chain.chunks(3) {
                if chunk.len() == 3 {
                    let backbone = [chunk[0], chunk[1], chunk[2]];
                    states.push(ResidueVisualState::new(backbone));
                }
            }
        }

        states
    }
}

impl Default for StructureState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_animator_initial_state() {
        let animator = StructureAnimator::new();
        assert!(animator.is_enabled());
        assert!(!animator.is_animating());
        assert_eq!(animator.residue_count(), 0);
    }

    // ── StructureState tests ────────────────────────────────────────────

    fn make_backbone(y: f32, num_residues: usize) -> Vec<Vec<Vec3>> {
        let atoms: Vec<Vec3> = (0..num_residues)
            .flat_map(|r| {
                let x = r as f32 * 3.0;
                vec![
                    Vec3::new(x, y, 0.0),
                    Vec3::new(x + 1.0, y, 0.0),
                    Vec3::new(x + 2.0, y, 0.0),
                ]
            })
            .collect();
        vec![atoms]
    }

    #[test]
    fn test_from_backbone() {
        let backbone = make_backbone(0.0, 3);
        let state = StructureState::from_backbone(&backbone);

        assert_eq!(state.residue_count(), 3);
    }

    #[test]
    fn test_to_backbone_roundtrip() {
        let original = make_backbone(5.0, 4);
        let state = StructureState::from_backbone(&original);
        let recovered = state.to_backbone_chains();

        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].len(), 12); // 4 residues * 3 atoms

        for (orig, rec) in original[0].iter().zip(recovered[0].iter()) {
            assert!((orig - rec).length() < 0.001);
        }
    }
}
