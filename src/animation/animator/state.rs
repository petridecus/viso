//! Structure state management for animation.

use glam::Vec3;

use crate::animation::behaviors::ResidueVisualState;

/// Holds the current and target visual state for a structure.
///
/// Responsible for:
/// - Converting between backbone chain format and per-residue states
/// - Tracking current visual state vs target state
/// - Detecting when states differ
#[derive(Debug, Clone)]
pub struct StructureState {
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

    /// Get mutable current visual state for a residue.
    pub fn get_current_mut(
        &mut self,
        idx: usize,
    ) -> Option<&mut ResidueVisualState> {
        self.current.get_mut(idx)
    }

    /// Get target state for a residue.
    pub fn get_target(&self, idx: usize) -> Option<&ResidueVisualState> {
        self.target.get(idx)
    }

    /// Get all current states.
    pub fn current_states(&self) -> &[ResidueVisualState] {
        &self.current
    }

    /// Get all target states.
    pub fn target_states(&self) -> &[ResidueVisualState] {
        &self.target
    }

    /// Number of residues.
    pub fn residue_count(&self) -> usize {
        self.current.len()
    }

    /// Whether no residues are stored.
    pub fn is_empty(&self) -> bool {
        self.current.is_empty()
    }

    /// Set new target state.
    pub fn set_target(&mut self, new_target: StructureState) {
        self.target = new_target.target;
        self.chain_lengths = new_target.chain_lengths;
    }

    /// Set current state to match target (snap to end).
    pub fn snap_to_target(&mut self) {
        self.current = self.target.clone();
    }

    /// Set current state for a specific residue.
    pub fn set_current(&mut self, idx: usize, state: ResidueVisualState) {
        if let Some(current) = self.current.get_mut(idx) {
            *current = state;
        }
    }

    /// Whether this state has a different residue count than `other`.
    pub fn size_differs(&self, other: &StructureState) -> bool {
        self.current.len() != other.current.len()
    }

    /// Whether any target residue state differs from `other`.
    pub fn target_differs(&self, other: &StructureState) -> bool {
        if self.target.len() != other.target.len() {
            return true;
        }

        self.target
            .iter()
            .zip(other.target.iter())
            .any(|(a, b)| Self::states_differ(a, b))
    }

    /// Find residues that differ between current and a new target.
    ///
    /// Returns indices of residues that need animation.
    pub fn differing_residues(
        &self,
        new_target: &StructureState,
    ) -> Vec<usize> {
        self.current
            .iter()
            .zip(new_target.target.iter())
            .enumerate()
            .filter_map(|(idx, (current, target))| {
                if Self::states_differ(current, target) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
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

        // Check backbone positions
        for i in 0..3 {
            if (a.backbone[i] - b.backbone[i]).length() > EPSILON {
                return true;
            }
        }

        // Check chi angles
        let num_chis = a.num_chis.max(b.num_chis);
        for i in 0..num_chis {
            if (a.chis[i] - b.chis[i]).abs() > EPSILON {
                return true;
            }
        }

        false
    }

    /// Resize current state to match a new target's dimensions.
    ///
    /// Preserves existing residue positions where possible:
    /// - If new target is larger: existing residues keep their current
    ///   positions, new residues start at their target positions (no animation
    ///   for those).
    /// - If new target is smaller: truncate to the new size.
    ///
    /// After calling this, `current` and `target` will have the same length
    /// as `new_target`, enabling normal animation logic to proceed.
    pub fn resize_to_match(&mut self, new_target: &StructureState) {
        let old_len = self.current.len();
        let new_len = new_target.target.len();

        if new_len > old_len {
            // Grow: append target positions for new residues (they won't
            // animate)
            self.current
                .extend_from_slice(&new_target.target[old_len..]);
        } else if new_len < old_len {
            // Shrink: truncate to new size
            self.current.truncate(new_len);
        }

        // Update target and chain_lengths to match the new target
        self.target = new_target.target.clone();
        self.chain_lengths = new_target.chain_lengths.clone();
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
                    states.push(ResidueVisualState::backbone_only(backbone));
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

    #[test]
    fn test_differing_residues() {
        let state_a = StructureState::from_backbone(&make_backbone(0.0, 3));
        let state_b = StructureState::from_backbone(&make_backbone(10.0, 3));

        let diffs = state_a.differing_residues(&state_b);
        assert_eq!(diffs.len(), 3); // All residues differ
    }

    #[test]
    fn test_snap_to_target() {
        let mut state = StructureState::from_backbone(&make_backbone(0.0, 2));
        let new_target = StructureState::from_backbone(&make_backbone(10.0, 2));
        state.set_target(new_target);

        // Current should still be at 0
        assert!((state.current[0].backbone[0].y - 0.0).abs() < 0.001);

        state.snap_to_target();

        // Now current should be at 10
        assert!((state.current[0].backbone[0].y - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_resize_to_match_grow() {
        let mut state = StructureState::from_backbone(&make_backbone(0.0, 2));
        let larger = StructureState::from_backbone(&make_backbone(5.0, 4));

        state.resize_to_match(&larger);

        // Should now have 4 residues
        assert_eq!(state.residue_count(), 4);
        // First 2 keep their old positions (y=0)
        assert!((state.current[0].backbone[0].y - 0.0).abs() < 0.001);
        assert!((state.current[1].backbone[0].y - 0.0).abs() < 0.001);
        // Extra residues get target positions (y=5)
        assert!((state.current[2].backbone[0].y - 5.0).abs() < 0.001);
        assert!((state.current[3].backbone[0].y - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_resize_to_match_shrink() {
        let mut state = StructureState::from_backbone(&make_backbone(0.0, 4));
        let smaller = StructureState::from_backbone(&make_backbone(5.0, 2));

        state.resize_to_match(&smaller);

        // Should now have 2 residues
        assert_eq!(state.residue_count(), 2);
        // Keep first 2 residues at old positions
        assert!((state.current[0].backbone[0].y - 0.0).abs() < 0.001);
        assert!((state.current[1].backbone[0].y - 0.0).abs() < 0.001);
    }
}
