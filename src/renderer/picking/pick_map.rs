//! Typed pick-target resolution from raw GPU pick IDs.

/// A typed pick target resolved from a raw GPU pick ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickTarget {
    /// No target (background click or no hover).
    None,
    /// A protein/nucleic-acid residue, identified by its flat index.
    Residue(u32),
    /// A small-molecule atom, identified by entity ID and atom index.
    Atom {
        /// Entity that owns this atom.
        entity_id: u32,
        /// Atom index within the entity.
        atom_idx: u32,
    },
}

impl PickTarget {
    /// Convert to the legacy `i32` residue index used by the camera uniform
    /// and input system. Returns the residue index for `Residue`, or `-1`
    /// for `None` and `Atom`.
    #[must_use]
    pub fn as_residue_i32(&self) -> i32 {
        match *self {
            Self::Residue(idx) => idx as i32,
            _ => -1,
        }
    }

    /// Returns `true` if this target is `None`.
    #[must_use]
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// Maps raw GPU pick IDs to typed [`PickTarget`] values.
///
/// Pick IDs are contiguous:
/// - `0` → no hit
/// - `1..=residue_count` → residue (ID = residue_idx + 1)
/// - `residue_count+1..=residue_count+atom_count` → small-molecule atom
///
/// Atom entries store `(entity_id, atom_idx)` pairs in contiguous order.
#[derive(Clone)]
pub(crate) struct PickMap {
    residue_count: u32,
    atom_entries: Vec<(u32, u32)>,
}

impl PickMap {
    /// Create a new PickMap.
    pub(crate) fn new(
        residue_count: u32,
        atom_entries: Vec<(u32, u32)>,
    ) -> Self {
        Self {
            residue_count,
            atom_entries,
        }
    }

    /// Resolve a raw pick ID (as read from the GPU picking buffer) to a typed
    /// target.
    pub(crate) fn resolve(&self, raw_id: u32) -> PickTarget {
        if raw_id == 0 {
            return PickTarget::None;
        }
        let idx = raw_id - 1; // pick IDs are 1-based
        if idx < self.residue_count {
            return PickTarget::Residue(idx);
        }
        let atom_local = idx - self.residue_count;
        if let Some(&(entity_id, atom_idx)) =
            self.atom_entries.get(atom_local as usize)
        {
            return PickTarget::Atom {
                entity_id,
                atom_idx,
            };
        }
        PickTarget::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_resolves_to_none() {
        let map = PickMap::new(10, vec![(0, 0), (0, 1)]);
        assert_eq!(map.resolve(0), PickTarget::None);
    }

    #[test]
    fn residue_ids_one_based() {
        let map = PickMap::new(5, vec![]);
        assert_eq!(map.resolve(1), PickTarget::Residue(0));
        assert_eq!(map.resolve(5), PickTarget::Residue(4));
    }

    #[test]
    fn atom_ids_follow_residues() {
        let map = PickMap::new(3, vec![(10, 0), (10, 1), (20, 0)]);
        // residues: IDs 1-3, atoms start at ID 4
        assert_eq!(
            map.resolve(4),
            PickTarget::Atom {
                entity_id: 10,
                atom_idx: 0,
            }
        );
        assert_eq!(
            map.resolve(6),
            PickTarget::Atom {
                entity_id: 20,
                atom_idx: 0,
            }
        );
    }

    #[test]
    fn out_of_range_resolves_to_none() {
        let map = PickMap::new(2, vec![(0, 0)]);
        // total valid IDs: 1,2 (residues) + 3 (atom) = 3
        assert_eq!(map.resolve(4), PickTarget::None);
        assert_eq!(map.resolve(100), PickTarget::None);
    }

    #[test]
    fn empty_map() {
        let map = PickMap::new(0, vec![]);
        assert_eq!(map.resolve(0), PickTarget::None);
        assert_eq!(map.resolve(1), PickTarget::None);
    }

    #[test]
    fn as_residue_i32() {
        assert_eq!(PickTarget::Residue(5).as_residue_i32(), 5);
        assert_eq!(PickTarget::None.as_residue_i32(), -1);
        assert_eq!(
            PickTarget::Atom {
                entity_id: 0,
                atom_idx: 0,
            }
            .as_residue_i32(),
            -1
        );
    }

    #[test]
    fn is_none_predicate() {
        assert!(PickTarget::None.is_none());
        assert!(!PickTarget::Residue(0).is_none());
        assert!(!PickTarget::Atom {
            entity_id: 0,
            atom_idx: 0,
        }
        .is_none());
    }
}
