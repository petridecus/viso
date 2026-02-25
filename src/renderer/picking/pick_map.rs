//! Typed pick-target resolution from raw GPU pick IDs.

/// A typed pick target resolved from a raw GPU pick ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickTarget {
    /// No target (background click or no hover).
    None,
    /// A protein/nucleic-acid residue, identified by its flat index.
    Residue(u32),
    /// A small-molecule atom, identified by entity ID and atom index.
    Atom { entity_id: u32, atom_idx: u32 },
}

impl PickTarget {
    /// Convert to the legacy `i32` residue index used by the camera uniform
    /// and input system. Returns the residue index for `Residue`, or `-1`
    /// for `None` and `Atom`.
    pub fn as_residue_i32(&self) -> i32 {
        match *self {
            Self::Residue(idx) => idx as i32,
            _ => -1,
        }
    }

    /// Returns `true` if this target is `None`.
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
pub struct PickMap {
    residue_count: u32,
    atom_entries: Vec<(u32, u32)>,
}

impl PickMap {
    /// Create a new PickMap.
    pub fn new(residue_count: u32, atom_entries: Vec<(u32, u32)>) -> Self {
        Self {
            residue_count,
            atom_entries,
        }
    }

    /// Resolve a raw pick ID (as read from the GPU picking buffer) to a typed
    /// target.
    pub fn resolve(&self, raw_id: u32) -> PickTarget {
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
