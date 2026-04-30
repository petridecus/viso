//! Per-entity atom position map: animator write surface, renderer
//! read surface.
//!
//! Positions are viso-local. They start as a snapshot of each entity's
//! `Assembly` reference positions on sync and are mutated in-place by
//! the animator every frame. The render path reads through [`get`]
//! without ever touching `Assembly`.
//!
//! [`get`]: EntityPositions::get

use glam::Vec3;
use molex::entity::molecule::id::EntityId;
use rustc_hash::FxHashMap;

/// Per-entity animator write surface and renderer read surface.
///
/// Reconciled on every [`Assembly`](molex::Assembly) sync: new entities
/// get an initial reference snapshot inserted; removed entities are
/// dropped.
#[derive(Default, Clone)]
pub(crate) struct EntityPositions {
    per_entity: FxHashMap<EntityId, Vec<Vec3>>,
}

impl EntityPositions {
    /// Empty positions map.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            per_entity: FxHashMap::default(),
        }
    }

    /// Read-only position slice for an entity.
    #[must_use]
    pub(crate) fn get(&self, id: EntityId) -> Option<&[Vec3]> {
        self.per_entity.get(&id).map(Vec::as_slice)
    }

    /// Mutable position slice for an entity.
    pub(crate) fn get_mut(&mut self, id: EntityId) -> Option<&mut Vec<Vec3>> {
        self.per_entity.get_mut(&id)
    }

    /// Replace the positions for an entity (overwrites existing slot).
    pub(crate) fn set(&mut self, id: EntityId, positions: Vec<Vec3>) {
        let _ = self.per_entity.insert(id, positions);
    }

    /// Insert positions for an entity from a reference snapshot, or
    /// reset them if the slot's atom count no longer matches the
    /// reference (entity replaced with a different-shaped one). Used
    /// on sync to seed positions for new entities and to invalidate
    /// stale buffers when an existing entity's topology changes (e.g.
    /// streaming backbone-only frames followed by a full-atom result).
    pub(crate) fn insert_from_reference(
        &mut self,
        id: EntityId,
        reference: &[Vec3],
    ) {
        match self.per_entity.entry(id) {
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                if slot.get().len() != reference.len() {
                    *slot.get_mut() = reference.to_vec();
                }
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                let _ = slot.insert(reference.to_vec());
            }
        }
    }

    /// Keep only entities for which `keep` returns true.
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(EntityId) -> bool) {
        self.per_entity.retain(|&id, _| keep(id));
    }
}
