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
pub struct EntityPositions {
    /// Per-entity atom positions, keyed by entity id.
    pub per_entity: FxHashMap<EntityId, Vec<Vec3>>,
}

impl EntityPositions {
    /// Empty positions map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            per_entity: FxHashMap::default(),
        }
    }

    /// Read-only position slice for an entity.
    #[must_use]
    pub fn get(&self, id: EntityId) -> Option<&[Vec3]> {
        self.per_entity.get(&id).map(Vec::as_slice)
    }

    /// Mutable position slice for an entity.
    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut Vec<Vec3>> {
        self.per_entity.get_mut(&id)
    }

    /// Replace the positions for an entity (overwrites existing slot).
    pub fn set(&mut self, id: EntityId, positions: Vec<Vec3>) {
        let _ = self.per_entity.insert(id, positions);
    }

    /// Insert a new entity from a reference position snapshot if absent.
    ///
    /// Used on sync when a new entity joined the assembly: the initial
    /// positions are copied from the assembly's reference positions so
    /// the animator has a visual state to interpolate from.
    pub fn insert_from_reference(&mut self, id: EntityId, reference: &[Vec3]) {
        let _ = self
            .per_entity
            .entry(id)
            .or_insert_with(|| reference.to_vec());
    }

    /// Keep only entities for which `keep` returns true.
    pub fn retain(&mut self, mut keep: impl FnMut(EntityId) -> bool) {
        self.per_entity.retain(|&id, _| keep(id));
    }
}
