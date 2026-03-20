//! Consolidated entity ownership, per-entity behaviors, focus state, and
//! structural dirty tracking.

use glam::Vec3;
use molex::types::assembly::update_protein_entities;
use molex::types::coords::Coords;
use molex::types::entity::{MoleculeEntity, MoleculeType};
use rustc_hash::FxHashMap;

use super::scene::Focus;
use super::scene_data::{PerEntityData, SceneEntity};
use crate::animation::transition::Transition;
use crate::options::{DisplayOptions, EntityDisplayOverride};

/// Consolidated entity storage.
///
/// Owns scene entities (the single source of truth), per-entity animation
/// behaviors, focus state, and structural dirty tracking.
pub(crate) struct EntityStore {
    /// Scene entities (rendering copy with visibility, SS override, scores).
    scene_entities: Vec<SceneEntity>,
    /// Entity ID → index in `scene_entities` for O(1) lookup.
    id_index: FxHashMap<u32, usize>,
    /// Per-entity animation behavior overrides.
    behaviors: FxHashMap<u32, Transition>,
    /// Per-entity display/geometry overrides.
    display_overrides: FxHashMap<u32, EntityDisplayOverride>,
    focus: Focus,
    next_entity_id: u32,
    /// Monotonically increasing generation; bumped on any structural mutation
    /// (add/remove entity, coords update).
    generation: u64,
    /// Generation that was last consumed by the renderer.
    rendered_generation: u64,
}

impl EntityStore {
    /// Create an empty entity store.
    pub fn new() -> Self {
        Self {
            scene_entities: Vec::new(),
            id_index: FxHashMap::default(),
            behaviors: FxHashMap::default(),
            display_overrides: FxHashMap::default(),
            focus: Focus::Session,
            next_entity_id: 0,
            generation: 0,
            rendered_generation: 0,
        }
    }

    // -- Dirty tracking --

    fn invalidate(&mut self) {
        self.generation += 1;
    }

    /// Whether entity data changed since last `mark_rendered()`.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.generation != self.rendered_generation
    }

    /// Force the store dirty (e.g. when display options change but entity data
    /// hasn't).
    pub fn force_dirty(&mut self) {
        self.invalidate();
    }

    /// Mark current generation as rendered (call after updating renderers).
    pub fn mark_rendered(&mut self) {
        self.rendered_generation = self.generation;
    }

    // -- Entity management --

    /// Add entities to the store. Entity IDs are preserved from the caller
    /// (the caller is responsible for assigning unique IDs). Returns the
    /// entity IDs as passed in.
    pub fn add_entities(&mut self, entities: Vec<MoleculeEntity>) -> Vec<u32> {
        let mut ids = Vec::with_capacity(entities.len());
        for entity in entities {
            let id = entity.entity_id;
            // Track the highest ID we've seen so auto-assignment stays safe
            if id >= self.next_entity_id {
                self.next_entity_id = id + 1;
            }
            let idx = self.scene_entities.len();
            self.scene_entities.push(SceneEntity::new(entity));
            let _ = self.id_index.insert(id, idx);
            ids.push(id);
        }
        self.invalidate();
        ids
    }

    /// Set initial visibility for ambient entity types (water, ion, solvent)
    /// based on display options. Call after `add_entities` to synchronize
    /// entity-level visibility with display toggles.
    pub fn apply_type_visibility(&mut self, display: &DisplayOptions) {
        for se in &mut self.scene_entities {
            let visible = match se.entity.molecule_type {
                MoleculeType::Water => display.show_waters,
                MoleculeType::Ion => display.show_ions,
                MoleculeType::Solvent => display.show_solvent,
                _ => se.visible,
            };
            se.visible = visible;
        }
    }

    /// Set visibility for all entities of a given molecule type.
    pub fn set_type_visible(&mut self, mol_type: MoleculeType, visible: bool) {
        let mut changed = false;
        for se in &mut self.scene_entities {
            if se.entity.molecule_type == mol_type && se.visible != visible {
                se.visible = visible;
                se.invalidate_render_cache();
                changed = true;
            }
        }
        if changed {
            self.invalidate();
        }
    }

    /// Whether the store contains an entity with the given ID.
    #[must_use]
    pub fn has_entity(&self, id: u32) -> bool {
        self.id_index.contains_key(&id)
    }

    /// Number of entities in the store.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.scene_entities.len()
    }

    /// Remove all entities from the store.
    pub fn clear(&mut self) {
        self.scene_entities.clear();
        self.id_index.clear();
        self.behaviors.clear();
        self.display_overrides.clear();
        self.focus = Focus::Session;
        self.invalidate();
    }

    /// Read access to a scene entity.
    #[must_use]
    pub fn entity(&self, id: u32) -> Option<&SceneEntity> {
        self.id_index.get(&id).map(|&idx| &self.scene_entities[idx])
    }

    /// Mutable access to a scene entity.
    pub fn entity_mut(&mut self, id: u32) -> Option<&mut SceneEntity> {
        self.id_index
            .get(&id)
            .map(|&idx| &mut self.scene_entities[idx])
    }

    /// Remove an entity by ID. Returns true if the entity existed.
    pub fn remove_entity(&mut self, id: u32) -> bool {
        let Some(idx) = self.id_index.remove(&id) else {
            return false;
        };
        drop(self.scene_entities.swap_remove(idx));
        let _ = self.display_overrides.remove(&id);
        // If swap_remove moved the last element into `idx`, update its index.
        if idx < self.scene_entities.len() {
            let swapped_id = self.scene_entities[idx].id();
            let _ = self.id_index.insert(swapped_id, idx);
        }
        self.invalidate();
        true
    }

    /// Read access to all scene entities (insertion order).
    #[must_use]
    pub fn entities(&self) -> &[SceneEntity] {
        &self.scene_entities
    }

    // -- Focus / tab cycling --

    /// Current focus state.
    #[must_use]
    pub fn focus(&self) -> &Focus {
        &self.focus
    }

    /// Set the focus state directly.
    pub fn set_focus(&mut self, focus: Focus) {
        self.focus = focus;
    }

    /// Cycle: Session -> Entity1 -> ... -> EntityN -> Session.
    pub fn cycle_focus(&mut self) -> Focus {
        let focusable: Vec<u32> = self
            .scene_entities
            .iter()
            .filter(|e| e.visible && e.entity.is_focusable())
            .map(SceneEntity::id)
            .collect();

        self.focus = match self.focus {
            Focus::Session => focusable
                .first()
                .map_or(Focus::Session, |&id| Focus::Entity(id)),
            Focus::Entity(current_id) => {
                let idx = focusable.iter().position(|&id| id == current_id);
                match idx {
                    Some(i) if i + 1 < focusable.len() => {
                        Focus::Entity(focusable[i + 1])
                    }
                    _ => Focus::Session,
                }
            }
        };
        self.focus
    }

    // -- Filtered entity access --

    /// Visible nucleic acid (DNA/RNA) entities.
    pub fn nucleic_acid_entities(&self) -> impl Iterator<Item = &SceneEntity> {
        self.scene_entities
            .iter()
            .filter(|e| e.visible && e.is_nucleic_acid())
    }

    /// Visible ligand entities (not protein, not nucleic acid).
    pub fn ligand_entities(&self) -> impl Iterator<Item = &SceneEntity> {
        self.scene_entities
            .iter()
            .filter(|e| e.visible && e.is_ligand())
    }

    // -- Per-entity behavior --

    /// Set the animation behavior override for a specific entity.
    pub fn set_behavior(&mut self, entity_id: u32, transition: Transition) {
        let _ = self.behaviors.insert(entity_id, transition);
    }

    /// Clear a per-entity behavior override, reverting to default.
    pub fn clear_behavior(&mut self, entity_id: u32) {
        let _ = self.behaviors.remove(&entity_id);
    }

    /// Look up a per-entity behavior override.
    #[must_use]
    pub fn behavior(&self, entity_id: u32) -> Option<&Transition> {
        self.behaviors.get(&entity_id)
    }

    // -- Per-entity display overrides --

    /// Set a display override for a specific entity.
    pub fn set_display_override(
        &mut self,
        entity_id: u32,
        overrides: EntityDisplayOverride,
    ) {
        let _ = self.display_overrides.insert(entity_id, overrides);
    }

    /// Clear a per-entity display override, reverting to session defaults.
    pub fn clear_display_override(&mut self, entity_id: u32) {
        let _ = self.display_overrides.remove(&entity_id);
    }

    /// Look up a per-entity display override.
    #[must_use]
    pub fn display_override(
        &self,
        entity_id: u32,
    ) -> Option<&EntityDisplayOverride> {
        self.display_overrides.get(&entity_id)
    }

    /// All display overrides (for building per-entity options maps).
    #[must_use]
    pub fn display_overrides(&self) -> &FxHashMap<u32, EntityDisplayOverride> {
        &self.display_overrides
    }

    // -- Per-entity data --

    /// Collect per-entity render data for all visible entities.
    pub fn per_entity_data(&self) -> Vec<PerEntityData> {
        self.scene_entities
            .iter()
            .filter(|e| e.visible)
            .filter_map(SceneEntity::to_per_entity_data)
            .collect()
    }

    /// Aggregate bounding sphere across all visible entities.
    ///
    /// Returns `None` when no visible entities have atoms. Merges per-entity
    /// cached bounding spheres into a single enclosing sphere (centroid is the
    /// weighted average of entity centroids; radius is the max distance from
    /// that centroid to any entity's outer edge).
    #[must_use]
    pub fn bounding_sphere(&self) -> Option<(Vec3, f32)> {
        let visible: Vec<&SceneEntity> =
            self.scene_entities.iter().filter(|e| e.visible).collect();
        if visible.is_empty() {
            return None;
        }

        // Weighted centroid (weight = atom count).
        let mut total_weight: f32 = 0.0;
        let mut weighted_sum = Vec3::ZERO;
        for e in &visible {
            let w = e.entity.atom_count() as f32;
            if w > 0.0 {
                weighted_sum += e.cached_centroid * w;
                total_weight += w;
            }
        }
        if total_weight == 0.0 {
            return None;
        }
        let centroid = weighted_sum / total_weight;

        // Radius: max distance from the combined centroid to each entity's
        // outer edge (entity centroid offset + entity radius).
        let radius = visible
            .iter()
            .map(|e| {
                (e.cached_centroid - centroid).length()
                    + e.cached_bounding_radius
            })
            .fold(0.0f32, f32::max);

        Some((centroid, radius))
    }

    // -- Entity mutation --

    /// Replace the `MoleculeEntity` for an existing entity by id.
    ///
    /// Bumps mesh version, recomputes bounds, and invalidates the generation
    /// counter.
    pub fn replace_entity(&mut self, entity: MoleculeEntity) {
        if let Some(&idx) = self.id_index.get(&entity.entity_id) {
            let se = &mut self.scene_entities[idx];
            se.entity = entity;
            se.recompute_bounds();
            se.invalidate_render_cache();
        }
        self.invalidate();
    }

    /// Update protein entity coords in a specific scene entity.
    pub fn update_entity_protein_coords(&mut self, id: u32, coords: &Coords) {
        if let Some(&idx) = self.id_index.get(&id) {
            let se = &mut self.scene_entities[idx];
            let mut entities = vec![se.entity.clone()];
            update_protein_entities(&mut entities, coords);
            if let Some(updated) = entities.into_iter().next() {
                se.entity = updated;
            }
            se.recompute_bounds();
            se.invalidate_render_cache();
        }
        self.invalidate();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::animation::transition::Transition;
    use crate::engine::scene::Focus;
    use crate::engine::test_fixtures::{
        make_protein_entity, make_water_entity,
    };

    #[test]
    fn new_is_empty() {
        let store = EntityStore::new();
        assert!(store.entities().is_empty());
        assert!(!store.is_dirty());
    }

    #[test]
    fn add_preserves_caller_ids() {
        let mut store = EntityStore::new();
        let ids_a = store.add_entities(vec![make_protein_entity(10, b'A', 3)]);
        let ids_b = store.add_entities(vec![make_protein_entity(20, b'B', 2)]);
        assert_eq!(ids_a, vec![10]);
        assert_eq!(ids_b, vec![20]);
    }

    #[test]
    fn add_bumps_generation() {
        let mut store = EntityStore::new();
        assert!(!store.is_dirty());
        let _ = store.add_entities(vec![make_protein_entity(0, b'A', 1)]);
        assert!(store.is_dirty());
    }

    #[test]
    fn lookup_by_id() {
        let mut store = EntityStore::new();
        let ids = store.add_entities(vec![make_protein_entity(0, b'A', 2)]);
        let se = store.entity(ids[0]);
        assert!(se.is_some());
        assert_eq!(se.unwrap().id(), ids[0]);
    }

    #[test]
    fn lookup_missing() {
        let store = EntityStore::new();
        assert!(store.entity(999).is_none());
    }

    #[test]
    fn remove_basic() {
        let mut store = EntityStore::new();
        let ids = store.add_entities(vec![make_protein_entity(0, b'A', 1)]);
        assert!(store.remove_entity(ids[0]));
        assert!(store.entity(ids[0]).is_none());
    }

    #[test]
    fn remove_missing() {
        let mut store = EntityStore::new();
        assert!(!store.remove_entity(999));
    }

    #[test]
    fn remove_swap_updates_index() {
        let mut store = EntityStore::new();
        let ids = store.add_entities(vec![
            make_protein_entity(10, b'A', 1),
            make_protein_entity(11, b'B', 1),
            make_protein_entity(12, b'C', 1),
        ]);
        // Remove first — swap_remove moves last element into slot 0
        assert!(store.remove_entity(ids[0]));
        assert!(store.entity(ids[0]).is_none());
        assert!(store.entity(ids[1]).is_some());
        assert!(store.entity(ids[2]).is_some());
    }

    #[test]
    fn mark_rendered_clears_dirty() {
        let mut store = EntityStore::new();
        let _ = store.add_entities(vec![make_protein_entity(0, b'A', 1)]);
        assert!(store.is_dirty());
        store.mark_rendered();
        assert!(!store.is_dirty());
    }

    #[test]
    fn force_dirty() {
        let mut store = EntityStore::new();
        let _ = store.add_entities(vec![make_protein_entity(0, b'A', 1)]);
        store.mark_rendered();
        assert!(!store.is_dirty());
        store.force_dirty();
        assert!(store.is_dirty());
    }

    #[test]
    fn cycle_focus_through_entities() {
        let mut store = EntityStore::new();
        let _ = store.add_entities(vec![
            make_protein_entity(10, b'A', 1),
            make_protein_entity(11, b'B', 1),
        ]);
        assert_eq!(*store.focus(), Focus::Session);
        let f1 = store.cycle_focus();
        assert_eq!(f1, Focus::Entity(10));
        let f2 = store.cycle_focus();
        assert_eq!(f2, Focus::Entity(11));
        let f3 = store.cycle_focus();
        assert_eq!(f3, Focus::Session);
    }

    #[test]
    fn cycle_focus_skips_non_focusable() {
        let mut store = EntityStore::new();
        // Water is not focusable, should be skipped
        let _ = store.add_entities(vec![make_water_entity(0)]);
        assert_eq!(*store.focus(), Focus::Session);
        let f = store.cycle_focus();
        assert_eq!(f, Focus::Session); // no focusable entities
    }

    #[test]
    fn behavior_set_get_clear() {
        let mut store = EntityStore::new();
        assert!(store.behavior(0).is_none());
        store.set_behavior(0, Transition::snap());
        assert!(store.behavior(0).is_some());
        store.clear_behavior(0);
        assert!(store.behavior(0).is_none());
    }
}
