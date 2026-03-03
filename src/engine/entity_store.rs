//! Consolidated entity ownership, per-entity behaviors, focus state, and
//! structural dirty tracking.

use foldit_conv::types::assembly::update_protein_entities;
use foldit_conv::types::coords::Coords;
use foldit_conv::types::entity::MoleculeEntity;
use glam::Vec3;
use rustc_hash::FxHashMap;

use super::scene::Focus;
use super::scene_data::{PerEntityData, SceneEntity};
use crate::animation::transition::Transition;

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
    pub(crate) behaviors: FxHashMap<u32, Transition>,
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

    /// Add entities to the store. Entity IDs are reassigned to be globally
    /// unique. Returns the assigned entity IDs.
    pub fn add_entities(&mut self, entities: Vec<MoleculeEntity>) -> Vec<u32> {
        let mut ids = Vec::with_capacity(entities.len());
        for mut entity in entities {
            let id = self.next_entity_id;
            self.next_entity_id += 1;
            entity.entity_id = id;
            let idx = self.scene_entities.len();
            self.scene_entities.push(SceneEntity {
                entity,
                visible: true,
                ss_override: None,
                per_residue_scores: None,
                mesh_version: 0,
            });
            let _ = self.id_index.insert(id, idx);
            ids.push(id);
        }
        self.invalidate();
        ids
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
            .filter(|e| e.visible)
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

    // -- Per-entity data --

    /// Collect per-entity render data for all visible entities.
    pub fn per_entity_data(&self) -> Vec<PerEntityData> {
        self.scene_entities
            .iter()
            .filter(|e| e.visible)
            .filter_map(SceneEntity::to_per_entity_data)
            .collect()
    }

    /// All atom positions across all visible entities (for camera fitting).
    #[must_use]
    pub fn all_positions(&self) -> Vec<Vec3> {
        self.scene_entities
            .iter()
            .filter(|e| e.visible)
            .flat_map(|e| {
                e.entity
                    .coords
                    .atoms
                    .iter()
                    .map(|a| Vec3::new(a.x, a.y, a.z))
            })
            .collect()
    }

    // -- Entity mutation --

    /// Replace the `MoleculeEntity` for an existing entity by id.
    ///
    /// Bumps mesh version and invalidates the generation counter.
    pub fn replace_entity(&mut self, entity: MoleculeEntity) {
        if let Some(&idx) = self.id_index.get(&entity.entity_id) {
            let se = &mut self.scene_entities[idx];
            se.entity = entity;
            se.invalidate_render_cache();
        }
        self.invalidate();
    }

    /// Update protein entity coords in a specific scene entity.
    pub fn update_entity_protein_coords(&mut self, id: u32, coords: Coords) {
        if let Some(&idx) = self.id_index.get(&id) {
            let se = &mut self.scene_entities[idx];
            let mut entities = vec![se.entity.clone()];
            update_protein_entities(&mut entities, coords);
            if let Some(updated) = entities.into_iter().next() {
                se.entity = updated;
            }
            se.invalidate_render_cache();
        }
        self.invalidate();
    }
}
