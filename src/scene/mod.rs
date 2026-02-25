//! Authoritative scene: flat entity storage, focus cycling, per-entity
//! render data.
//!
//! Everything is a [`MoleculeEntity`]. Each entity is wrapped in a
//! [`SceneEntity`] that pairs the core data with rendering metadata
//! (visibility, name, SS override, score cache, mesh version).

mod entity;
mod entity_data;
mod mesh_concat;
mod mesh_gen;
pub(crate) mod prepared;
pub mod processor;

pub use entity::*;
pub use entity_data::*;
use foldit_conv::types::assembly::update_protein_entities;
use foldit_conv::types::coords::Coords;
use foldit_conv::types::entity::MoleculeEntity;
use glam::Vec3;
pub use prepared::{PreparedAnimationFrame, PreparedScene, SceneRequest};

// ---------------------------------------------------------------------------
// Focus
// ---------------------------------------------------------------------------

/// Focus state for tab cycling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Focus {
    /// All entities.
    #[default]
    Session,
    /// A specific entity by ID.
    Entity(u32),
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// The authoritative scene. Owns all entities in a flat list.
pub struct Scene {
    /// Entities in insertion order.
    entities: Vec<SceneEntity>,
    focus: Focus,
    next_entity_id: u32,
    /// Monotonically increasing generation; bumped on any mutation.
    generation: u64,
    /// Generation that was last consumed by the renderer.
    rendered_generation: u64,
}

impl Scene {
    /// Create an empty scene with no entities and session-level focus.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            focus: Focus::Session,
            next_entity_id: 0,
            generation: 0,
            rendered_generation: 0,
        }
    }

    // -- Mutation helpers --

    fn invalidate(&mut self) {
        self.generation += 1;
    }

    /// Whether scene data changed since last `mark_rendered()`.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.generation != self.rendered_generation
    }

    /// Force the scene dirty (e.g. when display options change but scene data
    /// hasn't).
    pub fn force_dirty(&mut self) {
        self.invalidate();
    }

    /// Mark current generation as rendered (call after updating renderers).
    pub fn mark_rendered(&mut self) {
        self.rendered_generation = self.generation;
    }

    // -- Entity management --

    /// Add entities to the scene. Entity IDs are reassigned to be globally
    /// unique. Returns the assigned entity IDs.
    pub fn add_entities(&mut self, entities: Vec<MoleculeEntity>) -> Vec<u32> {
        let mut ids = Vec::with_capacity(entities.len());
        for mut entity in entities {
            let id = self.next_entity_id;
            self.next_entity_id += 1;
            entity.entity_id = id;
            let name = entity.label();
            self.entities.push(SceneEntity {
                entity,
                visible: true,
                name,
                ss_override: None,
                per_residue_scores: None,
                mesh_version: 0,
            });
            ids.push(id);
        }
        self.invalidate();
        ids
    }

    /// Remove an entity by ID. Returns the removed entity, if any.
    pub fn remove_entity(&mut self, id: u32) -> Option<SceneEntity> {
        let idx = self.entities.iter().position(|e| e.id() == id)?;
        let entity = self.entities.remove(idx);
        self.invalidate();
        Some(entity)
    }

    /// Remove multiple entities by ID.
    pub fn remove_entities(&mut self, ids: &[u32]) {
        self.entities.retain(|e| !ids.contains(&e.id()));
        self.invalidate();
    }

    /// Replace all entities whose IDs are in `old_ids` with new entities.
    /// Returns the new entity IDs.
    pub fn replace_entities(
        &mut self,
        old_ids: &[u32],
        new_entities: Vec<MoleculeEntity>,
    ) -> Vec<u32> {
        self.entities.retain(|e| !old_ids.contains(&e.id()));
        self.add_entities(new_entities)
    }

    /// Read access to an entity.
    #[must_use]
    pub fn entity(&self, id: u32) -> Option<&SceneEntity> {
        self.entities.iter().find(|e| e.id() == id)
    }

    /// Write access (invalidates cache).
    pub fn entity_mut(&mut self, id: u32) -> Option<&mut SceneEntity> {
        self.invalidate();
        self.entities.iter_mut().find(|e| e.id() == id)
    }

    /// Read access to all entities (insertion order).
    #[must_use]
    pub fn entities(&self) -> &[SceneEntity] {
        &self.entities
    }

    /// Number of entities.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Toggle visibility.
    pub fn set_visible(&mut self, id: u32, visible: bool) {
        if let Some(e) = self.entities.iter_mut().find(|e| e.id() == id) {
            if e.visible != visible {
                e.visible = visible;
                self.invalidate();
            }
        }
    }

    /// Remove all entities and reset.
    pub fn clear(&mut self) {
        self.entities.clear();
        self.focus = Focus::Session;
        self.invalidate();
    }

    /// Check if an entity exists.
    #[must_use]
    pub fn contains(&self, id: u32) -> bool {
        self.entities.iter().any(|e| e.id() == id)
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
            .entities
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

    /// Revert to Session if focused entity was removed.
    pub fn validate_focus(&mut self) {
        if let Focus::Entity(eid) = self.focus {
            if !self.contains(eid) {
                self.focus = Focus::Session;
            }
        }
    }

    /// Human-readable description of current focus.
    #[must_use]
    pub fn focus_description(&self) -> String {
        match self.focus {
            Focus::Session => "Session (all structures)".into(),
            Focus::Entity(eid) => self
                .entity(eid)
                .map_or_else(|| "Entity (unknown)".into(), |e| e.name.clone()),
        }
    }

    // -- Filtered entity access --

    /// Visible entities only.
    #[must_use]
    pub fn visible_entities(&self) -> Vec<&SceneEntity> {
        self.entities.iter().filter(|e| e.visible).collect()
    }

    /// Visible protein entities.
    #[must_use]
    pub fn protein_entities(&self) -> Vec<&SceneEntity> {
        self.entities
            .iter()
            .filter(|e| e.visible && e.is_protein())
            .collect()
    }

    /// Visible nucleic acid (DNA/RNA) entities.
    #[must_use]
    pub fn nucleic_acid_entities(&self) -> Vec<&SceneEntity> {
        self.entities
            .iter()
            .filter(|e| e.visible && e.is_nucleic_acid())
            .collect()
    }

    /// Visible ligand entities (not protein, not nucleic acid).
    #[must_use]
    pub fn ligand_entities(&self) -> Vec<&SceneEntity> {
        self.entities
            .iter()
            .filter(|e| e.visible && e.is_ligand())
            .collect()
    }

    // -- Per-entity data for scene processor --

    /// Collect per-entity render data for all visible entities.
    pub fn per_entity_data(&self) -> Vec<PerEntityData> {
        self.entities
            .iter()
            .filter(|e| e.visible)
            .filter_map(SceneEntity::to_per_entity_data)
            .collect()
    }

    /// All atom positions across all visible entities (for camera fitting).
    #[must_use]
    pub fn all_positions(&self) -> Vec<Vec3> {
        self.entities
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

    /// Update protein entity coords in a specific entity.
    pub fn update_entity_protein_coords(&mut self, id: u32, coords: Coords) {
        if let Some(se) =
            self.entities.iter_mut().find(|e| e.entity.entity_id == id)
        {
            // Wrap in a temporary Vec for the assembly update function
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

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
