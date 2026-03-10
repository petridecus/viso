//! Entity management: loading, updating, constraint visualization, behavior.

use std::collections::HashMap;

use foldit_conv::secondary_structure::SSType;
use foldit_conv::types::entity::MoleculeEntity;
use glam::Vec3;

use super::command::{BandInfo, PullInfo};
use super::scene_data::SceneEntity;
use super::{constraint, VisoEngine};
use crate::animation::transition::Transition;
use crate::renderer::geometry::BackboneUpdateData;

// ── Entity behavior ──

impl VisoEngine {
    /// Set the animation behavior for a specific entity.
    ///
    /// This behavior will be used when the entity is next updated.
    /// Overrides the default smooth transition for the given entity.
    pub fn set_entity_behavior(
        &mut self,
        entity_id: u32,
        transition: Transition,
    ) {
        self.entities.set_behavior(entity_id, transition);
    }

    /// Clear a per-entity behavior override, reverting to default (smooth).
    pub fn clear_entity_behavior(&mut self, entity_id: u32) {
        self.entities.clear_behavior(entity_id);
    }
}

// ── Scene data ──

impl VisoEngine {
    /// Load entities into the scene. Optionally fits camera.
    /// Returns the assigned entity IDs.
    pub fn load_entities(
        &mut self,
        entities: Vec<MoleculeEntity>,
        fit_camera: bool,
    ) -> Vec<u32> {
        // Check whether the animator has existing state to animate from.
        let was_empty = !self.animation.animator.is_animating()
            && self.visual.backbone_chains.is_empty();

        let ids = self.entities.add_entities(entities);
        if fit_camera {
            if was_empty {
                // No previous state — non-animated sync uploads backbone
                // directly via apply_prepared (the animator can't
                // interpolate from nothing).
                self.sync_scene_to_renderers(HashMap::new());
            } else {
                let snap_transitions: HashMap<u32, Transition> =
                    ids.iter().map(|&id| (id, Transition::snap())).collect();
                self.sync_scene_to_renderers(snap_transitions);
            }
            if let Some((centroid, radius)) = self.entities.bounding_sphere() {
                self.camera_controller.fit_to_sphere(centroid, radius);
            }
        }
        ids
    }

    /// Update backbone with new chains (regenerates the backbone mesh)
    /// Use this for designed backbones from ML models like RFDiffusion3
    pub fn update_backbone(&mut self, backbone_chains: &[Vec<Vec3>]) {
        self.gpu.renderers.backbone.update(
            &self.gpu.context,
            &BackboneUpdateData {
                protein_chains: backbone_chains,
                na_chains: &[],
                ss_types: None,
                geometry: &self.options.geometry,
            },
        );
    }

    /// Set SS override (from puzzle.toml annotation). Updates cached types
    /// and forces backbone renderer regeneration.
    pub fn set_ss_override(&mut self, ss_types: &[SSType]) {
        self.topology.ss_types = ss_types.to_vec();
        self.gpu
            .renderers
            .backbone
            .set_ss_override(Some(ss_types.to_vec()));
        let camera_eye = self.camera_controller.camera.eye;
        self.submit_per_chain_lod_remesh(camera_eye);
    }

    /// Replace the current set of constraint bands.
    ///
    /// Bands use structural references ([`super::command::AtomRef`]) and are
    /// resolved to world-space positions each frame, so they auto-track
    /// animated atoms.
    pub fn update_bands(&mut self, bands: Vec<BandInfo>) {
        self.constraints.band_specs = bands;
        self.resolve_and_render_constraints();
    }

    /// Set or clear the active pull constraint.
    ///
    /// Pulls use a structural reference for the atom and a screen-space
    /// target. The engine resolves positions each frame.
    pub fn update_pull(&mut self, pull: Option<PullInfo>) {
        self.constraints.pull_spec = pull;
        self.resolve_and_render_constraints();
    }
}

// ── Constraint resolution ──

impl VisoEngine {
    /// Resolve stored band/pull specs to world-space and update renderers.
    ///
    /// Called each frame from `pre_render` and immediately after
    /// `update_bands` / `update_pull` for instant visual feedback.
    pub(super) fn resolve_and_render_constraints(&mut self) {
        let scene = constraint::ScenePositions {
            visual: &self.visual,
            topology: &self.topology,
            cached_chains: self.gpu.renderers.backbone.cached_chains(),
        };

        // Bands
        let resolved_bands: Vec<_> = self
            .constraints
            .band_specs
            .iter()
            .filter_map(|b| constraint::resolve_band(&scene, b))
            .collect();
        self.gpu.renderers.band.update(
            &self.gpu.context.device,
            &self.gpu.context.queue,
            &resolved_bands,
            Some(&self.options.colors),
        );

        // Pull
        let viewport = (
            self.gpu.context.config.width,
            self.gpu.context.config.height,
        );
        let resolved_pull = self.constraints.pull_spec.as_ref().and_then(|p| {
            constraint::resolve_pull(
                &scene,
                &self.camera_controller,
                viewport,
                p,
            )
        });
        self.gpu.renderers.pull.update(
            &self.gpu.context.device,
            &self.gpu.context.queue,
            resolved_pull.as_ref(),
        );
    }
}

// ── Declarative reconciliation ──

impl VisoEngine {
    /// Reconcile the scene with a new set of entities.
    ///
    /// - New IDs (not in current scene) → added
    /// - Missing IDs (in scene but not in input) → removed
    /// - Existing IDs with changed data → updated with the given transition
    /// - Existing IDs, unchanged → no-op
    ///
    /// `default_transition` is used for new and updated entities.
    pub fn sync_entities(
        &mut self,
        entities: Vec<MoleculeEntity>,
        default_transition: &Transition,
    ) {
        use std::collections::HashSet;

        let incoming_ids: HashSet<u32> =
            entities.iter().map(|e| e.entity_id).collect();
        let current_ids: Vec<u32> = self
            .entities
            .entities()
            .iter()
            .map(SceneEntity::id)
            .collect();

        // Remove entities not in the incoming set
        for id in &current_ids {
            if !incoming_ids.contains(id) {
                self.entities.clear_behavior(*id);
                let _ = self.entities.remove_entity(*id);
            }
        }

        // Add or update entities
        let mut entity_transitions = HashMap::new();
        for entity in entities {
            let id = entity.entity_id;
            if self.entities.has_entity(id) {
                // Update existing entity
                self.entities.replace_entity(entity);
                let transition = self
                    .entities
                    .behavior(id)
                    .cloned()
                    .unwrap_or_else(|| default_transition.clone());
                let _ = entity_transitions.insert(id, transition);
            } else {
                // Add new entity
                let _ = self.entities.add_entities(vec![entity]);
                let _ =
                    entity_transitions.insert(id, default_transition.clone());
            }
        }

        self.sync_scene_to_renderers(entity_transitions);
    }

    /// Update a single entity by ID. Returns `Err` if the ID doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns [`crate::VisoError`] if no entity with the given ID exists.
    pub fn update_entity(
        &mut self,
        entity: MoleculeEntity,
        transition: Transition,
    ) -> Result<(), crate::VisoError> {
        let id = entity.entity_id;
        if !self.entities.has_entity(id) {
            return Err(crate::VisoError::StructureLoad(format!(
                "Entity {id} not found"
            )));
        }
        self.entities.replace_entity(entity);
        let effective =
            self.entities.behavior(id).cloned().unwrap_or(transition);
        let mut transitions = HashMap::new();
        let _ = transitions.insert(id, effective);
        self.sync_scene_to_renderers(transitions);
        Ok(())
    }
}

// ── Entity updates ──

impl VisoEngine {
    /// Update protein coords for a specific entity.
    ///
    /// Uses the entity's per-entity behavior override if set, otherwise
    /// falls back to the provided transition.
    pub fn update_entity_coords(
        &mut self,
        id: u32,
        coords: foldit_conv::types::coords::Coords,
        transition: Transition,
    ) {
        self.entities.update_entity_protein_coords(id, coords);

        // 3. Look up per-entity behavior override
        let effective_transition =
            self.entities.behavior(id).cloned().unwrap_or(transition);

        // 4. Sync with per-entity transition
        let mut entity_transitions = HashMap::new();
        let _ = entity_transitions.insert(id, effective_transition);
        self.sync_scene_to_renderers(entity_transitions);
    }

    /// Replace one or more entities with new `MoleculeEntity` data.
    ///
    /// Each entity is matched by `entity_id`. A targeted sync is triggered
    /// for all changed entities. Per-entity behavior overrides are used when
    /// set, otherwise `default_transition` is applied.
    pub fn update_entities(
        &mut self,
        updated: Vec<MoleculeEntity>,
        default_transition: &Transition,
    ) {
        let mut entity_transitions = HashMap::new();

        for new_entity in updated {
            let id = new_entity.entity_id;
            self.entities.replace_entity(new_entity);

            // Resolve per-entity behavior override
            let transition = self
                .entities
                .behavior(id)
                .cloned()
                .unwrap_or_else(|| default_transition.clone());
            let _ = entity_transitions.insert(id, transition);
        }

        if !entity_transitions.is_empty() {
            self.sync_scene_to_renderers(entity_transitions);
        }
    }

    /// Set the visibility of a specific entity.
    ///
    /// Hidden entities are excluded from rendering but remain in the scene.
    /// Forces a full scene sync.
    pub fn set_entity_visible(&mut self, id: u32, visible: bool) {
        if let Some(se) = self.entities.entity_mut(id) {
            if se.visible != visible {
                se.visible = visible;
                se.invalidate_render_cache();
                self.entities.force_dirty();
                self.sync_scene_to_renderers(HashMap::new());
            }
        }
    }

    /// Set per-residue scores for a specific entity.
    ///
    /// Scores drive color-by-score visualization. Pass `None` to clear.
    /// Forces a scene resync to update vertex colors.
    pub fn set_per_residue_scores(
        &mut self,
        id: u32,
        scores: Option<Vec<f64>>,
    ) {
        if let Some(se) = self.entities.entity_mut(id) {
            se.per_residue_scores = scores;
            se.invalidate_render_cache();
            self.entities.force_dirty();
            self.sync_scene_to_renderers(HashMap::new());
        }
    }

    /// Remove an entity from the engine and scene entirely.
    ///
    /// Forces a full scene resync.
    pub fn remove_entity(&mut self, id: u32) {
        self.entities.clear_behavior(id);
        if self.entities.remove_entity(id) {
            self.sync_scene_to_renderers(HashMap::new());
        }
    }
}
