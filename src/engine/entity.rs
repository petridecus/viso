//! Entity management: loading, updating, constraint visualization, behavior.

use std::collections::HashMap;

use foldit_conv::secondary_structure::SSType;
use foldit_conv::types::entity::MoleculeEntity;
use glam::Vec3;

use super::command::{
    AtomRef, BandInfo, BandTarget, PullInfo, ResolvedBand, ResolvedPull,
};
use super::VisoEngine;
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
        let _ = self.entities.behaviors.insert(entity_id, transition);
    }

    /// Clear a per-entity behavior override, reverting to default (smooth).
    pub fn clear_entity_behavior(&mut self, entity_id: u32) {
        let _ = self.entities.behaviors.remove(&entity_id);
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
        // Store canonical copy on the engine (source of truth)
        self.entities.source.clone_from(&entities);

        let ids = self.entities.add_entities(entities);
        if fit_camera {
            // Sync immediately so entity data is available for camera fit
            let snap_transitions: HashMap<u32, Transition> =
                ids.iter().map(|&id| (id, Transition::snap())).collect();
            self.sync_scene_to_renderers(snap_transitions);
            let positions = self.entities.all_positions();
            if !positions.is_empty() {
                self.camera_controller.fit_to_positions(&positions);
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
        self.gpu.renderers
            .backbone
            .set_ss_override(Some(ss_types.to_vec()));
        let camera_eye = self.camera_controller.camera.eye;
        self.submit_per_chain_lod_remesh(camera_eye);
    }

    /// Replace the current set of constraint bands.
    ///
    /// Bands use structural references ([`AtomRef`]) and are resolved to
    /// world-space positions each frame, so they auto-track animated atoms.
    pub fn update_bands(&mut self, bands: Vec<BandInfo>) {
        self.band_specs = bands;
        self.resolve_and_render_constraints();
    }

    /// Set or clear the active pull constraint.
    ///
    /// Pulls use a structural reference for the atom and a screen-space
    /// target. The engine resolves positions each frame.
    pub fn update_pull(&mut self, pull: Option<PullInfo>) {
        self.pull_spec = pull;
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
        // Bands
        let resolved_bands: Vec<ResolvedBand> = self
            .band_specs
            .iter()
            .filter_map(|b| self.resolve_band(b))
            .collect();
        self.gpu.renderers.band.update(
            &self.gpu.context.device,
            &self.gpu.context.queue,
            &resolved_bands,
            Some(&self.options.colors),
        );

        // Pull
        let resolved_pull =
            self.pull_spec.as_ref().and_then(|p| self.resolve_pull(p));
        self.gpu.renderers.pull.update(
            &self.gpu.context.device,
            &self.gpu.context.queue,
            resolved_pull.as_ref(),
        );
    }

    /// Resolve a single band spec to world-space positions.
    fn resolve_band(&self, band: &BandInfo) -> Option<ResolvedBand> {
        let endpoint_a = self.resolve_atom_ref(&band.anchor_a)?;
        let endpoint_b = match &band.anchor_b {
            BandTarget::Atom(atom) => self.resolve_atom_ref(atom)?,
            BandTarget::Position(pos) => *pos,
        };
        let is_space_pull = matches!(band.anchor_b, BandTarget::Position(_));

        Some(ResolvedBand {
            endpoint_a,
            endpoint_b,
            is_disabled: band.is_disabled,
            strength: band.strength,
            target_length: band.target_length,
            residue_idx: band.anchor_a.residue,
            is_space_pull,
            band_type: band.band_type,
            from_script: band.from_script,
        })
    }

    /// Resolve a pull spec to world-space positions.
    fn resolve_pull(&self, pull: &PullInfo) -> Option<ResolvedPull> {
        let atom_pos = self.resolve_atom_ref(&pull.atom)?;
        let target_pos = self.camera_controller.screen_to_world_at_depth(
            glam::Vec2::new(pull.screen_target.0, pull.screen_target.1),
            glam::UVec2::new(
                self.gpu.context.config.width,
                self.gpu.context.config.height,
            ),
            atom_pos,
        );

        Some(ResolvedPull {
            atom_pos,
            target_pos,
            residue_idx: pull.atom.residue,
        })
    }

    /// Resolve an [`AtomRef`] to a world-space position from Scene data.
    ///
    /// Uses interpolated visual positions during animation so constraints
    /// track animated atoms.
    fn resolve_atom_ref(&self, atom: &AtomRef) -> Option<Vec3> {
        let name = atom.atom_name.as_str();

        // Backbone atoms: N, CA, C — look up in visual_backbone_chains
        if name == "N" || name == "CA" || name == "C" {
            let offset = match name {
                "N" => 0,
                "CA" => 1,
                "C" => 2,
                _ => return None,
            };
            let chains = if self.visual.backbone_chains.is_empty() {
                // Before first animation, fall back to renderer cache
                self.gpu.renderers.backbone.cached_chains()
            } else {
                &self.visual.backbone_chains
            };
            let mut current_idx: u32 = 0;
            for chain in chains {
                let residues_in_chain = (chain.len() / 3) as u32;
                if atom.residue < current_idx + residues_in_chain {
                    let local = (atom.residue - current_idx) as usize;
                    return chain.get(local * 3 + offset).copied();
                }
                current_idx += residues_in_chain;
            }
            return None;
        }

        // Sidechain atoms — look up in visual_sidechain_positions
        let positions = if self.visual.sidechain_positions.is_empty() {
            &self.topology.sidechain_topology.target_positions
        } else {
            &self.visual.sidechain_positions
        };
        for (i, (res_idx, sc_name)) in self
            .topology
            .sidechain_topology
            .residue_indices
            .iter()
            .zip(self.topology.sidechain_topology.atom_names.iter())
            .enumerate()
        {
            if *res_idx == atom.residue && sc_name == name {
                return positions.get(i).copied();
            }
        }

        None
    }
}

// ── Entity updates ──

impl VisoEngine {
    /// Update protein coords for a specific entity.
    ///
    /// Updates the engine's source-of-truth entities first, then derives
    /// Scene from them. Uses the entity's per-entity behavior override if
    /// set, otherwise falls back to the provided transition.
    pub fn update_entity_coords(
        &mut self,
        id: u32,
        coords: foldit_conv::types::coords::Coords,
        transition: Transition,
    ) {
        // 1. Update source-of-truth on the engine
        if let Some(entity) =
            self.entities.source.iter_mut().find(|e| e.entity_id == id)
        {
            let mut entities = vec![entity.clone()];
            foldit_conv::types::assembly::update_protein_entities(
                &mut entities,
                coords.clone(),
            );
            if let Some(updated) = entities.into_iter().next() {
                *entity = updated;
            }
        }

        // 2. Update scene entities (rendering copy)
        self.entities.update_entity_protein_coords(id, coords);

        // 3. Look up per-entity behavior override
        let effective_transition = self
            .entities
            .behaviors
            .get(&id)
            .cloned()
            .unwrap_or(transition);

        // 4. Sync with per-entity transition
        let mut entity_transitions = HashMap::new();
        let _ = entity_transitions.insert(id, effective_transition);
        self.sync_scene_to_renderers(entity_transitions);
    }

    /// Replace one or more entities with new `MoleculeEntity` data.
    ///
    /// Each entity is matched by `entity_id`. The engine's source-of-truth
    /// and Scene are both updated, then a targeted sync is triggered for all
    /// changed entities. Per-entity behavior overrides are used when set,
    /// otherwise `default_transition` is applied.
    pub fn update_entities(
        &mut self,
        updated: Vec<MoleculeEntity>,
        default_transition: &Transition,
    ) {
        let mut entity_transitions = HashMap::new();

        for new_entity in updated {
            let id = new_entity.entity_id;

            // Update engine source-of-truth
            if let Some(slot) =
                self.entities.source.iter_mut().find(|e| e.entity_id == id)
            {
                *slot = new_entity.clone();
            }

            // Update scene entities (rendering copy)
            self.entities.replace_entity(new_entity);

            // Resolve per-entity behavior override
            let transition = self
                .entities
                .behaviors
                .get(&id)
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
    /// Removes from both the engine's source-of-truth and the Scene.
    /// Forces a full scene resync.
    pub fn remove_entity(&mut self, id: u32) {
        self.entities.source.retain(|e| e.entity_id != id);
        let _ = self.entities.behaviors.remove(&id);
        if self.entities.remove_entity(id) {
            self.sync_scene_to_renderers(HashMap::new());
        }
    }
}
