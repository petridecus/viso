//! Scene Sync methods for VisoEngine

use std::collections::HashMap;

use foldit_conv::secondary_structure::SSType;
use glam::Vec3;

use super::VisoEngine;
use crate::animation::transition::Transition;
use crate::animation::SidechainAnimPositions;
use crate::renderer::geometry::backbone::{
    BackboneUpdateData, PreparedBackboneData,
};
use crate::renderer::geometry::ball_and_stick::PreparedBallAndStickData;
use crate::renderer::geometry::sidechain::SidechainView;
use crate::scene::{PreparedScene, SceneRequest};
use crate::util::score_color;

/// Input data for [`VisoEngine::update_from_aggregated`].
pub struct AggregatedSceneData<'a> {
    pub backbone_chains: &'a [Vec<Vec3>],
    pub sidechain: &'a SidechainView<'a>,
    pub sidechain_atom_names: &'a [String],
    pub all_positions: &'a [Vec3],
    pub fit_camera: bool,
    pub ss_types: Option<&'a [SSType]>,
}

impl VisoEngine {
    /// Update all renderers from aggregated scene data
    ///
    /// This is the main integration point for the Scene-based rendering model.
    /// Call this whenever structures are added, removed, or modified in the
    /// scene.
    pub fn update_from_aggregated(&mut self, data: &AggregatedSceneData) {
        // Calculate total residues from backbone chains (3 atoms per residue:
        // N, CA, C)
        let total_residues = crate::util::sheet_adjust::backbone_residue_count(
            data.backbone_chains,
        );

        // Ensure selection buffer has capacity for all residues (including new
        // structures)
        self.pick
            .selection
            .ensure_capacity(&self.context.device, total_residues);

        // Update backbone renderer (protein + NA)
        self.renderers.backbone.update(
            &self.context,
            &BackboneUpdateData {
                protein_chains: data.backbone_chains,
                na_chains: &[],
                ss_types: data.ss_types,
                geometry: &self.options.geometry,
            },
        );

        // Translate sidechains onto sheet surface (whole sidechain, not just
        // CA-CB bond)
        let offset_map = self.sheet_offset_map();
        let adjusted = crate::util::sheet_adjust::sheet_adjusted_view(
            data.sidechain,
            &offset_map,
        );

        self.renderers.sidechain.update(
            &self.context.device,
            &self.context.queue,
            &adjusted.as_view(),
        );

        // Update capsule picking bind group (buffer may have been reallocated)
        self.pick.groups.rebuild_capsule(
            &self.pick.picking,
            &self.context.device,
            &self.renderers.sidechain,
        );

        // Cache secondary structure types for double-click segment selection
        self.sc_cache.ss_types = data.ss_types.map_or_else(
            || Self::compute_ss_types(data.backbone_chains),
            <[SSType]>::to_vec,
        );

        // Cache atom names for lookup by name (used for band tracking during
        // animation)
        self.sc_cache.sidechain_atom_names = data.sidechain_atom_names.to_vec();

        // Fit camera if requested and we have positions
        if data.fit_camera && !data.all_positions.is_empty() {
            self.camera_controller.fit_to_positions(data.all_positions);
        }
    }

    /// Sync scene data to renderers with a global transition.
    ///
    /// All entities animate with the same transition (or snap if `None`).
    pub fn sync_scene_to_renderers(&mut self, transition: Option<Transition>) {
        if !self.scene.is_dirty() && transition.is_none() {
            return;
        }

        let entities = self.scene.per_entity_data();
        // Build entity_transitions: all entities get the same transition
        let entity_transitions = transition.map_or_else(HashMap::new, |t| {
            entities.iter().map(|e| (e.id, t.clone())).collect()
        });
        self.scene.mark_rendered();

        self.scene_processor.submit(SceneRequest::FullRebuild {
            entities,
            entity_transitions,
            display: self.options.display.clone(),
            colors: self.options.colors.clone(),
            geometry: self.options.geometry.clone(),
        });
    }

    /// Sync scene data to renderers with per-entity transitions.
    ///
    /// Entities in the map animate with their transition; all others snap.
    pub fn sync_scene_to_renderers_targeted(
        &mut self,
        entity_transitions: HashMap<u32, Transition>,
    ) {
        if !self.scene.is_dirty() && entity_transitions.is_empty() {
            return;
        }

        let entities = self.scene.per_entity_data();
        self.scene.mark_rendered();

        self.scene_processor.submit(SceneRequest::FullRebuild {
            entities,
            entity_transitions,
            display: self.options.display.clone(),
            colors: self.options.colors.clone(),
            geometry: self.options.geometry.clone(),
        });
    }

    /// Upload prepared scene geometry to GPU renderers.
    fn upload_prepared_to_gpu(
        &mut self,
        prepared: &PreparedScene,
        animating: bool,
        suppress_sidechains: bool,
    ) {
        if animating {
            self.renderers.backbone.update_metadata(
                prepared.backbone_chains.clone(),
                prepared.na_chains.clone(),
                prepared.ss_types.clone(),
            );
        } else {
            self.renderers.backbone.apply_prepared(
                &self.context.device,
                &self.context.queue,
                PreparedBackboneData {
                    vertices: &prepared.backbone.vertices,
                    tube_indices: &prepared.backbone.tube_indices,
                    ribbon_indices: &prepared.backbone.ribbon_indices,
                    tube_index_count: prepared.backbone.tube_index_count,
                    ribbon_index_count: prepared.backbone.ribbon_index_count,
                    sheet_offsets: prepared.backbone.sheet_offsets.clone(),
                    chain_ranges: prepared.backbone.chain_ranges.clone(),
                    cached_chains: prepared.backbone_chains.clone(),
                    cached_na_chains: prepared.na_chains.clone(),
                    ss_override: prepared.ss_types.clone(),
                },
            );
            if !suppress_sidechains {
                let _ = self.renderers.sidechain.apply_prepared(
                    &self.context.device,
                    &self.context.queue,
                    &prepared.sidechain_instances,
                    prepared.sidechain_instance_count,
                );
            }
        }

        self.renderers.ball_and_stick.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &PreparedBallAndStickData {
                sphere_bytes: &prepared.bns.sphere_instances,
                sphere_count: prepared.bns.sphere_count,
                capsule_bytes: &prepared.bns.capsule_instances,
                capsule_count: prepared.bns.capsule_count,
            },
        );

        self.renderers.nucleic_acid.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &prepared.na,
        );

        // Store the pick map for typed target resolution
        self.pick.pick_map = Some(prepared.pick_map.clone());

        // Recreate picking bind groups (buffers may have been reallocated)
        self.pick.groups.rebuild_all(
            &self.pick.picking,
            &self.context.device,
            &self.renderers.sidechain,
            &self.renderers.ball_and_stick,
        );
    }

    /// Apply any pending scene data from the background SceneProcessor.
    ///
    /// Called every frame from the main loop. If the background thread has
    /// finished generating geometry, this uploads it to the GPU (<1ms) and
    /// sets up animation.
    pub fn apply_pending_scene(&mut self) {
        let Some(prepared) = self.scene_processor.try_recv_scene() else {
            return;
        };

        // Store entity residue ranges on both engine and Scene
        self.entity_ranges
            .clone_from(&prepared.entity_residue_ranges);
        self.scene
            .set_entity_residue_ranges(prepared.entity_residue_ranges.clone());

        let animating = !prepared.entity_transitions.is_empty();

        if animating {
            self.setup_per_entity_animation(&prepared);
            self.submit_animation_frame();
        } else {
            self.snap_from_prepared(&prepared);
        }

        let suppress_sidechains = prepared
            .entity_transitions
            .values()
            .any(|t| t.suppress_initial_sidechains);
        self.upload_prepared_to_gpu(&prepared, animating, suppress_sidechains);
    }

    /// Set up per-entity animation from prepared scene data.
    ///
    /// For each entity with a transition, dispatches to
    /// `animator.animate_entity()` so each entity gets its own runner.
    /// Entities without transitions are not animated.
    fn setup_per_entity_animation(&mut self, prepared: &PreparedScene) {
        let new_backbone = &prepared.backbone_chains;

        // Extract sidechain data for per-entity dispatch
        let ca_positions =
            foldit_conv::render::backbone::ca_positions_from_chains(
                new_backbone,
            );
        let sidechain_positions = prepared.sidechain.positions();
        let sidechain_residue_indices = prepared.sidechain.residue_indices();

        // Update sidechain cache
        self.sc_cache
            .update_from_sidechain_atoms(&prepared.sidechain);

        // Cache secondary structure types
        if let Some(ref ss) = prepared.ss_types {
            self.sc_cache.ss_types.clone_from(ss);
        } else {
            self.sc_cache.ss_types = Self::compute_ss_types(new_backbone);
        }

        // Cache per-residue colors (derived from scores by scene processor)
        self.sc_cache
            .per_residue_colors
            .clone_from(&prepared.per_residue_colors);

        // Dispatch per-entity animation with sidechain data
        for range in &prepared.entity_residue_ranges {
            let transition = prepared
                .entity_transitions
                .get(&range.entity_id)
                .cloned()
                .unwrap_or_default();

            let entity_sc = Self::extract_entity_sidechain(
                &sidechain_positions,
                &sidechain_residue_indices,
                &ca_positions,
                range,
                prepared.entity_transitions.get(&range.entity_id),
            );

            self.animator.animate_entity(
                range,
                new_backbone,
                &transition,
                entity_sc,
            );
        }

        // Ensure selection/color buffers have capacity and update colors
        let total_residues =
            crate::util::sheet_adjust::backbone_residue_count(new_backbone);
        self.pick
            .selection
            .ensure_capacity(&self.context.device, total_residues);
        self.pick
            .residue_colors
            .ensure_capacity(&self.context.device, total_residues);

        // Animate colors to new target
        let colors = self.compute_per_residue_colors(new_backbone);
        self.pick.residue_colors.set_target_colors(&colors);
        self.sc_cache.per_residue_colors = Some(colors);
    }

    /// Snap update from prepared scene data (no animation).
    fn snap_from_prepared(&mut self, prepared: &PreparedScene) {
        // Write at-rest backbone to Scene's visual state
        self.scene
            .update_visual_positions(prepared.backbone_chains.clone());

        self.sc_cache
            .update_from_sidechain_atoms(&prepared.sidechain);

        if let Some(ref ss) = prepared.ss_types {
            self.sc_cache.ss_types.clone_from(ss);
        } else {
            self.sc_cache.ss_types =
                Self::compute_ss_types(&prepared.backbone_chains);
        }

        self.sc_cache
            .per_residue_colors
            .clone_from(&prepared.per_residue_colors);

        let total_residues = crate::util::sheet_adjust::backbone_residue_count(
            &prepared.backbone_chains,
        );
        self.pick
            .selection
            .ensure_capacity(&self.context.device, total_residues);
        self.pick
            .residue_colors
            .ensure_capacity(&self.context.device, total_residues);

        let colors = self.compute_per_residue_colors(&prepared.backbone_chains);
        self.pick
            .residue_colors
            .set_colors_immediate(&self.context.queue, &colors);
        self.sc_cache.per_residue_colors = Some(colors);
    }

    /// Compute a rainbow chain color for parameter `t` in [0, 1].
    pub(crate) fn chain_color(t: f32) -> [f32; 3] {
        let hue = (1.0 - t) * 240.0;
        let sector = hue / 60.0;
        let frac = sector - sector.floor();
        match sector as u32 {
            0 => [1.0, frac, 0.0],       // red → yellow
            1 => [1.0 - frac, 1.0, 0.0], // yellow → green
            2 => [0.0, 1.0, frac],       // green → cyan
            3 => [0.0, 1.0 - frac, 1.0], // cyan → blue
            _ => [0.0, 0.0, 1.0],        // blue
        }
    }

    /// Compute per-residue colors based on the current backbone_color_mode.
    pub(crate) fn compute_per_residue_colors(
        &self,
        backbone_chains: &[Vec<Vec3>],
    ) -> Vec<[f32; 3]> {
        use crate::options::BackboneColorMode;
        let residue_count = self.sc_cache.ss_types.len().max(1);
        match self.options.display.backbone_color_mode {
            BackboneColorMode::Score | BackboneColorMode::ScoreRelative => {
                let mut all_scores: Vec<f64> = Vec::new();
                let mut has_any = false;
                for entity in self.scene.entities() {
                    if let Some(ref scores) = entity.per_residue_scores {
                        has_any = true;
                        all_scores.extend_from_slice(scores);
                    }
                }
                if !has_any {
                    return vec![[0.5, 0.5, 0.5]; residue_count];
                }
                match self.options.display.backbone_color_mode {
                    BackboneColorMode::Score => {
                        score_color::per_residue_score_colors(&all_scores)
                    }
                    _ => score_color::per_residue_score_colors_relative(
                        &all_scores,
                    ),
                }
            }
            BackboneColorMode::SecondaryStructure => {
                if self.sc_cache.ss_types.is_empty() {
                    vec![[0.5, 0.5, 0.5]; residue_count]
                } else {
                    self.sc_cache.ss_types.iter().map(SSType::color).collect()
                }
            }
            BackboneColorMode::Chain => {
                let num_chains = backbone_chains.len();
                if num_chains == 0 {
                    return vec![[0.5, 0.5, 0.5]; residue_count];
                }
                let mut colors = Vec::with_capacity(residue_count);
                for (chain_idx, chain) in backbone_chains.iter().enumerate() {
                    let t = if num_chains > 1 {
                        chain_idx as f32 / (num_chains - 1) as f32
                    } else {
                        0.0
                    };
                    let color = Self::chain_color(t);
                    let n_residues = chain.len() / 3;
                    for _ in 0..n_residues {
                        colors.push(color);
                    }
                }
                colors
            }
        }
    }

    /// Extract sidechain animation positions for a single entity.
    ///
    /// Returns `None` if the entity has no sidechain atoms in its range,
    /// or if `transition` is `None` (entity has no active transition).
    fn extract_entity_sidechain(
        all_positions: &[Vec3],
        all_residue_indices: &[u32],
        all_ca: &[Vec3],
        range: &crate::scene::EntityResidueRange,
        transition: Option<&Transition>,
    ) -> Option<SidechainAnimPositions> {
        let transition = transition?;
        let res_start = range.start as usize;
        let res_end = range.end() as usize;
        let collapse_to_ca = transition.allows_size_change;

        let mut start = Vec::new();
        let mut target = Vec::new();

        for (i, &res_idx) in all_residue_indices.iter().enumerate() {
            let r = res_idx as usize;
            if !(res_start..res_end).contains(&r) {
                continue;
            }
            let Some(&pos) = all_positions.get(i) else {
                continue;
            };
            target.push(pos);
            if collapse_to_ca {
                start.push(all_ca.get(r).copied().unwrap_or(Vec3::ZERO));
            } else {
                start.push(pos);
            }
        }

        if target.is_empty() {
            return None;
        }

        Some(SidechainAnimPositions { start, target })
    }

    /// Check per-chain LOD tiers and submit a background remesh if any
    /// chain's tier has changed.
    pub(crate) fn check_and_submit_lod(&mut self) {
        let camera_eye = self.camera_controller.camera.eye;
        let per_chain_tiers: Vec<u8> = self
            .renderers
            .backbone
            .chain_ranges()
            .iter()
            .map(|r| {
                crate::options::select_chain_lod_tier(
                    r.bounding_center,
                    camera_eye,
                )
            })
            .collect();
        if per_chain_tiers != self.renderers.backbone.cached_lod_tiers() {
            self.renderers
                .backbone
                .set_cached_lod_tiers(per_chain_tiers);
            self.submit_per_chain_lod_remesh(camera_eye);
        }
    }

    /// Submit an animation frame to the background thread for mesh generation.
    pub(crate) fn submit_animation_frame(&self) {
        let visual_backbone = self.animator.get_backbone();
        let has_sidechains =
            !self.sc_cache.target_sidechain_positions.is_empty()
                && self.animator.should_include_sidechains();
        self.submit_animation_frame_with_backbone(
            visual_backbone,
            has_sidechains,
        );
    }

    /// Submit an animation frame with explicit backbone chains.
    pub(crate) fn submit_animation_frame_with_backbone(
        &self,
        backbone_chains: Vec<Vec<Vec3>>,
        include_sidechains: bool,
    ) {
        let sidechains = if include_sidechains {
            let interpolated_positions = self
                .animator
                .get_aggregated_sidechain_positions()
                .unwrap_or_else(|| {
                    self.sc_cache.target_sidechain_positions.clone()
                });
            let interpolated_bonds =
                self.sc_cache.interpolated_backbone_bonds(&self.animator);
            Some(self.sc_cache.to_interpolated_sidechain_atoms(
                &interpolated_positions,
                &interpolated_bonds,
            ))
        } else {
            None
        };

        let ss_types = if self.sc_cache.ss_types.is_empty() {
            None
        } else {
            Some(self.sc_cache.ss_types.clone())
        };

        self.scene_processor.submit(SceneRequest::AnimationFrame {
            backbone_chains,
            na_chains: self.renderers.backbone.cached_na_chains().to_vec(),
            sidechains,
            ss_types,
            per_residue_colors: self.sc_cache.per_residue_colors.clone(),
            geometry: self.options.geometry.clone(),
            per_chain_lod: None,
        });
    }

    /// Submit a backbone-only remesh with per-chain LOD to the background
    /// thread. Each chain gets its own `(spr, csv)` based on its distance
    /// from the camera. No sidechains — they don't change with LOD.
    pub(crate) fn submit_per_chain_lod_remesh(&self, camera_eye: Vec3) {
        use crate::options::{lod_scaled, select_chain_lod_tier};

        // Use clamped geometry as the base for LOD scaling
        let total_residues = crate::util::sheet_adjust::backbone_residue_count(
            self.renderers.backbone.cached_chains(),
        ) + self
            .renderers
            .backbone
            .cached_na_chains()
            .iter()
            .map(Vec::len)
            .sum::<usize>();
        let base_geo =
            self.options.geometry.clamped_for_residues(total_residues);
        let max_spr = base_geo.segments_per_residue;
        let max_csv = base_geo.cross_section_verts;

        let per_chain_lod: Vec<(usize, usize)> = self
            .renderers
            .backbone
            .chain_ranges()
            .iter()
            .map(|r| {
                let tier = select_chain_lod_tier(r.bounding_center, camera_eye);
                lod_scaled(max_spr, max_csv, tier)
            })
            .collect();

        let ss_types = if self.sc_cache.ss_types.is_empty() {
            None
        } else {
            Some(self.sc_cache.ss_types.clone())
        };

        self.scene_processor.submit(SceneRequest::AnimationFrame {
            backbone_chains: self.renderers.backbone.cached_chains().to_vec(),
            na_chains: self.renderers.backbone.cached_na_chains().to_vec(),
            sidechains: None,
            ss_types,
            per_residue_colors: self.sc_cache.per_residue_colors.clone(),
            geometry: base_geo,
            per_chain_lod: Some(per_chain_lod),
        });
    }

    /// Apply any pending animation frame from the background thread.
    pub(crate) fn apply_pending_animation(&mut self) {
        let Some(prepared) = self.scene_processor.try_recv_animation() else {
            return;
        };

        self.renderers.backbone.apply_mesh(
            &self.context.device,
            &self.context.queue,
            prepared.backbone,
        );

        if let Some(ref instances) = prepared.sidechain_instances {
            let reallocated = self.renderers.sidechain.apply_prepared(
                &self.context.device,
                &self.context.queue,
                instances,
                prepared.sidechain_instance_count,
            );
            if reallocated {
                self.pick.groups.rebuild_capsule(
                    &self.pick.picking,
                    &self.context.device,
                    &self.renderers.sidechain,
                );
            }
        }
    }

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
            self.entities.iter_mut().find(|e| e.entity_id == id)
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

        // 2. Update Scene (derived copy)
        self.scene.update_entity_protein_coords(id, coords);

        // 3. Look up per-entity behavior override
        let effective_transition = self
            .entity_behaviors
            .get(&id)
            .cloned()
            .unwrap_or(transition);

        // 4. Sync with per-entity transition
        let mut entity_transitions = HashMap::new();
        let _ = entity_transitions.insert(id, effective_transition);
        self.sync_scene_to_renderers_targeted(entity_transitions);
    }
}
