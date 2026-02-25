//! Scene Sync methods for ProteinRenderEngine

use std::collections::{HashMap, HashSet};

use foldit_conv::secondary_structure::SSType;
use glam::Vec3;

use super::ProteinRenderEngine;
use crate::animation::transition::Transition;
use crate::renderer::geometry::backbone::PreparedBackboneData;
use crate::renderer::geometry::ball_and_stick::PreparedBallAndStickData;
use crate::renderer::geometry::sidechain::SidechainView;
use crate::scene::{PreparedScene, SceneRequest};
use crate::util::score_color;

impl ProteinRenderEngine {
    /// Update all renderers from aggregated scene data
    ///
    /// This is the main integration point for the Scene-based rendering model.
    /// Call this whenever structures are added, removed, or modified in the
    /// scene.
    pub fn update_from_aggregated(
        &mut self,
        backbone_chains: &[Vec<Vec3>],
        sidechain: &SidechainView,
        sidechain_atom_names: &[String],
        all_positions: &[Vec3],
        fit_camera: bool,
        ss_types: Option<&[SSType]>,
    ) {
        // Calculate total residues from backbone chains (3 atoms per residue:
        // N, CA, C)
        let total_residues =
            crate::util::sheet_adjust::backbone_residue_count(backbone_chains);

        // Ensure selection buffer has capacity for all residues (including new
        // structures)
        self.selection_buffer
            .ensure_capacity(&self.context.device, total_residues);

        // Update backbone renderer (protein + NA)
        self.backbone_renderer.update(
            &self.context.device,
            &self.context.queue,
            backbone_chains,
            &[], // NA chains not available in this path
            ss_types,
            &self.options.geometry,
        );

        // Translate sidechains onto sheet surface (whole sidechain, not just
        // CA-CB bond)
        let offset_map = self.sheet_offset_map();
        let adjusted =
            crate::util::sheet_adjust::sheet_adjusted_view(sidechain, &offset_map);

        self.sidechain_renderer.update(
            &self.context.device,
            &self.context.queue,
            &adjusted.as_view(),
        );

        // Update capsule picking bind group (buffer may have been reallocated)
        self.picking_groups.rebuild_capsule(
            &self.picking,
            &self.context.device,
            &self.sidechain_renderer,
        );

        // Cache secondary structure types for double-click segment selection
        self.sc.cached_ss_types = ss_types.map_or_else(
            || Self::compute_ss_types(backbone_chains),
            <[SSType]>::to_vec,
        );

        // Cache atom names for lookup by name (used for band tracking during
        // animation)
        self.sc.cached_sidechain_atom_names = sidechain_atom_names.to_vec();

        // Fit camera if requested and we have positions
        if fit_camera && !all_positions.is_empty() {
            self.camera_controller.fit_to_positions(all_positions);
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

    /// Apply any pending scene data from the background SceneProcessor.
    ///
    /// Called every frame from the main loop. If the background thread has
    /// finished generating geometry, this uploads it to the GPU (<1ms) and
    /// sets up animation.
    pub fn apply_pending_scene(&mut self) {
        let Some(prepared) = self.scene_processor.try_recv_scene() else {
            return;
        };

        // Triple buffer automatically returns only the latest result,
        // so no drain loop needed — stale intermediates are skipped.

        // Animation target setup or snap update (fast: array copies + animator)
        let dominant_transition = prepared.entity_transitions.values().next();
        if prepared.entity_transitions.is_empty() {
            self.snap_from_prepared(&prepared);
        } else {
            let transition = dominant_transition.cloned().unwrap_or_default();
            self.setup_animation_targets_from_prepared(&prepared, &transition);

            // Snap residues for entities NOT in entity_transitions
            let active: HashSet<u32> =
                prepared.entity_transitions.keys().copied().collect();
            self.animator.snap_entities_without_action(
                &prepared.entity_residue_ranges,
                &active,
            );

            // Remove non-targeted entity residues from the AnimationRunner
            // so apply_to_state doesn't overwrite their snapped backbone state
            self.animator.remove_non_targeted_from_runner(
                &prepared.entity_residue_ranges,
                &active,
            );

            // Also snap engine-level sidechain start positions for non-targeted
            // entities
            self.snap_non_targeted_sidechains(
                &prepared.entity_residue_ranges,
                &active,
            );

            // Immediately submit an animation frame so the background thread
            // starts generating an interpolated mesh right away.  Since we skip
            // vertex uploads from the FullRebuild during animation (to avoid a
            // one-frame jump), this ensures the next render picks up a
            // correctly interpolated mesh with minimal latency.
            self.submit_animation_frame();
        }

        // GPU uploads — each is <0.2ms
        // When animating, skip uploading vertex data for backbone renderers
        // (tube + ribbon + sidechains) to avoid a one-frame jump to target
        // positions. The animation frame path will provide interpolated meshes.
        // We still update metadata (chains, SS types) so animation frames
        // generate correct topology.
        let animating = !prepared.entity_transitions.is_empty();
        if animating {
            self.backbone_renderer.update_metadata(
                prepared.backbone_chains.clone(),
                prepared.na_chains.clone(),
                prepared.ss_types.clone(),
            );
        } else {
            self.backbone_renderer.apply_prepared(
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
            let suppress_sidechains = dominant_transition
                .is_some_and(|t| t.suppress_initial_sidechains);
            if !suppress_sidechains {
                let _ = self.sidechain_renderer.apply_prepared(
                    &self.context.device,
                    &self.context.queue,
                    &prepared.sidechain_instances,
                    prepared.sidechain_instance_count,
                );
            }
        }

        self.ball_and_stick_renderer.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &PreparedBallAndStickData {
                sphere_bytes: &prepared.bns.sphere_instances,
                sphere_count: prepared.bns.sphere_count,
                capsule_bytes: &prepared.bns.capsule_instances,
                capsule_count: prepared.bns.capsule_count,
                picking_bytes: &prepared.bns.picking_capsules,
                picking_count: prepared.bns.picking_count,
            },
        );

        self.nucleic_acid_renderer.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &prepared.na,
        );

        // Recreate picking bind groups (buffers may have been reallocated)
        self.picking_groups.rebuild_all(
            &self.picking,
            &self.context.device,
            &self.sidechain_renderer,
            &self.ball_and_stick_renderer,
        );
    }

    /// Snap sidechain start positions to target for entities that are NOT
    /// part of the current transition, so they don't animate.
    fn snap_non_targeted_sidechains(
        &mut self,
        entity_residue_ranges: &[crate::scene::EntityResidueRange],
        active: &HashSet<u32>,
    ) {
        for range in entity_residue_ranges {
            if active.contains(&range.entity_id) {
                continue;
            }
            let res_start = range.start as usize;
            let res_end = range.end() as usize;
            for (i, &res_idx) in
                self.sc.cached_sidechain_residue_indices.iter().enumerate()
            {
                let r = res_idx as usize;
                if r < res_start
                    || r >= res_end
                    || i >= self.sc.start_sidechain_positions.len()
                    || i >= self.sc.target_sidechain_positions.len()
                {
                    continue;
                }
                self.sc.start_sidechain_positions[i] =
                    self.sc.target_sidechain_positions[i];
            }
            for (j, &(_, cb_idx)) in
                self.sc.target_backbone_sidechain_bonds.iter().enumerate()
            {
                let res_idx = self
                    .sc
                    .cached_sidechain_residue_indices
                    .get(cb_idx as usize)
                    .copied()
                    .unwrap_or(u32::MAX) as usize;
                if res_idx < res_start
                    || res_idx >= res_end
                    || j >= self.sc.start_backbone_sidechain_bonds.len()
                {
                    continue;
                }
                self.sc.start_backbone_sidechain_bonds[j] =
                    self.sc.target_backbone_sidechain_bonds[j];
            }
        }
    }

    /// Set up animation targets from prepared scene data.
    fn setup_animation_targets_from_prepared(
        &mut self,
        prepared: &PreparedScene,
        transition: &Transition,
    ) {
        let new_backbone = &prepared.backbone_chains;
        let sidechain_positions = prepared.sidechain.positions();
        let sidechain_residue_indices = prepared.sidechain.residue_indices();

        // Capture current VISUAL positions as start (for smooth preemption)
        if self.sc.target_sidechain_positions.len() == sidechain_positions.len()
        {
            if self.animator.is_animating()
                && self.animator.has_sidechain_data()
            {
                self.sc.start_sidechain_positions =
                    self.animator.get_sidechain_positions();
                let ctx = self.animator.interpolation_context();
                self.sc.start_backbone_sidechain_bonds = self
                    .sc
                    .start_backbone_sidechain_bonds
                    .iter()
                    .zip(self.sc.target_backbone_sidechain_bonds.iter())
                    .map(|((start_pos, idx), (target_pos, _))| {
                        let pos = *start_pos
                            + (*target_pos - *start_pos) * ctx.eased_t;
                        (pos, *idx)
                    })
                    .collect();
            } else {
                self.sc
                    .start_sidechain_positions
                    .clone_from(&self.sc.target_sidechain_positions);
                self.sc
                    .start_backbone_sidechain_bonds
                    .clone_from(&self.sc.target_backbone_sidechain_bonds);
            }
        } else {
            let ca_positions =
                foldit_conv::render::backbone::ca_positions_from_chains(
                    new_backbone,
                );
            self.sc.start_sidechain_positions = sidechain_residue_indices
                .iter()
                .map(|&ri| {
                    ca_positions.get(ri as usize).copied().unwrap_or(Vec3::ZERO)
                })
                .collect();
            self.sc
                .start_backbone_sidechain_bonds
                .clone_from(&prepared.sidechain.backbone_bonds);
        }

        // Extract CA positions for sidechain collapse animation
        let ca_positions =
            foldit_conv::render::backbone::ca_positions_from_chains(
                new_backbone,
            );

        // Pass sidechain data to animator FIRST (before set_target)
        self.animator.set_sidechain_target_with_transition(
            &sidechain_positions,
            &sidechain_residue_indices,
            &ca_positions,
            Some(transition),
        );

        // Set new targets and cached data
        self.sc
            .update_cached_from_sidechain_atoms(&prepared.sidechain);

        // Cache secondary structure types
        if let Some(ref ss) = prepared.ss_types {
            self.sc.cached_ss_types.clone_from(ss);
        } else {
            self.sc.cached_ss_types = Self::compute_ss_types(new_backbone);
        }

        // Cache per-residue colors (derived from scores by scene processor)
        self.sc
            .cached_per_residue_colors
            .clone_from(&prepared.per_residue_colors);

        // Set backbone target (starts the animation)
        self.animator.set_target(new_backbone, transition);

        // Ensure selection/color buffers have capacity and update colors
        let total_residues =
            crate::util::sheet_adjust::backbone_residue_count(new_backbone);
        self.selection_buffer
            .ensure_capacity(&self.context.device, total_residues);
        self.residue_color_buffer
            .ensure_capacity(&self.context.device, total_residues);

        // Animate colors to new target
        let colors = self.compute_per_residue_colors(new_backbone);
        self.residue_color_buffer.set_target_colors(&colors);
        self.sc.cached_per_residue_colors = Some(colors);
    }

    /// Snap update from prepared scene data (no animation).
    fn snap_from_prepared(&mut self, prepared: &PreparedScene) {
        self.sc
            .update_cached_from_sidechain_atoms(&prepared.sidechain);
        self.sc.snap_positions();

        if let Some(ref ss) = prepared.ss_types {
            self.sc.cached_ss_types.clone_from(ss);
        } else {
            self.sc.cached_ss_types =
                Self::compute_ss_types(&prepared.backbone_chains);
        }

        self.sc
            .cached_per_residue_colors
            .clone_from(&prepared.per_residue_colors);

        let total_residues = crate::util::sheet_adjust::backbone_residue_count(
            &prepared.backbone_chains,
        );
        self.selection_buffer
            .ensure_capacity(&self.context.device, total_residues);
        self.residue_color_buffer
            .ensure_capacity(&self.context.device, total_residues);

        let colors = self.compute_per_residue_colors(&prepared.backbone_chains);
        self.residue_color_buffer
            .set_colors_immediate(&self.context.queue, &colors);
        self.sc.cached_per_residue_colors = Some(colors);
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
        let residue_count = self.sc.cached_ss_types.len().max(1);
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
                if self.sc.cached_ss_types.is_empty() {
                    vec![[0.5, 0.5, 0.5]; residue_count]
                } else {
                    self.sc.cached_ss_types.iter().map(SSType::color).collect()
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

    /// Submit an animation frame to the background thread for mesh generation.
    pub(crate) fn submit_animation_frame(&self) {
        let visual_backbone = self.animator.get_backbone();
        let has_sidechains = self.animator.has_sidechain_data()
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
            let interpolated_positions =
                self.animator.get_sidechain_positions();
            let interpolated_bonds =
                self.sc.interpolated_backbone_bonds(&self.animator);
            Some(self.sc.to_interpolated_sidechain_atoms(
                &interpolated_positions,
                &interpolated_bonds,
            ))
        } else {
            None
        };

        let ss_types = if self.sc.cached_ss_types.is_empty() {
            None
        } else {
            Some(self.sc.cached_ss_types.clone())
        };

        self.scene_processor.submit(SceneRequest::AnimationFrame {
            backbone_chains,
            na_chains: self.backbone_renderer.cached_na_chains().to_vec(),
            sidechains,
            ss_types,
            per_residue_colors: self.sc.cached_per_residue_colors.clone(),
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
        let total_residues =
            crate::util::sheet_adjust::backbone_residue_count(
                self.backbone_renderer.cached_chains(),
            ) + self
                .backbone_renderer
                .cached_na_chains()
                .iter()
                .map(Vec::len)
                .sum::<usize>();
        let base_geo =
            self.options.geometry.clamped_for_residues(total_residues);
        let max_spr = base_geo.segments_per_residue;
        let max_csv = base_geo.cross_section_verts;

        let per_chain_lod: Vec<(usize, usize)> = self
            .backbone_renderer
            .chain_ranges()
            .iter()
            .map(|r| {
                let tier = select_chain_lod_tier(r.bounding_center, camera_eye);
                lod_scaled(max_spr, max_csv, tier)
            })
            .collect();

        let ss_types = if self.sc.cached_ss_types.is_empty() {
            None
        } else {
            Some(self.sc.cached_ss_types.clone())
        };

        self.scene_processor.submit(SceneRequest::AnimationFrame {
            backbone_chains: self.backbone_renderer.cached_chains().to_vec(),
            na_chains: self.backbone_renderer.cached_na_chains().to_vec(),
            sidechains: None,
            ss_types,
            per_residue_colors: self.sc.cached_per_residue_colors.clone(),
            geometry: base_geo,
            per_chain_lod: Some(per_chain_lod),
        });
    }

    /// Apply any pending animation frame from the background thread.
    pub(crate) fn apply_pending_animation(&mut self) {
        let Some(prepared) = self.scene_processor.try_recv_animation() else {
            return;
        };

        self.backbone_renderer.apply_mesh(
            &self.context.device,
            &self.context.queue,
            prepared.backbone,
        );

        if let Some(ref instances) = prepared.sidechain_instances {
            let reallocated = self.sidechain_renderer.apply_prepared(
                &self.context.device,
                &self.context.queue,
                instances,
                prepared.sidechain_instance_count,
            );
            if reallocated {
                self.picking_groups.rebuild_capsule(
                    &self.picking,
                    &self.context.device,
                    &self.sidechain_renderer,
                );
            }
        }
    }

    /// Update protein coords for a specific entity.
    pub fn update_entity_coords(
        &mut self,
        id: u32,
        coords: foldit_conv::types::coords::Coords,
        transition: Transition,
    ) {
        self.scene.update_entity_protein_coords(id, coords);
        self.sync_scene_to_renderers(Some(transition));
    }
}
