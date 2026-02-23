//! Scene Sync methods for ProteinRenderEngine

use std::collections::{HashMap, HashSet};

use foldit_conv::secondary_structure::SSType;
use glam::Vec3;

use super::ProteinRenderEngine;
use crate::{
    animation::AnimationAction,
    renderer::molecular::{
        ball_and_stick::PreparedBallAndStickData,
        capsule_sidechain::SidechainData,
    },
    scene::processor::{AnimationSidechainData, PreparedScene, SceneRequest},
    util::score_color,
};

impl ProteinRenderEngine {
    /// Update all renderers from aggregated scene data
    ///
    /// This is the main integration point for the Scene-based rendering model.
    /// Call this whenever structures are added, removed, or modified in the
    /// scene.
    pub fn update_from_aggregated(
        &mut self,
        backbone_chains: &[Vec<Vec3>],
        sidechain: &SidechainData,
        sidechain_atom_names: &[String],
        all_positions: &[Vec3],
        fit_camera: bool,
        ss_types: Option<&[SSType]>,
    ) {
        // Calculate total residues from backbone chains (3 atoms per residue:
        // N, CA, C)
        let total_residues: usize =
            backbone_chains.iter().map(|c| c.len() / 3).sum();

        // Ensure selection buffer has capacity for all residues (including new
        // structures)
        self.selection_buffer
            .ensure_capacity(&self.context.device, total_residues);

        // Update backbone tubes
        self.tube_renderer.update(
            &self.context.device,
            &self.context.queue,
            backbone_chains,
            ss_types,
        );

        // Update ribbon renderer
        self.ribbon_renderer.update(
            &self.context.device,
            &self.context.queue,
            backbone_chains,
            ss_types,
        );

        // Translate sidechains onto sheet surface (whole sidechain, not just
        // CA-CB bond)
        let offset_map = self.sheet_offset_map();
        let adjusted_positions =
            crate::util::sheet_adjust::adjust_sidechains_for_sheet(
                sidechain.positions,
                sidechain.residue_indices,
                &offset_map,
            );
        let adjusted_bonds = crate::util::sheet_adjust::adjust_bonds_for_sheet(
            sidechain.backbone_bonds,
            sidechain.residue_indices,
            &offset_map,
        );

        self.sidechain_renderer.update(
            &self.context.device,
            &self.context.queue,
            &SidechainData {
                positions: &adjusted_positions,
                bonds: sidechain.bonds,
                backbone_bonds: &adjusted_bonds,
                hydrophobicity: sidechain.hydrophobicity,
                residue_indices: sidechain.residue_indices,
            },
        );

        // Update capsule picking bind group (buffer may have been reallocated)
        self.picking_groups.rebuild_capsule(
            &self.picking,
            &self.context.device,
            &self.sidechain_renderer,
        );

        // Cache secondary structure types for double-click segment selection
        self.sc.cached_ss_types = if let Some(ss) = ss_types {
            ss.to_vec()
        } else {
            self.compute_ss_types(backbone_chains)
        };

        // Cache atom names for lookup by name (used for band tracking during
        // animation)
        self.sc.cached_sidechain_atom_names = sidechain_atom_names.to_vec();

        // Fit camera if requested and we have positions
        if fit_camera && !all_positions.is_empty() {
            self.camera_controller.fit_to_positions(all_positions);
        }
    }

    /// Sync scene data to renderers with a global animation action.
    ///
    /// All entities animate with the same action (or snap if `None`).
    pub fn sync_scene_to_renderers(&mut self, action: Option<AnimationAction>) {
        if !self.scene.is_dirty() && action.is_none() {
            return;
        }

        let entities = self.scene.per_entity_data();
        // Build entity_actions: all entities get the same action
        let entity_actions = match action {
            Some(a) => entities.iter().map(|e| (e.id, a)).collect(),
            None => HashMap::new(),
        };
        self.scene.mark_rendered();

        self.scene_processor.submit(SceneRequest::FullRebuild {
            entities,
            entity_actions,
            display: self.options.display.clone(),
            colors: self.options.colors.clone(),
        });
    }

    /// Sync scene data to renderers with per-entity animation actions.
    ///
    /// Entities in the map animate with their action; all others snap.
    pub fn sync_scene_to_renderers_targeted(
        &mut self,
        entity_actions: HashMap<u32, AnimationAction>,
    ) {
        if !self.scene.is_dirty() && entity_actions.is_empty() {
            return;
        }

        let entities = self.scene.per_entity_data();
        self.scene.mark_rendered();

        self.scene_processor.submit(SceneRequest::FullRebuild {
            entities,
            entity_actions,
            display: self.options.display.clone(),
            colors: self.options.colors.clone(),
        });
    }

    /// Apply any pending scene data from the background SceneProcessor.
    ///
    /// Called every frame from the main loop. If the background thread has
    /// finished generating geometry, this uploads it to the GPU (<1ms) and
    /// sets up animation.
    pub fn apply_pending_scene(&mut self) {
        let prepared = match self.scene_processor.try_recv_scene() {
            Some(p) => p,
            None => return,
        };

        // Triple buffer automatically returns only the latest result,
        // so no drain loop needed — stale intermediates are skipped.

        // Animation target setup or snap update (fast: array copies + animator)
        let dominant_action = prepared.entity_actions.values().next().copied();
        if !prepared.entity_actions.is_empty() {
            let action = dominant_action.unwrap_or(AnimationAction::Wiggle);
            self.setup_animation_targets_from_prepared(&prepared, action);

            // Snap residues for entities NOT in entity_actions
            let active: HashSet<u32> =
                prepared.entity_actions.keys().copied().collect();
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
            for &(eid, start_residue, residue_count) in
                &prepared.entity_residue_ranges
            {
                if active.contains(&eid) {
                    continue;
                }
                let res_start = start_residue as usize;
                let res_end = (start_residue + residue_count) as usize;
                for (i, &res_idx) in
                    self.sc.cached_sidechain_residue_indices.iter().enumerate()
                {
                    let r = res_idx as usize;
                    if r >= res_start
                        && r < res_end
                        && i < self.sc.start_sidechain_positions.len()
                        && i < self.sc.target_sidechain_positions.len()
                    {
                        self.sc.start_sidechain_positions[i] =
                            self.sc.target_sidechain_positions[i];
                    }
                }
                for (j, &(_, cb_idx)) in
                    self.sc.target_backbone_sidechain_bonds.iter().enumerate()
                {
                    let res_idx = self
                        .sc
                        .cached_sidechain_residue_indices
                        .get(cb_idx as usize)
                        .copied()
                        .unwrap_or(u32::MAX)
                        as usize;
                    if res_idx >= res_start
                        && res_idx < res_end
                        && j < self.sc.start_backbone_sidechain_bonds.len()
                    {
                        self.sc.start_backbone_sidechain_bonds[j] =
                            self.sc.target_backbone_sidechain_bonds[j];
                    }
                }
            }

            // Immediately submit an animation frame so the background thread
            // starts generating an interpolated mesh right away.  Since we skip
            // vertex uploads from the FullRebuild during animation (to avoid a
            // one-frame jump), this ensures the next render picks up a
            // correctly interpolated mesh with minimal latency.
            self.submit_animation_frame();
        } else {
            self.snap_from_prepared(&prepared);
        }

        // GPU uploads — each is <0.2ms
        // When animating, skip uploading vertex data for backbone renderers
        // (tube + ribbon + sidechains) to avoid a one-frame jump to target
        // positions. The animation frame path will provide interpolated meshes.
        // We still update metadata (chains, SS types) so animation frames
        // generate correct topology.
        let animating = !prepared.entity_actions.is_empty();
        if animating {
            self.tube_renderer.update_metadata(
                prepared.backbone_chains.clone(),
                prepared.ss_types.clone(),
            );
            self.ribbon_renderer.update_metadata(
                prepared.backbone_chains.clone(),
                prepared.ss_types.clone(),
            );
        } else {
            self.tube_renderer.apply_prepared(
                &self.context.device,
                &self.context.queue,
                crate::renderer::molecular::tube::PreparedTubeData {
                    vertices: &prepared.tube_vertices,
                    indices: &prepared.tube_indices,
                    index_count: prepared.tube_index_count,
                    cached_chains: prepared.backbone_chains.clone(),
                    ss_override: prepared.ss_types.clone(),
                },
            );
            self.ribbon_renderer.apply_prepared(
                &self.context.device,
                &self.context.queue,
                crate::renderer::molecular::ribbon::PreparedRibbonData {
                    vertices: &prepared.ribbon_vertices,
                    indices: &prepared.ribbon_indices,
                    index_count: prepared.ribbon_index_count,
                    sheet_offsets: prepared.sheet_offsets.clone(),
                    cached_chains: prepared.backbone_chains.clone(),
                    ss_override: prepared.ss_types.clone(),
                },
            );
            let suppress_sidechains =
                dominant_action == Some(AnimationAction::DiffusionFinalize);
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
                sphere_bytes: &prepared.bns_sphere_instances,
                sphere_count: prepared.bns_sphere_count,
                capsule_bytes: &prepared.bns_capsule_instances,
                capsule_count: prepared.bns_capsule_count,
                picking_bytes: &prepared.bns_picking_capsules,
                picking_count: prepared.bns_picking_count,
            },
        );

        self.nucleic_acid_renderer.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &prepared.na_vertices,
            &prepared.na_indices,
            prepared.na_index_count,
        );

        // Recreate picking bind groups (buffers may have been reallocated)
        self.picking_groups.rebuild_all(
            &self.picking,
            &self.context.device,
            &self.sidechain_renderer,
            &self.ball_and_stick_renderer,
        );
    }

    /// Set up animation targets from prepared scene data.
    fn setup_animation_targets_from_prepared(
        &mut self,
        prepared: &PreparedScene,
        action: AnimationAction,
    ) {
        let new_backbone = &prepared.backbone_chains;
        let sidechain_positions = &prepared.sidechain_positions;
        let sidechain_bonds = &prepared.sidechain_bonds;
        let sidechain_hydrophobicity = &prepared.sidechain_hydrophobicity;
        let sidechain_residue_indices = &prepared.sidechain_residue_indices;
        let sidechain_atom_names = &prepared.sidechain_atom_names;
        let backbone_sidechain_bonds = &prepared.backbone_sidechain_bonds;

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
                self.sc.start_sidechain_positions =
                    self.sc.target_sidechain_positions.clone();
                self.sc.start_backbone_sidechain_bonds =
                    self.sc.target_backbone_sidechain_bonds.clone();
            }
        } else {
            let ca_positions: Vec<Vec3> = new_backbone
                .iter()
                .flat_map(|chain| {
                    chain.chunks(3).filter_map(|chunk| chunk.get(1).copied())
                })
                .collect();
            self.sc.start_sidechain_positions = sidechain_residue_indices
                .iter()
                .map(|&ri| {
                    ca_positions.get(ri as usize).copied().unwrap_or(Vec3::ZERO)
                })
                .collect();
            self.sc.start_backbone_sidechain_bonds =
                backbone_sidechain_bonds.to_vec();
        }

        // Set new targets and cached data
        self.sc.target_sidechain_positions = sidechain_positions.to_vec();
        self.sc.target_backbone_sidechain_bonds =
            backbone_sidechain_bonds.to_vec();
        self.sc.cached_sidechain_bonds = sidechain_bonds.to_vec();
        self.sc.cached_sidechain_hydrophobicity =
            sidechain_hydrophobicity.to_vec();
        self.sc.cached_sidechain_residue_indices =
            sidechain_residue_indices.to_vec();
        self.sc.cached_sidechain_atom_names = sidechain_atom_names.to_vec();

        // Cache secondary structure types
        if let Some(ref ss) = prepared.ss_types {
            self.sc.cached_ss_types = ss.clone();
        } else {
            self.sc.cached_ss_types = self.compute_ss_types(new_backbone);
        }

        // Cache per-residue colors (derived from scores by scene processor)
        self.sc.cached_per_residue_colors = prepared.per_residue_colors.clone();

        // Extract CA positions for sidechain collapse animation
        let ca_positions: Vec<Vec3> = new_backbone
            .iter()
            .flat_map(|chain| {
                chain.chunks(3).filter_map(|chunk| chunk.get(1).copied())
            })
            .collect();

        // Pass sidechain data to animator FIRST (before set_target)
        self.animator.set_sidechain_target_with_action(
            sidechain_positions,
            sidechain_residue_indices,
            &ca_positions,
            Some(action),
        );

        // Set backbone target (starts the animation)
        self.animator.set_target(new_backbone, action);

        // Ensure selection/color buffers have capacity and update colors
        let total_residues: usize =
            new_backbone.iter().map(|c| c.len() / 3).sum();
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
        self.sc.target_sidechain_positions =
            prepared.sidechain_positions.clone();
        self.sc.start_sidechain_positions =
            prepared.sidechain_positions.clone();
        self.sc.target_backbone_sidechain_bonds =
            prepared.backbone_sidechain_bonds.clone();
        self.sc.start_backbone_sidechain_bonds =
            prepared.backbone_sidechain_bonds.clone();
        self.sc.cached_sidechain_bonds = prepared.sidechain_bonds.clone();
        self.sc.cached_sidechain_hydrophobicity =
            prepared.sidechain_hydrophobicity.clone();
        self.sc.cached_sidechain_residue_indices =
            prepared.sidechain_residue_indices.clone();
        self.sc.cached_sidechain_atom_names =
            prepared.sidechain_atom_names.clone();

        if let Some(ref ss) = prepared.ss_types {
            self.sc.cached_ss_types = ss.clone();
        } else {
            self.sc.cached_ss_types =
                self.compute_ss_types(&prepared.backbone_chains);
        }

        self.sc.cached_per_residue_colors = prepared.per_residue_colors.clone();

        let total_residues: usize =
            prepared.backbone_chains.iter().map(|c| c.len() / 3).sum();
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
        let (r, g, b) = match sector as u32 {
            0 => (1.0, frac, 0.0),       // red → yellow
            1 => (1.0 - frac, 1.0, 0.0), // yellow → green
            2 => (0.0, 1.0, frac),       // green → cyan
            3 => (0.0, 1.0 - frac, 1.0), // cyan → blue
            _ => (0.0, 0.0, 1.0),        // blue
        };
        [r, g, b]
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
                    self.sc
                        .cached_ss_types
                        .iter()
                        .map(|ss| ss.color())
                        .collect()
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
    pub(crate) fn submit_animation_frame(&mut self) {
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
        &mut self,
        backbone_chains: Vec<Vec<Vec3>>,
        include_sidechains: bool,
    ) {
        let sidechains = if include_sidechains {
            let interpolated_positions =
                self.animator.get_sidechain_positions();

            let interpolated_bs_bonds: Vec<(Vec3, u32)> = self
                .sc
                .target_backbone_sidechain_bonds
                .iter()
                .map(|(target_ca_pos, cb_idx)| {
                    let res_idx =
                        self.sc
                            .cached_sidechain_residue_indices
                            .get(*cb_idx as usize)
                            .copied()
                            .unwrap_or(0) as usize;
                    let ca_pos = self
                        .animator
                        .get_ca_position(res_idx)
                        .unwrap_or(*target_ca_pos);
                    (ca_pos, *cb_idx)
                })
                .collect();

            Some(AnimationSidechainData {
                sidechain_positions: interpolated_positions,
                sidechain_bonds: self.sc.cached_sidechain_bonds.clone(),
                backbone_sidechain_bonds: interpolated_bs_bonds,
                sidechain_hydrophobicity: self
                    .sc
                    .cached_sidechain_hydrophobicity
                    .clone(),
                sidechain_residue_indices: self
                    .sc
                    .cached_sidechain_residue_indices
                    .clone(),
            })
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
            sidechains,
            ss_types,
            per_residue_colors: self.sc.cached_per_residue_colors.clone(),
        });
    }

    /// Apply any pending animation frame from the background thread.
    pub(crate) fn apply_pending_animation(&mut self) {
        let prepared = match self.scene_processor.try_recv_animation() {
            Some(p) => p,
            None => return,
        };

        self.tube_renderer.apply_mesh(
            &self.context.device,
            &self.context.queue,
            &prepared.tube_vertices,
            &prepared.tube_indices,
            prepared.tube_index_count,
        );

        self.ribbon_renderer.apply_mesh(
            &self.context.device,
            &self.context.queue,
            &prepared.ribbon_vertices,
            &prepared.ribbon_indices,
            prepared.ribbon_index_count,
            prepared.sheet_offsets,
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
        coords: foldit_conv::coords::Coords,
        action: AnimationAction,
    ) {
        self.scene.update_entity_protein_coords(id, coords);
        self.sync_scene_to_renderers(Some(action));
    }
}
