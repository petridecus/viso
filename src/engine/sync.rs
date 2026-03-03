//! Scene → renderer pipeline: metadata preparation, GPU upload, animation
//! setup, frustum culling, LOD management.

use std::collections::HashMap;

use glam::Vec3;

use super::{scene_data, VisoEngine};
use crate::animation::transition::Transition;
use crate::animation::AnimationFrame;
use crate::renderer::geometry::SidechainView;
use crate::renderer::gpu_pipeline::SceneChainData;
use crate::renderer::pipeline::{PreparedScene, SceneRequest};

// ── Scene sync ──

impl VisoEngine {
    /// Compute metadata from entities and store on Scene before background
    /// submission. Returns the per-entity data and entity transitions.
    fn prepare_scene_metadata(
        &mut self,
        entity_transitions: HashMap<u32, Transition>,
    ) -> (Vec<scene_data::PerEntityData>, HashMap<u32, Transition>) {
        let mut entities = self.entities.per_entity_data();

        // Rebuild structural topology (residue ranges, sidechain topology,
        // SS types, NA chains) from entity data.
        self.topology.rebuild(&entities);

        // Concatenate backbone chains and store on visual state for
        // animation / apply_pending_scene.
        let backbone_chains: Vec<Vec<Vec3>> = entities
            .iter()
            .flat_map(|e| e.backbone_chains.iter().cloned())
            .collect();
        self.visual.backbone_chains.clone_from(&backbone_chains);

        // Compute per-residue colors on main thread and distribute to
        // entities for vertex coloring (avoids background round-trip)
        let per_entity_scores: Vec<Option<&[f64]>> = self
            .entities
            .entities()
            .iter()
            .map(|e| e.per_residue_scores.as_deref())
            .collect();
        let colors = crate::options::score_color::compute_per_residue_colors(
            &backbone_chains,
            &self.topology.ss_types,
            &per_entity_scores,
            &self.options.display.backbone_color_mode,
        );
        for (e, range) in entities
            .iter_mut()
            .zip(&self.topology.entity_residue_ranges)
        {
            let start = range.start as usize;
            let end = range.end() as usize;
            e.per_residue_colors = colors.get(start..end).map(<[_]>::to_vec);
        }
        self.topology.per_residue_colors = Some(colors);

        (entities, entity_transitions)
    }

    /// Sync scene data to renderers with per-entity transitions.
    ///
    /// Entities in the map animate with their transition; entities not in
    /// the map snap. Pass an empty map for a non-animated sync.
    pub fn sync_scene_to_renderers(
        &mut self,
        entity_transitions: HashMap<u32, Transition>,
    ) {
        if !self.entities.is_dirty() && entity_transitions.is_empty() {
            return;
        }

        let (entities, transitions) =
            self.prepare_scene_metadata(entity_transitions);
        self.animation.pending_transitions = transitions;
        self.entities.mark_rendered();

        self.gpu.scene_processor.submit(SceneRequest::FullRebuild {
            entities,
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
        let scene = SceneChainData {
            backbone_chains: &self.visual.backbone_chains,
            na_chains: &self.topology.na_chains,
            ss_types: &self.topology.ss_types,
        };
        self.gpu.upload_prepared(
            prepared,
            animating,
            suppress_sidechains,
            &scene,
        );
    }

    /// Apply any pending scene data from the background SceneProcessor.
    ///
    /// Called every frame from the main loop. If the background thread has
    /// finished generating geometry, this uploads it to the GPU (<1ms) and
    /// sets up animation.
    pub fn apply_pending_scene(&mut self) {
        let Some(prepared) = self.gpu.scene_processor.try_recv_scene() else {
            return;
        };

        let entity_transitions =
            std::mem::take(&mut self.animation.pending_transitions);
        let animating = !entity_transitions.is_empty();

        if animating {
            let backbone = self.visual.backbone_chains.clone();
            let frame = self.animation.setup_per_entity(
                &entity_transitions,
                &backbone,
                &self.topology.sidechain_topology,
                &self.topology.entity_residue_ranges.clone(),
            );
            self.ensure_gpu_capacity_and_colors(&backbone);
            self.submit_animation_frame_from(&frame);
        } else {
            self.snap_from_prepared();
        }

        let suppress_sidechains = entity_transitions
            .values()
            .any(|t| t.suppress_initial_sidechains);
        self.upload_prepared_to_gpu(&prepared, animating, suppress_sidechains);
    }

    /// Snap update from prepared scene data (no animation).
    ///
    /// All metadata (backbone chains, sidechain topology, SS types, colors)
    /// is already on Scene from `prepare_scene_metadata`. This just sets
    /// visual state and ensures GPU buffer capacity.
    fn snap_from_prepared(&mut self) {
        // Write full at-rest visual state to Scene (backbone chains
        // already set in prepare_scene_metadata; sidechain topology too)
        self.visual.update(
            self.visual.backbone_chains.clone(),
            self.topology.sidechain_topology.target_positions.clone(),
            self.topology
                .sidechain_topology
                .target_backbone_bonds
                .clone(),
        );

        let total_residues =
            crate::renderer::geometry::sheet_adjust::backbone_residue_count(
                &self.visual.backbone_chains,
            );
        self.gpu.ensure_residue_capacity(total_residues);

        // Colors already computed in prepare_scene_metadata
        if let Some(ref colors) = self.topology.per_residue_colors {
            self.gpu.set_colors_immediate(colors);
        }
    }

    /// Apply any pending animation frame from the background thread.
    pub(crate) fn apply_pending_animation(&mut self) {
        if self.gpu.apply_pending_animation() {
            self.visual.mark_rendered();
        }
    }
}

// ── Animation setup ──

impl VisoEngine {
    /// Ensure GPU buffer capacity and update colors after animation setup.
    fn ensure_gpu_capacity_and_colors(
        &mut self,
        backbone_chains: &[Vec<Vec3>],
    ) {
        let total_residues =
            crate::renderer::geometry::sheet_adjust::backbone_residue_count(
                backbone_chains,
            );
        self.gpu.ensure_residue_capacity(total_residues);

        if let Some(ref colors) = self.topology.per_residue_colors {
            self.gpu.set_target_colors(colors);
        }
    }

    /// Submit an animation frame to the background thread for mesh
    /// generation, using a unified [`AnimationFrame`] from the animator.
    pub(crate) fn submit_animation_frame_from(&self, frame: &AnimationFrame) {
        self.gpu.submit_animation_frame(
            frame,
            &self.topology.sidechain_topology,
            &self.options.geometry,
        );
    }
}

// ── Frustum + LOD ──

impl VisoEngine {
    /// Update sidechain instances with frustum culling when camera moves
    /// significantly. This filters out sidechains behind the camera to
    /// reduce draw calls.
    pub(crate) fn update_frustum_culling(&mut self) {
        // Skip if no sidechain data
        if self.topology.sidechain_topology.target_positions.is_empty() {
            return;
        }

        // Only update culling when camera moves more than 5 units.
        // Exception: always update during animation so sidechain positions
        // reflect the interpolated state.
        if !self.should_update_culling() {
            return;
        }

        let camera_eye = self.camera_controller.camera.eye;
        self.gpu.last_cull_camera_eye = camera_eye;

        let frustum = self.camera_controller.frustum();
        // Read visual state from Scene (populated by tick_animation or
        // snap_from_prepared).
        let positions = if self.visual.sidechain_positions.is_empty() {
            &self.topology.sidechain_topology.target_positions
        } else {
            &self.visual.sidechain_positions
        };
        let bs_bonds = if self.visual.backbone_sidechain_bonds.is_empty() {
            self.topology
                .sidechain_topology
                .target_backbone_bonds
                .clone()
        } else {
            self.visual.backbone_sidechain_bonds.clone()
        };

        // Translate sidechains onto sheet surface and apply frustum culling
        let offset_map = self.sheet_offset_map();
        let raw_view = SidechainView {
            positions,
            bonds: &self.topology.sidechain_topology.bonds,
            backbone_bonds: &bs_bonds,
            hydrophobicity: &self.topology.sidechain_topology.hydrophobicity,
            residue_indices: &self.topology.sidechain_topology.residue_indices,
        };
        let adjusted =
            crate::renderer::geometry::sheet_adjust::sheet_adjusted_view(
                &raw_view,
                &offset_map,
            );

        self.gpu.renderers.sidechain.update_with_frustum(
            &self.gpu.context.device,
            &self.gpu.context.queue,
            &adjusted.as_view(),
            Some(&frustum),
        );

        // Recreate picking bind group since buffer may have changed
        self.gpu.pick.groups.rebuild_capsule(
            &self.gpu.pick.picking,
            &self.gpu.context.device,
            &self.gpu.renderers.sidechain,
        );
    }

    /// Whether frustum culling should be recalculated this frame.
    ///
    /// Returns `true` when the camera has moved more than the threshold or
    /// an animation with sidechain data is active (positions are
    /// interpolated and need continuous updates).
    fn should_update_culling(&self) -> bool {
        const CULL_UPDATE_THRESHOLD: f32 = 5.0;

        let animating = self.animation.animator.is_animating()
            && !self.topology.sidechain_topology.target_positions.is_empty();
        if animating {
            return true;
        }

        let camera_eye = self.camera_controller.camera.eye;
        let camera_delta =
            (camera_eye - self.gpu.last_cull_camera_eye).length();
        camera_delta >= CULL_UPDATE_THRESHOLD
    }

    /// Check per-chain LOD tiers and submit a background remesh if any
    /// chain's tier has changed.
    pub(crate) fn check_and_submit_lod(&mut self) {
        let camera_eye = self.camera_controller.camera.eye;
        self.gpu
            .check_and_submit_lod(camera_eye, &self.options.geometry);
    }

    /// Submit a backbone-only remesh with per-chain LOD to the background
    /// thread.
    pub(crate) fn submit_per_chain_lod_remesh(&self, camera_eye: Vec3) {
        self.gpu
            .submit_lod_remesh(camera_eye, &self.options.geometry);
    }

    /// Build a map of sheet residue offsets (residue_idx -> offset vector).
    pub(crate) fn sheet_offset_map(&self) -> HashMap<u32, Vec3> {
        self.gpu
            .renderers
            .backbone
            .sheet_offsets()
            .iter()
            .copied()
            .collect()
    }
}
