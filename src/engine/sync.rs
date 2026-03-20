//! Scene → renderer pipeline: metadata preparation, GPU upload, animation
//! setup, frustum culling, LOD management.

use std::collections::HashMap;

use glam::Vec3;
use rustc_hash::FxHashMap;

use super::{scene_data, VisoEngine};
use crate::animation::transition::Transition;
use crate::animation::AnimationFrame;
use crate::options::{DisplayOptions, GeometryOptions};
use crate::renderer::gpu_pipeline::SceneChainData;
use crate::renderer::pipeline::{PreparedScene, SceneRequest};

// ── Scene sync ──

impl VisoEngine {
    /// Compute metadata from entities and store on Scene before background
    /// submission. Returns the per-entity data, entity transitions, and
    /// resolved per-entity display/geometry overrides.
    #[allow(clippy::type_complexity)]
    fn prepare_scene_metadata(
        &mut self,
        entity_transitions: HashMap<u32, Transition>,
    ) -> (
        Vec<scene_data::PerEntityData>,
        HashMap<u32, Transition>,
        FxHashMap<u32, (DisplayOptions, GeometryOptions)>,
    ) {
        let mut entities = self.entities.per_entity_data();

        // Rebuild structural topology (residue ranges, sidechain topology,
        // SS types, NA chains) from entity data.
        self.topology.rebuild(&entities);

        // Resolve per-entity display overrides into concrete options.
        let resolved_geometry = self.options.geometry.resolve_cartoon_style();
        let entity_options: FxHashMap<u32, (DisplayOptions, GeometryOptions)> =
            self.entities
                .display_overrides()
                .iter()
                .map(|(&id, ovr)| {
                    (
                        id,
                        (
                            ovr.resolve_display(&self.options.display),
                            ovr.resolve_geometry(&resolved_geometry),
                        ),
                    )
                })
                .collect();

        // Concatenate backbone chains for color computation (all
        // entities, including Stick/BnS which keep backbone data for
        // per-residue color calculation).
        let entity_chain_counts: Vec<usize> = entities
            .iter()
            .map(|e| e.backbone_chains.len())
            .filter(|&c| c > 0)
            .collect();
        let backbone_chains: Vec<Vec<Vec3>> = entities
            .iter()
            .flat_map(|e| e.backbone_chains.iter().cloned())
            .collect();

        // Visual backbone chains only include Cartoon-mode entities.
        // Stick/BnS entities keep backbone_chains for color computation
        // but must NOT feed into the LOD/animation backbone mesh path.
        let cartoon_backbone_chains: Vec<Vec<Vec3>> = entities
            .iter()
            .filter(|e| {
                e.drawing_mode == crate::options::DrawingMode::Cartoon
            })
            .flat_map(|e| e.backbone_chains.iter().cloned())
            .collect();
        self.visual
            .backbone_chains
            .clone_from(&cartoon_backbone_chains);

        // Compute per-residue colors on main thread and distribute to
        // entities for vertex coloring (avoids background round-trip).
        // Entities with color overrides get per-entity color computation.
        let per_entity_scores: Vec<Option<&[f64]>> = self
            .entities
            .entities()
            .iter()
            .map(|e| e.per_residue_scores.as_deref())
            .collect();
        let entity_molecule_types: Vec<molex::types::entity::MoleculeType> =
            self.entities
                .entities()
                .iter()
                .filter(|e| e.visible)
                .filter(|e| !e.entity.extract_backbone().chains.is_empty())
                .map(|e| e.entity.molecule_type)
                .collect();

        // Default (session-wide) colors for entities without overrides.
        let colors =
            crate::options::score_color::compute_per_residue_colors_styled(
                &backbone_chains,
                &self.topology.ss_types,
                &per_entity_scores,
                &self.options.display.backbone_color_scheme,
                &self.options.display.backbone_palette(),
                Some(&entity_chain_counts),
                Some(&entity_molecule_types),
            );

        // Per-entity color override: recompute colors for entities that
        // override color scheme or palette, then splice into the flat
        // color array.
        let mut final_colors = colors;
        for (e, range) in
            entities.iter().zip(&self.topology.entity_residue_ranges)
        {
            let Some((ref disp, _)) = entity_options.get(&e.id) else {
                continue;
            };
            let same_colors = disp.backbone_color_scheme
                == self.options.display.backbone_color_scheme
                && disp.backbone_palette()
                    == self.options.display.backbone_palette();
            if same_colors {
                continue;
            }
            let start = range.start as usize;
            let end = range.end() as usize;
            let recolored =
                self.recolor_entity(e, start, end, disp, &per_entity_scores);
            if let Some(slice) = final_colors.get_mut(start..end) {
                let n = slice.len().min(recolored.len());
                slice[..n].copy_from_slice(&recolored[..n]);
            }
        }

        for (e, range) in entities
            .iter_mut()
            .zip(&self.topology.entity_residue_ranges)
        {
            let start = range.start as usize;
            let end = range.end() as usize;
            e.per_residue_colors =
                final_colors.get(start..end).map(<[_]>::to_vec);
        }
        self.topology.per_residue_colors = Some(final_colors);

        (entities, entity_transitions, entity_options)
    }

    /// Recompute per-residue colors for a single entity with overridden
    /// display options.
    fn recolor_entity(
        &self,
        e: &scene_data::PerEntityData,
        start: usize,
        end: usize,
        disp: &DisplayOptions,
        per_entity_scores: &[Option<&[f64]>],
    ) -> Vec<[f32; 3]> {
        let entity_ss = self.topology.ss_types.get(start..end).unwrap_or(&[]);
        let entity_scores =
            per_entity_scores.get(e.id as usize).copied().flatten();
        let mol_types: Vec<_> = self
            .entities
            .entities()
            .iter()
            .filter(|se| se.id() == e.id && se.visible)
            .filter(|se| !se.entity.extract_backbone().chains.is_empty())
            .map(|se| se.entity.molecule_type)
            .collect();
        let chain_counts: Vec<usize> = e
            .backbone_chains
            .iter()
            .map(Vec::len)
            .filter(|&c| c > 0)
            .collect();
        crate::options::score_color::compute_per_residue_colors_styled(
            &e.backbone_chains,
            entity_ss,
            &[entity_scores],
            &disp.backbone_color_scheme,
            &disp.backbone_palette(),
            Some(&chain_counts),
            Some(&mol_types),
        )
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

        let (entities, transitions, entity_options) =
            self.prepare_scene_metadata(entity_transitions);
        self.animation.pending_transitions = transitions;
        self.entities.mark_rendered();

        let generation = self.gpu.scene_processor.next_generation();
        log::debug!(
            "sync_scene_to_renderers: submitting FullRebuild \
             gen={generation}, entity_count={}, backbone_chains={}",
            entities.len(),
            self.visual.backbone_chains.len(),
        );
        self.gpu.scene_processor.submit(SceneRequest::FullRebuild {
            entities,
            display: self.options.display.clone(),
            colors: self.options.colors.clone(),
            geometry: self.options.geometry.resolve_cartoon_style(),
            entity_options,
            generation,
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
        log::debug!(
            "apply_pending_scene: consumed gen={}, backbone verts={}, \
             tube_idx={}, ribbon_idx={}",
            prepared.generation,
            prepared.backbone.vertices.len(),
            prepared.backbone.tube_index_count,
            prepared.backbone.ribbon_index_count,
        );

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
        if self.topology.sidechain_topology.target_positions.is_empty() {
            return;
        }
        if !self.should_update_culling() {
            return;
        }
        let sc_colors = if self.options.display.sidechain_color_mode
            == crate::options::SidechainColorMode::Backbone
        {
            self.topology.per_residue_colors.as_deref()
        } else {
            None
        };
        self.gpu.update_frustum_culling(
            &self.camera_controller,
            &self.visual,
            &self.topology,
            sc_colors,
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
}
