//! Scene → renderer pipeline: metadata preparation, GPU upload, animation
//! setup, frustum culling, LOD management.

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec3;
use molex::Assembly;
use rustc_hash::FxHashMap;

use super::viso_state::{EntityTopology, SceneRenderState, VisoEntityState};
use super::{scene_data, VisoEngine};
use crate::animation::transition::Transition;
use crate::animation::AnimationFrame;
use crate::options::{DisplayOptions, DrawingMode, GeometryOptions};
use crate::renderer::gpu_pipeline::SceneChainData;
use crate::renderer::pipeline::prepared::FullRebuildBody;
use crate::renderer::pipeline::{PreparedScene, SceneRequest};

// ── Assembly sync ──

impl VisoEngine {
    /// Rederive viso-side state from an `Assembly` snapshot.
    ///
    /// Called when the triple buffer yields a snapshot whose generation
    /// differs from [`VisoEngine::last_seen_generation`]. Populates
    /// `scene_state`, `entity_state` and `positions` from the snapshot;
    /// entries for entities no longer present in the snapshot are
    /// dropped, and entries for entities new to the snapshot are
    /// inserted with their reference positions.
    pub(crate) fn sync_from_assembly(&mut self, assembly: &Assembly) {
        self.scene_state = Arc::new(SceneRenderState::from_assembly(assembly));

        let mut seen = std::collections::HashSet::new();
        for entity in assembly.entities() {
            let id = entity.id();
            let _ = seen.insert(id);
            let ss = assembly.ss_types(id);
            let topology = Arc::new(EntityTopology::from_entity(entity, ss));
            match self.entity_state.entry(id) {
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    let state = slot.get_mut();
                    state.topology = topology;
                    state.mesh_version = state.mesh_version.wrapping_add(1);
                }
                std::collections::hash_map::Entry::Vacant(slot) => {
                    let _ = slot.insert(VisoEntityState {
                        drawing_mode: DrawingMode::Cartoon,
                        ss_override: None,
                        topology,
                        mesh_version: 0,
                    });
                }
            }
            self.positions
                .insert_from_reference(id, &entity.positions());
        }

        self.entity_state.retain(|id, _| seen.contains(id));
        self.positions.retain(|id| seen.contains(&id));
    }
}

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
        let mut entities = self.entities.per_entity_data(&self.options.display);

        // Rebuild structural topology (residue ranges, sidechain topology,
        // SS types, NA chains, structural bonds) from entity data.
        let molecule_entities: Vec<molex::MoleculeEntity> = self
            .entities
            .entities()
            .iter()
            .filter(|se| se.visible)
            .map(|se| se.entity.clone())
            .collect();
        self.topology.rebuild(&entities, &molecule_entities);

        // Resolve per-entity appearance overrides into concrete options.
        // The base geometry incorporates display helix/sheet style so
        // entities without per-entity style overrides inherit the global.
        let resolved_geometry = self.resolved_geometry();
        let entity_options: FxHashMap<u32, (DisplayOptions, GeometryOptions)> =
            self.entities
                .appearance_overrides()
                .iter()
                .map(|(&id, ovr)| {
                    (
                        id,
                        (
                            ovr.to_display_options(&self.options.display),
                            ovr.to_geometry_options(&resolved_geometry),
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
            .filter(|e| e.drawing_mode == DrawingMode::Cartoon)
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

        // Default (session-wide) colors for entities without overrides.
        let colors =
            crate::options::score_color::compute_per_residue_colors_styled(
                &backbone_chains,
                &self.topology.ss_types,
                &per_entity_scores,
                &self.options.display.backbone_color_scheme,
                &self.options.display.backbone_palette(),
                Some(&entity_chain_counts),
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
        // Build a separate color array with only Cartoon-mode entities'
        // colors — this must match `cartoon_backbone_chains` for the GPU
        // color buffer.
        let cartoon_colors: Vec<[f32; 3]> = entities
            .iter()
            .zip(&self.topology.entity_residue_ranges)
            .filter(|(e, _)| {
                e.drawing_mode == DrawingMode::Cartoon
            })
            .flat_map(|(_, range)| {
                let start = range.start as usize;
                let end = range.end() as usize;
                final_colors.get(start..end).unwrap_or(&[]).iter().copied()
            })
            .collect();
        self.topology.per_residue_colors = Some(final_colors);
        self.topology.cartoon_per_residue_colors = Some(cartoon_colors);

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
        self.mirror_per_entity_colors_to_topology(&entities);

        let generation = self.gpu.scene_processor.next_generation();
        log::debug!(
            "sync_scene_to_renderers: submitting FullRebuild \
             gen={generation}, entity_count={}, backbone_chains={}",
            entities.len(),
            self.visual.backbone_chains.len(),
        );

        // Resolve detected structural bonds (H-bonds + disulfides) from
        // scene state + current positions and upload to the bond
        // renderer.
        let bonds = crate::renderer::geometry::bond::resolve_structural_bonds(
            &self.scene_state,
            &self.positions,
            &self.options.display.bonds,
            &self.options.colors,
        );
        let _ = self.gpu.renderers.bond.update(
            &self.gpu.context.device,
            &self.gpu.context.queue,
            &bonds,
        );

        let request_entities = self.build_full_rebuild_entities(&entities);
        let scene_state = Arc::clone(&self.scene_state);

        self.gpu.scene_processor.submit(SceneRequest::FullRebuild(
            Box::new(FullRebuildBody {
                entities: request_entities,
                scene_state,
                display: self.options.display.clone(),
                colors: self.options.colors.clone(),
                geometry: self.resolved_geometry(),
                entity_options,
                generation,
            }),
        ));
    }

    /// Mirror per-entity colors computed during
    /// [`prepare_scene_metadata`] into the matching `entity_state`
    /// topology, so the background mesh worker sees them.
    fn mirror_per_entity_colors_to_topology(
        &mut self,
        entities: &[scene_data::PerEntityData],
    ) {
        let id_lookup = self.entity_id_lookup();
        for pe in entities {
            let Some(&id) = id_lookup.get(&pe.id) else {
                continue;
            };
            let Some(state) = self.entity_state.get_mut(&id) else {
                continue;
            };
            let topology = Arc::make_mut(&mut state.topology);
            topology
                .per_residue_colors
                .clone_from(&pe.per_residue_colors);
            topology
                .sheet_plane_normals
                .clone_from(&pe.sheet_plane_normals);
        }
    }

    /// Build the per-entity snapshot list the background worker
    /// consumes, in the same ordering `prepare_scene_metadata` produced.
    fn build_full_rebuild_entities(
        &self,
        entities: &[scene_data::PerEntityData],
    ) -> Vec<crate::renderer::pipeline::prepared::FullRebuildEntity> {
        use crate::renderer::pipeline::prepared::FullRebuildEntity;
        let id_lookup = self.entity_id_lookup();
        entities
            .iter()
            .filter_map(|pe| {
                let id = *id_lookup.get(&pe.id)?;
                let state = self.entity_state.get(&id)?;
                let positions = self
                    .positions
                    .get(id)
                    .map(<[Vec3]>::to_vec)
                    .unwrap_or_default();
                Some(FullRebuildEntity {
                    id,
                    mesh_version: state.mesh_version.max(pe.mesh_version),
                    drawing_mode: pe.drawing_mode,
                    topology: Arc::clone(&state.topology),
                    positions,
                    ss_override: pe.ss_override.clone(),
                })
            })
            .collect()
    }

    /// `raw u32` → opaque `EntityId` lookup built by scanning
    /// `entity_state`'s keys. Needed because `EntityId` is
    /// allocator-constructed and can't be synthesized from a raw value.
    fn entity_id_lookup(
        &self,
    ) -> FxHashMap<u32, molex::entity::molecule::id::EntityId> {
        self.entity_state.keys().map(|id| (**id, *id)).collect()
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

        // Upload colors for Cartoon-mode entities only (matching
        // the backbone chains in self.visual.backbone_chains).
        if let Some(ref colors) = self.topology.cartoon_per_residue_colors {
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

        if let Some(ref colors) = self.topology.cartoon_per_residue_colors {
            self.gpu.set_target_colors(colors);
        }
    }

    /// Submit an animation frame to the background thread using the
    /// engine's current [`EntityPositions`] snapshot. The unified
    /// [`AnimationFrame`] produced by the legacy animator still drives
    /// this path (its `sidechains_visible` flag selects whether
    /// sidechain capsules regenerate this frame).
    pub(crate) fn submit_animation_frame_from(&self, frame: &AnimationFrame) {
        self.gpu.submit_animation_frame(
            &self.positions,
            &self.options.geometry,
            frame.sidechains_visible,
        );
    }
}

// ── Frustum + LOD ──

impl VisoEngine {
    /// Update sidechain instances with frustum culling when camera moves
    /// significantly. This filters out sidechains behind the camera to
    /// reduce draw calls.
    ///
    /// Builds the flat sidechain view from legacy engine state
    /// (`visual` + `topology.sidechain_topology`) then asks the GPU
    /// pipeline to apply sheet-plane adjustment, frustum cull, and
    /// upload. Session C replaces the source state with
    /// `entity_state` + `positions` once legacy types are deleted.
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

        self.gpu
            .set_last_cull_camera_eye(self.camera_controller.camera.eye);
        let frustum = self.camera_controller.frustum();
        let positions: &[Vec3] = if self.visual.sidechain_positions.is_empty() {
            &self.topology.sidechain_topology.target_positions
        } else {
            &self.visual.sidechain_positions
        };
        let bs_bonds: Vec<(Vec3, u32)> =
            if self.visual.backbone_sidechain_bonds.is_empty() {
                self.topology.sidechain_topology.target_backbone_bonds.clone()
            } else {
                self.visual.backbone_sidechain_bonds.clone()
            };
        let offset_map: HashMap<u32, Vec3> = self
            .gpu
            .backbone_sheet_offsets()
            .iter()
            .copied()
            .collect();
        let raw_view = crate::renderer::geometry::SidechainView {
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
        self.gpu.upload_frustum_culled_sidechains(
            &adjusted.as_view(),
            &frustum,
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
        let geo = self.resolved_geometry();
        self.gpu.check_and_submit_lod(camera_eye, &geo, &self.positions);
    }

    /// Submit a backbone-only remesh with per-chain LOD to the background
    /// thread.
    pub(crate) fn submit_per_chain_lod_remesh(&self, camera_eye: Vec3) {
        let geo = self.resolved_geometry();
        self.gpu
            .submit_lod_remesh(camera_eye, &geo, &self.positions);
    }

    /// Geometry options with display helix/sheet style folded in.
    fn resolved_geometry(&self) -> GeometryOptions {
        self.options
            .geometry
            .resolve_cartoon_style()
            .with_helix_style(self.options.display.helix_style)
            .with_sheet_style(self.options.display.sheet_style)
    }
}
