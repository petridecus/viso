//! Assembly sync + scene → renderer pipeline.
//!
//! All functions read [`VisoEngine`] state derived from the last
//! [`Assembly`] snapshot and the animator's [`EntityPositions`].
//! Neither `&Assembly` nor `&MoleculeEntity` appear in render-path
//! signatures — the sync layer is the only place they're read, and
//! it produces render-ready derived state downstream consumers use.

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec3;
use molex::entity::molecule::id::EntityId;
use molex::{Assembly, MoleculeEntity, MoleculeType, SSType};
use rustc_hash::FxHashMap;

use super::entity_view::{EntityView, RibbonBackbone};
use super::scene_state::{BondResolveInput, SceneRenderState};
use crate::renderer::entity_topology::EntityTopology;
use super::VisoEngine;
use crate::animation::transition::Transition;
use crate::options::{DisplayOptions, DrawingMode, GeometryOptions};
use crate::renderer::gpu_pipeline::SceneChainData;
use crate::renderer::pipeline::prepared::{
    FullRebuildBody, FullRebuildEntity, PreparedScene,
};
use crate::renderer::pipeline::SceneRequest;

// ── Assembly sync ──

impl VisoEngine {
    /// Rederive viso-side state from an `Assembly` snapshot.
    ///
    /// Called when the triple buffer yields a snapshot whose
    /// generation differs from [`Self::last_seen_generation`].
    pub(crate) fn sync_from_assembly(&mut self, assembly: &Assembly) {
        self.scene_state =
            Arc::new(SceneRenderState::from_assembly(assembly));

        let mut seen: std::collections::HashSet<EntityId> =
            std::collections::HashSet::default();
        for entity in assembly.entities() {
            let id = entity.id();
            let _ = seen.insert(id);
            let ss = assembly.ss_types(id);
            let ss_override = self.entity_ss_overrides.get(&id.raw()).cloned();
            let topology =
                Arc::new(crate::engine::entity_view::derive_topology(entity, ss));
            let drawing_mode = self
                .resolved_drawing_mode(id.raw(), topology.molecule_type);
            let fresh_version = self.bump_mesh_version();
            match self.entity_state.entry(id) {
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    let state = slot.get_mut();
                    state.topology = topology;
                    state.ss_override = ss_override;
                    state.drawing_mode = drawing_mode;
                    state.mesh_version = fresh_version;
                }
                std::collections::hash_map::Entry::Vacant(slot) => {
                    let _ = slot.insert(EntityView {
                        drawing_mode,
                        ss_override,
                        topology,
                        mesh_version: fresh_version,
                    });
                }
            }
            self.positions
                .insert_from_reference(id, &entity.positions());

            // New entity? Seed visibility from ambient-type defaults.
            if !self.entity_visibility.contains_key(&id.raw()) {
                let visible = match entity.molecule_type() {
                    MoleculeType::Water => self.options.display.show_waters,
                    MoleculeType::Ion => self.options.display.show_ions,
                    MoleculeType::Solvent => self.options.display.show_solvent,
                    _ => true,
                };
                let _ = self.entity_visibility.insert(id.raw(), visible);
            }
        }

        self.entity_state.retain(|id, _| seen.contains(id));
        self.positions.retain(|id| seen.contains(&id));
        let raw_seen: std::collections::HashSet<u32> =
            seen.iter().map(|id| id.raw()).collect();
        self.entity_visibility.retain(|id, _| raw_seen.contains(id));
        self.entity_behaviors.retain(|id, _| raw_seen.contains(id));
        self.appearance_overrides.retain(|id, _| raw_seen.contains(id));
        self.entity_scores.retain(|id, _| raw_seen.contains(id));
        self.entity_ss_overrides.retain(|id, _| raw_seen.contains(id));
    }
}

// ── Scene sync ──

impl VisoEngine {
    /// Poll the consumer, then submit a full-rebuild request to the
    /// background mesh processor using the current pending transitions.
    pub(crate) fn sync_now(&mut self) {
        self.poll_assembly_force();
        let transitions =
            std::mem::take(&mut self.animation.pending_transitions);
        self.sync_scene_to_renderers(transitions);
    }

    /// Poll the consumer and immediately apply any new snapshot. Used
    /// by [`Self::sync_now`] when a mutation has just published a new
    /// snapshot that must be reflected before the next render.
    fn poll_assembly_force(&mut self) {
        let Some(assembly) = self.assembly_consumer.latest() else {
            return;
        };
        if assembly.generation() == self.last_seen_generation {
            return;
        }
        self.sync_from_assembly(&assembly);
        self.current_assembly = assembly;
        self.last_seen_generation = self.current_assembly.generation();
    }

    /// Sync scene data to renderers with per-entity transitions.
    ///
    /// Entities in the map animate with their transition; entities
    /// not in the map snap. Pass an empty map for a non-animated sync.
    pub fn sync_scene_to_renderers(
        &mut self,
        entity_transitions: HashMap<u32, Transition>,
    ) {
        self.install_per_entity_render_data();
        self.resolve_structural_bonds_into_scene_state();
        let _ = self.gpu.renderers.bond.update(
            &self.gpu.context.device,
            &self.gpu.context.queue,
            self.scene_state.structural_bonds(),
        );

        let entity_options = self.resolve_entity_options();
        let request_entities = self.build_full_rebuild_entities();
        self.animation.pending_transitions = entity_transitions;

        let scene_state = Arc::clone(&self.scene_state);
        let generation = self.gpu.scene_processor.next_generation();
        log::debug!(
            "sync_scene_to_renderers: submitting FullRebuild gen={generation}, \
             entity_count={}",
            request_entities.len(),
        );
        self.gpu.scene_processor.submit(SceneRequest::FullRebuild(Box::new(
            FullRebuildBody {
                entities: request_entities,
                scene_state,
                display: self.options.display.clone(),
                colors: self.options.colors.clone(),
                geometry: self.resolved_geometry(),
                entity_options,
                generation,
            },
        )));
    }

    /// Build the per-sync ribbon cache and let
    /// [`SceneRenderState::update_structural_bonds`] produce this
    /// frame's `Vec<StructuralBond>` directly on `scene_state`. The
    /// resolver reads visibility, drawing mode, and topology straight
    /// off the engine's existing maps — the only thing worth caching
    /// is the spline projection.
    fn resolve_structural_bonds_into_scene_state(&mut self) {
        let ribbons: FxHashMap<EntityId, RibbonBackbone> = self
            .entity_state
            .iter()
            .filter(|(_, state)| {
                state.drawing_mode == DrawingMode::Cartoon
                    && state.topology.is_protein()
            })
            .filter_map(|(&id, state)| {
                let positions = self.positions.get(id)?;
                let ribbon = RibbonBackbone::project(&state.topology, positions)?;
                Some((id, ribbon))
            })
            .collect();
        let input = BondResolveInput {
            positions: &self.positions,
            entity_views: &self.entity_state,
            entity_visibility: &self.entity_visibility,
            ribbons: &ribbons,
            options: &self.options.display.bonds,
            colors: &self.options.colors,
        };
        Arc::make_mut(&mut self.scene_state).update_structural_bonds(&input);
    }

    /// Walk protein entities, compute per-residue colors + sheet-plane
    /// normals, and mirror them onto the Arc-shared topology so the
    /// mesh worker sees them.
    fn install_per_entity_render_data(&mut self) {
        // Take an Arc handle so the entity walk is decoupled from
        // `&mut self` borrows used for per-entity mutation below.
        let assembly = Arc::clone(&self.current_assembly);
        let hbond_ranges = protein_hbond_ranges(assembly.as_ref());

        for (entity_index, entity) in assembly.entities().iter().enumerate() {
            let eid = entity.id();
            let Some(positions) = self.positions.get(eid) else {
                continue;
            };
            let positions = positions.to_vec();
            let Some(state) = self.entity_state.get_mut(&eid) else {
                continue;
            };
            let display = self
                .appearance_overrides
                .get(&eid.raw())
                .map_or_else(
                    || self.options.display.clone(),
                    |ovr| ovr.to_display_options(&self.options.display),
                );
            let ss_types: Vec<SSType> = state
                .ss_override
                .clone()
                .unwrap_or_else(|| state.topology.ss_types.clone());
            let backbone_chains =
                state.topology.backbone_chain_positions(&positions);
            let colors = if state.topology.is_protein() {
                per_entity_colors(
                    entity_index,
                    &backbone_chains,
                    &ss_types,
                    self.entity_scores.get(&eid.raw()).map(Vec::as_slice),
                    &display,
                )
            } else {
                None
            };

            let sheet_plane_normals = if state.topology.is_protein()
                && state.drawing_mode == DrawingMode::Cartoon
            {
                entity_sheet_plane_normals(
                    &self.current_assembly,
                    eid,
                    &ss_types,
                    &positions,
                    &state.topology,
                    &hbond_ranges,
                )
            } else {
                Vec::new()
            };

            let topology = Arc::make_mut(&mut state.topology);
            topology.per_residue_colors = colors;
            topology.sheet_plane_normals = sheet_plane_normals;
        }
    }

    fn resolve_entity_options(
        &self,
    ) -> FxHashMap<u32, (DisplayOptions, GeometryOptions)> {
        let resolved_geometry = self.resolved_geometry();
        self.appearance_overrides
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
            .collect()
    }

    /// Iterate every visible entity that has an `entity_state` slot,
    /// yielding `(entity, entity_id, view)`. Skips invisible entities
    /// and entities not yet reconciled into `entity_state`. Callers
    /// that also need positions look them up via `self.positions.get`.
    fn visible_entities(
        &self,
    ) -> impl Iterator<Item = (&MoleculeEntity, EntityId, &EntityView)> {
        self.current_assembly.entities().iter().filter_map(|entity| {
            let eid = entity.id();
            if !self.is_entity_visible(eid.raw()) {
                return None;
            }
            let state = self.entity_state.get(&eid)?;
            Some((entity, eid, state))
        })
    }

    fn build_full_rebuild_entities(&self) -> Vec<FullRebuildEntity> {
        self.visible_entities()
            .map(|(_, eid, state)| {
                let positions = self
                    .positions
                    .get(eid)
                    .map(<[Vec3]>::to_vec)
                    .unwrap_or_default();
                FullRebuildEntity {
                    id: eid,
                    mesh_version: state.mesh_version,
                    drawing_mode: state.drawing_mode,
                    topology: Arc::clone(&state.topology),
                    positions,
                    ss_override: state.ss_override.clone(),
                }
            })
            .collect()
    }

    /// Upload prepared scene geometry to GPU renderers. Rebuilds the
    /// flat [`SceneChainData`] from entity_state + positions for the
    /// renderers that still consume it internally (backbone metadata
    /// cache used by frustum culling + LOD tier comparison).
    fn upload_prepared_to_gpu(
        &mut self,
        prepared: &PreparedScene,
        animating: bool,
        suppress_sidechains: bool,
    ) {
        let (backbone_chains, na_chains) = self.flat_scene_chains();
        let scene = SceneChainData {
            backbone_chains: &backbone_chains,
            na_chains: &na_chains,
        };
        self.gpu.upload_prepared(
            prepared,
            animating,
            suppress_sidechains,
            &scene,
        );
    }

    /// Flatten per-entity backbone / NA chains in assembly order. Only
    /// Cartoon-mode protein entities contribute to the flat backbone
    /// (matching pre-migration behaviour).
    fn flat_scene_chains(&self) -> (Vec<Vec<Vec3>>, Vec<Vec<Vec3>>) {
        let mut backbone = Vec::new();
        let mut na = Vec::new();
        for (_, eid, state) in self.visible_entities() {
            let Some(positions) = self.positions.get(eid) else {
                continue;
            };
            if state.topology.is_protein()
                && state.drawing_mode == DrawingMode::Cartoon
            {
                backbone
                    .extend(state.topology.backbone_chain_positions(positions));
            } else if state.topology.is_nucleic_acid() {
                na.extend(state.topology.backbone_chain_positions(positions));
            }
        }
        (backbone, na)
    }

    /// Apply any pending scene data from the background `SceneProcessor`.
    pub fn apply_pending_scene(&mut self) {
        let Some(prepared) = self.gpu.scene_processor.try_recv_scene() else {
            return;
        };

        let entity_transitions =
            std::mem::take(&mut self.animation.pending_transitions);
        let animating = !entity_transitions.is_empty();

        if animating {
            self.start_per_entity_animations(&entity_transitions);
            self.ensure_gpu_capacity_and_colors();
            self.submit_animation_frame();
        } else {
            self.snap_from_prepared();
        }

        let suppress_sidechains = entity_transitions
            .values()
            .any(|t| t.suppress_initial_sidechains);
        self.upload_prepared_to_gpu(&prepared, animating, suppress_sidechains);
    }

    /// Kick off per-entity animation runners using the current
    /// positions as `start` and each entity's reference positions as
    /// `target`.
    fn start_per_entity_animations(
        &mut self,
        entity_transitions: &HashMap<u32, Transition>,
    ) {
        let targets: Vec<(EntityId, Vec<Vec3>)> = self
            .current_assembly
            .entities()
            .iter()
            .map(|entity| (entity.id(), entity.positions()))
            .collect();
        for (eid, target) in targets {
            let raw = eid.raw();
            let Some(transition) = entity_transitions.get(&raw) else {
                continue;
            };
            let current = self
                .positions
                .get(eid)
                .map(<[Vec3]>::to_vec)
                .unwrap_or_default();
            self.animation.animator.animate_entity(
                eid, current, target, transition,
            );
        }
    }

    fn snap_from_prepared(&mut self) {
        self.ensure_gpu_capacity_and_colors();
        let flat_colors = self.flat_cartoon_colors();
        if !flat_colors.is_empty() {
            self.gpu.set_colors_immediate(&flat_colors);
        }
    }

    pub(crate) fn apply_pending_animation(&mut self) {
        let _ = self.gpu.apply_pending_animation();
    }

    fn ensure_gpu_capacity_and_colors(&mut self) {
        let (backbone_chains, _na) = self.flat_scene_chains();
        let total_residues =
            crate::renderer::geometry::sheet_adjust::backbone_residue_count(
                &backbone_chains,
            );
        self.gpu.ensure_residue_capacity(total_residues);
        let flat_colors = self.flat_cartoon_colors();
        if !flat_colors.is_empty() {
            self.gpu.set_target_colors(&flat_colors);
        }
    }

    fn flat_cartoon_colors(&self) -> Vec<[f32; 3]> {
        let mut out = Vec::new();
        for (_, _, state) in self.visible_entities() {
            if !state.topology.is_protein()
                || state.drawing_mode != DrawingMode::Cartoon
            {
                continue;
            }
            if let Some(colors) = &state.topology.per_residue_colors {
                out.extend_from_slice(colors);
            } else {
                let residue_count = state.topology.residue_atom_ranges.len();
                out.extend(std::iter::repeat_n(
                    [0.5_f32, 0.5, 0.5],
                    residue_count,
                ));
            }
        }
        out
    }

    /// Submit an animation frame to the background thread using the
    /// engine's current [`super::positions::EntityPositions`].
    pub(crate) fn submit_animation_frame(&self) {
        let include_sidechains =
            self.animation.animator.should_include_sidechains();
        self.gpu.submit_animation_frame(
            &self.positions,
            &self.options.geometry,
            include_sidechains,
        );
    }

    /// Apply a trajectory frame's atom-index updates to
    /// [`super::positions::EntityPositions`].
    pub(crate) fn apply_trajectory_frame(
        &mut self,
        frame: &super::trajectory::TrajectoryFrame,
    ) {
        let Some(slot) = self.positions.get_mut(frame.entity) else {
            return;
        };
        for (i, &idx) in frame.atom_indices.iter().enumerate() {
            let Some(pos) = frame.positions.get(i).copied() else {
                continue;
            };
            if let Some(target) = slot.get_mut(idx as usize) {
                *target = pos;
            }
        }
    }
}

// ── Frustum + LOD ──

impl VisoEngine {
    /// Update sidechain instances with frustum culling when the camera
    /// moves, rebuilding the flat sidechain view from per-entity
    /// topology + positions.
    pub(crate) fn update_frustum_culling(&mut self) {
        if !self.has_any_sidechain_atoms() {
            return;
        }
        if !self.should_update_culling() {
            return;
        }

        self.gpu
            .set_last_cull_camera_eye(self.camera_controller.camera.eye);
        let frustum = self.camera_controller.frustum();

        let (positions, bonds, backbone_bonds, hydrophobicity, residue_indices) =
            self.flat_sidechain_state();
        let offset_map: HashMap<u32, Vec3> = self
            .gpu
            .backbone_sheet_offsets()
            .iter()
            .copied()
            .collect();
        let raw_view = crate::renderer::geometry::SidechainView {
            positions: &positions,
            bonds: &bonds,
            backbone_bonds: &backbone_bonds,
            hydrophobicity: &hydrophobicity,
            residue_indices: &residue_indices,
        };
        let adjusted =
            crate::renderer::geometry::sheet_adjust::sheet_adjusted_view(
                &raw_view,
                &offset_map,
            );
        let sc_colors = if self.options.display.sidechain_color_mode
            == crate::options::SidechainColorMode::Backbone
        {
            let flat = self.flat_cartoon_colors();
            if flat.is_empty() {
                None
            } else {
                Some(flat)
            }
        } else {
            None
        };
        self.gpu.upload_frustum_culled_sidechains(
            &adjusted.as_view(),
            &frustum,
            sc_colors.as_deref(),
        );
    }

    fn has_any_sidechain_atoms(&self) -> bool {
        self.entity_state
            .values()
            .any(|s| !s.topology.sidechain_layout.atom_indices.is_empty())
    }

    /// Flatten per-entity sidechain layout + positions into a single
    /// sidechain-view payload. Residue indices are offset per entity so
    /// each entity's sidechain atoms get a unique global residue index.
    #[allow(clippy::type_complexity)]
    fn flat_sidechain_state(
        &self,
    ) -> (
        Vec<Vec3>,
        Vec<(u32, u32)>,
        Vec<(Vec3, u32)>,
        Vec<bool>,
        Vec<u32>,
    ) {
        let mut positions: Vec<Vec3> = Vec::new();
        let mut bonds: Vec<(u32, u32)> = Vec::new();
        let mut backbone_bonds: Vec<(Vec3, u32)> = Vec::new();
        let mut hydrophobicity: Vec<bool> = Vec::new();
        let mut residue_indices: Vec<u32> = Vec::new();
        let mut residue_offset: u32 = 0;

        for (_, eid, state) in self.visible_entities() {
            let layout = &state.topology.sidechain_layout;
            if layout.atom_indices.is_empty() {
                residue_offset +=
                    state.topology.residue_atom_ranges.len() as u32;
                continue;
            }
            let Some(entity_positions) = self.positions.get(eid) else {
                continue;
            };
            let layout_offset = positions.len() as u32;
            for &atom_idx in &layout.atom_indices {
                let pos = entity_positions
                    .get(atom_idx as usize)
                    .copied()
                    .unwrap_or(Vec3::ZERO);
                positions.push(pos);
            }
            for &(a, b) in &layout.bonds {
                bonds.push((a + layout_offset, b + layout_offset));
            }
            for &(ca_atom_idx, layout_idx) in &layout.backbone_bonds {
                let ca = entity_positions
                    .get(ca_atom_idx as usize)
                    .copied()
                    .unwrap_or(Vec3::ZERO);
                backbone_bonds.push((ca, layout_idx + layout_offset));
            }
            hydrophobicity.extend_from_slice(&layout.hydrophobicity);
            for &ri in &layout.residue_indices {
                residue_indices.push(ri + residue_offset);
            }
            residue_offset += state.topology.residue_atom_ranges.len() as u32;
        }
        (positions, bonds, backbone_bonds, hydrophobicity, residue_indices)
    }

    fn should_update_culling(&self) -> bool {
        const CULL_UPDATE_THRESHOLD: f32 = 5.0;
        if self.animation.animator.is_animating() {
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
        self.gpu
            .check_and_submit_lod(camera_eye, &geo, &self.positions);
    }

    /// Submit a backbone-only remesh with per-chain LOD.
    pub(crate) fn submit_per_chain_lod_remesh(&self, camera_eye: Vec3) {
        let geo = self.resolved_geometry();
        self.gpu.submit_lod_remesh(camera_eye, &geo, &self.positions);
    }

    /// Geometry options with display helix/sheet style folded in.
    pub(crate) fn resolved_geometry(&self) -> GeometryOptions {
        self.options
            .geometry
            .resolve_cartoon_style()
            .with_helix_style(self.options.display.helix_style)
            .with_sheet_style(self.options.display.sheet_style)
    }

    /// Concatenated SS across all Cartoon protein entities, in
    /// assembly order. Used by the `SelectSegment` command path.
    pub(crate) fn concatenated_cartoon_ss(&self) -> Vec<SSType> {
        let mut ss = Vec::new();
        for (_, _, state) in self.visible_entities() {
            if state.topology.is_protein()
                && state.drawing_mode == DrawingMode::Cartoon
            {
                let ss_slice = state
                    .ss_override
                    .as_deref()
                    .unwrap_or(&state.topology.ss_types);
                ss.extend_from_slice(ss_slice);
            }
        }
        ss
    }

    /// Re-run the full-rebuild path after a display/color option
    /// change that affects ball-and-stick rendering.
    pub(crate) fn refresh_ball_and_stick(&mut self) {
        self.sync_scene_to_renderers(HashMap::new());
    }

    /// Recompute per-chain backbone colors and upload them immediately.
    pub(crate) fn recompute_backbone_colors(&mut self) {
        self.install_per_entity_render_data();
        let flat = self.flat_cartoon_colors();
        if !flat.is_empty() {
            self.gpu.set_colors_immediate(&flat);
        }
    }
}

// ── Helpers ──

/// Flat-index range of each protein entity in `assembly.hbonds()`.
///
/// `assembly.hbonds()` indexes into a concatenated flat array of
/// `ProteinEntity::to_backbone()` outputs in entity order. This
/// rebuilds the same per-entity offsets so sync can filter hbonds to
/// a single entity.
fn protein_hbond_ranges(
    assembly: &Assembly,
) -> FxHashMap<EntityId, (u32, u32)> {
    let mut ranges = FxHashMap::default();
    let mut offset: u32 = 0;
    for entity in assembly.entities() {
        let Some(protein) = entity.as_protein() else {
            continue;
        };
        let n = protein.residues.len() as u32;
        let _ = ranges.insert(protein.id, (offset, offset + n));
        offset += n;
    }
    ranges
}

fn entity_sheet_plane_normals(
    assembly: &Assembly,
    eid: EntityId,
    ss_types: &[SSType],
    positions: &[Vec3],
    topology: &EntityTopology,
    ranges: &FxHashMap<EntityId, (u32, u32)>,
) -> Vec<(u32, Vec3)> {
    let Some(&(start, end)) = ranges.get(&eid) else {
        return Vec::new();
    };
    if end <= start {
        return Vec::new();
    }

    let hbonds_slice: Vec<molex::HBond> = assembly
        .hbonds()
        .iter()
        .filter_map(|h| {
            let donor = h.donor as u32;
            let acceptor = h.acceptor as u32;
            if donor >= start
                && donor < end
                && acceptor >= start
                && acceptor < end
            {
                Some(molex::HBond {
                    donor: (donor - start) as usize,
                    acceptor: (acceptor - start) as usize,
                    energy: h.energy,
                })
            } else {
                None
            }
        })
        .collect();

    let ca_positions: Vec<Vec3> = topology
        .residue_atom_ranges
        .iter()
        .map(|range| {
            let ca_idx = range.start as usize + 1;
            positions.get(ca_idx).copied().unwrap_or(Vec3::ZERO)
        })
        .collect();

    crate::renderer::geometry::backbone::sheet_fit::compute_sheet_plane_normals(
        &hbonds_slice,
        ss_types,
        &ca_positions,
    )
}

fn per_entity_colors(
    entity_index: usize,
    backbone_chains: &[Vec<Vec3>],
    ss_types: &[SSType],
    scores: Option<&[f64]>,
    display: &DisplayOptions,
) -> Option<Vec<[f32; 3]>> {
    if backbone_chains.is_empty() {
        return None;
    }
    let scores_slice = [scores];
    let colors =
        crate::options::score_color::compute_per_residue_colors_styled(
            backbone_chains,
            ss_types,
            &scores_slice,
            &display.backbone_color_scheme,
            &display.backbone_palette(),
            entity_index,
        );
    if colors.is_empty() {
        None
    } else {
        Some(colors)
    }
}
