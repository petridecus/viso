//! All GPU infrastructure grouped together.

use std::collections::HashMap;

use glam::{Mat4, Vec3};
use molex::secondary_structure::SSType;
use molex::types::entity::MoleculeEntity;

use crate::animation::AnimationFrame;
use crate::camera::controller::CameraController;
use crate::camera::core::Camera;
use crate::engine::scene::{SceneTopology, SidechainTopology, VisualState};
use crate::gpu::lighting::Lighting;
use crate::gpu::{RenderContext, ShaderComposer};
use crate::options::{
    ColorOptions, DisplayOptions, GeometryOptions, LightingOptions, VisoOptions,
};
use crate::renderer::draw_context::DrawBindGroups;
use crate::renderer::geometry::{
    PreparedBackboneData, PreparedBallAndStickData, SidechainView,
};
use crate::renderer::picking::PickingSystem;
use crate::renderer::pipeline::prepared::PreparedScene;
use crate::renderer::pipeline::{SceneProcessor, SceneRequest};
use crate::renderer::postprocess::post_process::PostProcessCamera;
use crate::renderer::postprocess::PostProcessStack;
use crate::renderer::{GeometryPassInput, Renderers};

/// Borrowed scene chain data needed by [`GpuPipeline::upload_prepared`].
pub(crate) struct SceneChainData<'a> {
    /// Backbone chains (interpolated or at-rest).
    pub backbone_chains: &'a [Vec<Vec3>],
    /// Nucleic-acid chains.
    pub na_chains: &'a [Vec<Vec3>],
    /// Per-residue secondary-structure types.
    pub ss_types: &'a [SSType],
}

/// All GPU infrastructure grouped together: device/queue, renderers,
/// picking, background mesh processor, post-processing, lighting, and
/// per-frame cursor/culling state.
pub(crate) struct GpuPipeline {
    /// Core wgpu device, queue, and surface.
    pub context: RenderContext,
    /// All geometry renderers (backbone, sidechain, band, pull,
    /// ball-and-stick, nucleic acid).
    pub renderers: Renderers,
    /// GPU picking, selection, and per-residue color buffers.
    pub pick: PickingSystem,
    /// Background thread for off-main-thread mesh generation.
    pub scene_processor: SceneProcessor,
    /// Post-processing pass stack (SSAO, bloom, composite, FXAA).
    pub post_process: PostProcessStack,
    /// GPU lighting uniform and bind group.
    pub lighting: Lighting,
    /// Current cursor position in physical pixels (set by the viewer /
    /// input processor each frame for GPU picking).
    pub cursor_pos: (f32, f32),
    /// Camera eye position at the last frustum-culling update.
    pub last_cull_camera_eye: Vec3,
    /// Retained so compiled shader modules stay alive for the engine lifetime.
    #[allow(dead_code)]
    pub(crate) shader_composer: ShaderComposer,
}

impl GpuPipeline {
    /// Core render — geometry, post-process, picking — targeting the given
    /// view. Returns the encoder so the caller can submit it.
    pub(crate) fn render_to_view(
        &mut self,
        view: &wgpu::TextureView,
        camera: &CameraController,
        show_sidechains: bool,
    ) -> wgpu::CommandEncoder {
        let mut encoder = self.context.create_encoder();

        // Geometry pass
        let input = GeometryPassInput {
            color: self.post_process.color_view(),
            normal: &self.post_process.normal_view,
            depth: &self.post_process.depth_view,
            show_sidechains,
        };
        let bind_groups = DrawBindGroups {
            camera: &camera.bind_group,
            lighting: &self.lighting.bind_group,
            selection: &self.pick.selection.bind_group,
            color: Some(&self.pick.residue_colors.bind_group),
        };
        let frustum = camera.frustum();
        self.renderers.encode_geometry_pass(
            &mut encoder,
            &input,
            &bind_groups,
            &frustum,
        );

        // Post-processing: SSAO → bloom → composite → FXAA
        let cam = &camera.camera;
        self.post_process.render(
            &mut encoder,
            &self.context.queue,
            &PostProcessCamera {
                proj: cam.build_projection(),
                view_matrix: Mat4::look_at_rh(cam.eye, cam.target, cam.up),
                znear: cam.znear,
                zfar: cam.zfar,
            },
            view.clone(),
        );

        // GPU Picking pass
        let picking_geometry =
            self.pick.build_geometry(&self.renderers, show_sidechains);
        self.pick.picking.render(
            &mut encoder,
            &camera.bind_group,
            &picking_geometry,
            (self.cursor_pos.0 as u32, self.cursor_pos.1 as u32),
        );

        encoder
    }

    /// Resize all GPU surfaces to match the new window size.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.context.resize(width, height);
        self.post_process.resize(&self.context);
        self.pick
            .picking
            .resize(&self.context.device, width, height);
    }

    /// Upload prepared scene geometry to GPU renderers.
    pub(crate) fn upload_prepared(
        &mut self,
        prepared: &PreparedScene,
        animating: bool,
        suppress_sidechains: bool,
        scene: &SceneChainData<'_>,
    ) {
        let ss_override = if scene.ss_types.is_empty() {
            None
        } else {
            Some(scene.ss_types.to_vec())
        };

        if animating {
            self.renderers.backbone.update_metadata(
                scene.backbone_chains,
                scene.na_chains,
                ss_override,
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
                    cached_chains: scene.backbone_chains,
                    cached_na_chains: scene.na_chains,
                    ss_override,
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
        self.upload_non_backbone(prepared);
    }

    /// Upload BnS, NA, and pick data (shared by animating and non-animating).
    fn upload_non_backbone(&mut self, prepared: &PreparedScene) {
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
        self.pick.pick_map = Some(prepared.pick_map.clone());
        self.pick.groups.rebuild_all(
            &self.pick.picking,
            &self.context.device,
            &self.renderers.sidechain,
            &self.renderers.ball_and_stick,
        );
    }

    /// Apply any pending animation frame from the background thread.
    ///
    /// Returns `true` if a frame was applied, `false` otherwise.
    pub(crate) fn apply_pending_animation(&mut self) -> bool {
        let Some(prepared) = self.scene_processor.try_recv_animation() else {
            return false;
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

        true
    }

    /// Submit an animation frame to the background thread for mesh
    /// generation, using a unified [`AnimationFrame`] from the animator.
    pub(crate) fn submit_animation_frame(
        &self,
        frame: &AnimationFrame,
        sidechain_topology: &SidechainTopology,
        geometry: &GeometryOptions,
    ) {
        let has_sc = !sidechain_topology.target_positions.is_empty()
            && frame.sidechains_visible;

        let sidechains = if has_sc {
            let positions = frame
                .sidechain_positions
                .as_deref()
                .unwrap_or(&sidechain_topology.target_positions);
            let bonds = frame
                .backbone_sidechain_bonds
                .as_deref()
                .unwrap_or(&sidechain_topology.target_backbone_bonds);
            Some(sidechain_topology.to_interpolated_atoms(positions, bonds))
        } else {
            None
        };

        self.scene_processor.submit(SceneRequest::AnimationFrame {
            backbone_chains: frame.backbone_chains.clone(),
            na_chains: None,
            sidechains,
            ss_types: None,
            per_residue_colors: None,
            geometry: geometry.clone(),
            per_chain_lod: None,
            generation: self.scene_processor.generation(),
        });
    }

    /// Submit a backbone-only remesh with per-chain LOD to the background
    /// thread. Each chain gets its own `(spr, csv)` based on its distance
    /// from the camera. No sidechains — they don't change with LOD.
    ///
    /// The base geometry is first clamped via
    /// `GeometryOptions::clamped_for_residues` to stay within the 256 MB
    /// buffer limit, then each chain is further scaled by its distance tier.
    /// For very large structures (>50 K residues) this per-chain scaling is
    /// critical — without it the vertex buffer can exceed GPU limits.
    pub(crate) fn submit_lod_remesh(
        &self,
        camera_eye: Vec3,
        geometry: &GeometryOptions,
    ) {
        // While a FullRebuild is in flight, the backbone renderer's
        // cached chains are stale. Submitting a LOD remesh now would
        // produce an AnimationFrame with old backbone geometry that
        // could overwrite the correct PreparedScene upload.
        if self.scene_processor.is_scene_pending() {
            return;
        }
        use crate::options::{lod_scaled, select_chain_lod_tier};

        // Use clamped geometry as the base for LOD scaling
        let total_residues =
            crate::renderer::geometry::sheet_adjust::backbone_residue_count(
                self.renderers.backbone.cached_chains(),
            ) + self
                .renderers
                .backbone
                .cached_na_chains()
                .iter()
                .map(Vec::len)
                .sum::<usize>();
        let base_geo = geometry.clamped_for_residues(total_residues);
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

        self.scene_processor.submit(SceneRequest::AnimationFrame {
            backbone_chains: self.renderers.backbone.cached_chains().to_vec(),
            na_chains: None,
            sidechains: None,
            ss_types: None,
            per_residue_colors: None,
            geometry: base_geo,
            per_chain_lod: Some(per_chain_lod),
            generation: self.scene_processor.generation(),
        });
    }

    /// Check per-chain LOD tiers and submit a background remesh if any
    /// chain's tier has changed. Skipped while a `FullRebuild` is
    /// pending — the backbone renderer's cached chains are stale.
    pub(crate) fn check_and_submit_lod(
        &mut self,
        camera_eye: Vec3,
        geometry: &GeometryOptions,
    ) {
        if self.scene_processor.is_scene_pending() {
            return;
        }
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
            self.submit_lod_remesh(camera_eye, geometry);
        }
    }

    /// Push lighting options to the GPU uniform.
    pub(crate) fn apply_lighting(&mut self, lo: &LightingOptions) {
        self.lighting.apply_options(lo, &self.context.queue);
    }

    /// Push post-processing options to the composite pass.
    pub(crate) fn apply_post_processing(&mut self, options: &VisoOptions) {
        self.post_process
            .apply_options(options, &self.context.queue);
    }

    /// Refresh ball-and-stick renderer with current visibility flags.
    pub(crate) fn refresh_ball_and_stick(
        &mut self,
        entities: impl Iterator<Item = MoleculeEntity>,
        display: &DisplayOptions,
        colors: Option<&ColorOptions>,
    ) {
        let entities: Vec<MoleculeEntity> = entities.collect();
        self.renderers.ball_and_stick.update_from_entities(
            &self.context,
            &entities,
            display,
            colors,
        );
        self.pick.groups.rebuild_bns_bond(
            &self.pick.picking,
            &self.context.device,
            &self.renderers.ball_and_stick,
        );
        self.pick.groups.rebuild_bns_sphere(
            &self.pick.picking,
            &self.context.device,
            &self.renderers.ball_and_stick,
        );
    }

    /// Update headlamp direction from the camera and push to the GPU.
    pub(crate) fn update_headlamp(&mut self, camera: &Camera) {
        self.lighting.update_headlamp_from_camera(camera);
        self.lighting.update_gpu(&self.context.queue);
    }

    /// Stop the background scene processor thread.
    pub(crate) fn shutdown(&mut self) {
        self.scene_processor.shutdown();
    }

    /// Ensure the selection and residue-color buffers have enough capacity.
    pub(crate) fn ensure_residue_capacity(&mut self, total_residues: usize) {
        self.pick
            .selection
            .ensure_capacity(&self.context.device, total_residues);
        self.pick
            .residue_colors
            .ensure_capacity(&self.context.device, total_residues);
    }

    /// Immediately upload per-residue colors (no transition).
    pub(crate) fn set_colors_immediate(&mut self, colors: &[[f32; 3]]) {
        self.pick
            .residue_colors
            .set_colors_immediate(&self.context.queue, colors);
    }

    /// Set target per-residue colors (animated transition).
    pub(crate) fn set_target_colors(&mut self, colors: &[[f32; 3]]) {
        self.pick.residue_colors.set_target_colors(colors);
    }

    /// Update sidechain instances with frustum culling.
    ///
    /// Builds the sidechain view from visual/topology state, applies
    /// sheet-surface adjustment, frustum-culls, and uploads to GPU.
    /// Rebuilds the capsule pick bind group afterward.
    pub(crate) fn update_frustum_culling(
        &mut self,
        camera: &CameraController,
        visual: &VisualState,
        topology: &SceneTopology,
    ) {
        self.last_cull_camera_eye = camera.camera.eye;

        let frustum = camera.frustum();
        let positions = if visual.sidechain_positions.is_empty() {
            &topology.sidechain_topology.target_positions
        } else {
            &visual.sidechain_positions
        };
        let bs_bonds = if visual.backbone_sidechain_bonds.is_empty() {
            topology.sidechain_topology.target_backbone_bonds.clone()
        } else {
            visual.backbone_sidechain_bonds.clone()
        };

        // Build sheet offset map from backbone renderer state
        let offset_map: HashMap<u32, Vec3> = self
            .renderers
            .backbone
            .sheet_offsets()
            .iter()
            .copied()
            .collect();

        // Translate sidechains onto sheet surface and apply frustum culling
        let raw_view = SidechainView {
            positions,
            bonds: &topology.sidechain_topology.bonds,
            backbone_bonds: &bs_bonds,
            hydrophobicity: &topology.sidechain_topology.hydrophobicity,
            residue_indices: &topology.sidechain_topology.residue_indices,
        };
        let adjusted =
            crate::renderer::geometry::sheet_adjust::sheet_adjusted_view(
                &raw_view,
                &offset_map,
            );

        self.renderers.sidechain.update_with_frustum(
            &self.context.device,
            &self.context.queue,
            &adjusted.as_view(),
            Some(&frustum),
        );

        // Recreate picking bind group since buffer may have changed
        self.pick.groups.rebuild_capsule(
            &self.pick.picking,
            &self.context.device,
            &self.renderers.sidechain,
        );
    }
}
