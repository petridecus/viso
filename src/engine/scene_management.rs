//! Scene Management methods for VisoEngine

use std::collections::HashMap;

use foldit_conv::secondary_structure::SSType;
use foldit_conv::types::entity::MoleculeEntity;
use glam::Vec3;

use super::command::{BandInfo, PullInfo};
use super::VisoEngine;
use crate::animation::transition::Transition;
use crate::renderer::geometry::backbone::BackboneUpdateData;
use crate::renderer::geometry::sidechain::SidechainView;

impl VisoEngine {
    /// Update backbone with new chains (regenerates the backbone mesh)
    /// Use this for designed backbones from ML models like RFDiffusion3
    pub fn update_backbone(&mut self, backbone_chains: &[Vec<Vec3>]) {
        self.renderers.backbone.update(
            &self.context,
            &BackboneUpdateData {
                protein_chains: backbone_chains,
                na_chains: &[],
                ss_types: None,
                geometry: &self.options.geometry,
            },
        );
    }

    /// Update sidechain instances with frustum culling when camera moves
    /// significantly. This filters out sidechains behind the camera to
    /// reduce draw calls.
    pub(crate) fn update_frustum_culling(&mut self) {
        // Skip if no sidechain data
        if self.sc_cache.target_sidechain_positions.is_empty() {
            return;
        }

        // Only update culling when camera moves more than 5 units.
        // Exception: always update during animation so sidechain positions
        // reflect the interpolated state.
        if !self.should_update_culling() {
            return;
        }

        let camera_eye = self.camera_controller.camera.eye;
        self.last_cull_camera_eye = camera_eye;

        let frustum = self.camera_controller.frustum();
        let positions = self.get_current_sidechain_positions();
        let bs_bonds = self.get_current_backbone_sidechain_bonds();

        // Translate sidechains onto sheet surface and apply frustum culling
        let offset_map = self.sheet_offset_map();
        let raw_view = SidechainView {
            positions: &positions,
            bonds: &self.sc_cache.sidechain_bonds,
            backbone_bonds: &bs_bonds,
            hydrophobicity: &self.sc_cache.sidechain_hydrophobicity,
            residue_indices: &self.sc_cache.sidechain_residue_indices,
        };
        let adjusted = crate::util::sheet_adjust::sheet_adjusted_view(
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

    /// Whether frustum culling should be recalculated this frame.
    ///
    /// Returns `true` when the camera has moved more than the threshold or
    /// an animation with sidechain data is active (positions are
    /// interpolated and need continuous updates).
    fn should_update_culling(&self) -> bool {
        const CULL_UPDATE_THRESHOLD: f32 = 5.0;

        let animating = self.animator.is_animating()
            && !self.sc_cache.target_sidechain_positions.is_empty();
        if animating {
            return true;
        }

        let camera_eye = self.camera_controller.camera.eye;
        let camera_delta = (camera_eye - self.last_cull_camera_eye).length();
        camera_delta >= CULL_UPDATE_THRESHOLD
    }

    /// Refresh ball-and-stick renderer with current visibility flags.
    pub(crate) fn refresh_ball_and_stick(&mut self) {
        // Collect all ligand entities (not protein, not nucleic acid)
        let entities: Vec<MoleculeEntity> = self
            .scene
            .ligand_entities()
            .iter()
            .map(|se| se.entity.clone())
            .collect();
        self.renderers.ball_and_stick.update_from_entities(
            &self.context,
            &entities,
            &self.options.display,
            Some(&self.options.colors),
        );
        // Recreate picking bind groups
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

    /// Set SS override (from puzzle.toml annotation). Updates cached types
    /// and forces backbone renderer regeneration.
    pub fn set_ss_override(&mut self, ss_types: &[SSType]) {
        self.sc_cache.ss_types = ss_types.to_vec();
        self.renderers
            .backbone
            .set_ss_override(Some(ss_types.to_vec()));
        let camera_eye = self.camera_controller.camera.eye;
        self.submit_per_chain_lod_remesh(camera_eye);
    }

    /// Compute secondary structure types for all residues across all chains
    pub(crate) fn compute_ss_types(
        backbone_chains: &[Vec<Vec3>],
    ) -> Vec<SSType> {
        use foldit_conv::secondary_structure::auto::detect as detect_ss;

        let mut all_ss_types = Vec::new();

        for chain in backbone_chains {
            let ca_positions: Vec<Vec3> =
                foldit_conv::render::backbone::ca_positions_from_chains(
                    std::slice::from_ref(chain),
                );
            let ss_types = detect_ss(&ca_positions);
            all_ss_types.extend(ss_types);
        }

        all_ss_types
    }

    /// Build a map of sheet residue offsets (residue_idx -> offset vector).
    pub(crate) fn sheet_offset_map(&self) -> HashMap<u32, Vec3> {
        self.renderers
            .backbone
            .sheet_offsets()
            .iter()
            .copied()
            .collect()
    }

    /// Update the band visualization.
    /// Call this when bands are added, removed, or modified.
    pub fn update_bands(&mut self, bands: &[BandInfo]) {
        self.renderers.band.update(
            &self.context.device,
            &self.context.queue,
            bands,
            Some(&self.options.colors),
        );
    }

    /// Update the pull visualization (only one pull at a time).
    /// Pass None to clear the pull visualization.
    pub fn update_pull(&mut self, pull: Option<&PullInfo>) {
        self.renderers.pull.update(
            &self.context.device,
            &self.context.queue,
            pull,
        );
    }

    /// Load entities into the scene. Optionally fits camera.
    /// Returns the assigned entity IDs.
    pub fn load_entities(
        &mut self,
        entities: Vec<MoleculeEntity>,
        fit_camera: bool,
    ) -> Vec<u32> {
        // Store canonical copy on the engine (source of truth)
        self.entities.clone_from(&entities);

        let ids = self.scene.add_entities(entities);
        if fit_camera {
            // Sync immediately so entity data is available for camera fit
            self.sync_scene_to_renderers(Some(Transition::snap()));
            let positions = self.scene.all_positions();
            if !positions.is_empty() {
                self.camera_controller.fit_to_positions(&positions);
            }
        }
        ids
    }
}
