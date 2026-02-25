//! Scene Management methods for ProteinRenderEngine

use std::collections::HashMap;

use foldit_conv::{
    coords::{MoleculeEntity, MoleculeType},
    secondary_structure::SSType,
};
use glam::Vec3;

use super::ProteinRenderEngine;
use crate::{
    animation::transition::Transition,
    renderer::geometry::{
        band::BandRenderInfo, pull::PullRenderInfo, sidechain::SidechainData,
    },
};

impl ProteinRenderEngine {
    /// Update backbone with new chains (regenerates the backbone mesh)
    /// Use this for designed backbones from ML models like RFDiffusion3
    pub fn update_backbone(&mut self, backbone_chains: &[Vec<Vec3>]) {
        self.backbone_renderer.update(
            &self.context.device,
            &self.context.queue,
            backbone_chains,
            &[], // NA chains unchanged
            None,
            &self.options.geometry,
        );
    }

    /// Update sidechain instances with frustum culling when camera moves
    /// significantly. This filters out sidechains behind the camera to
    /// reduce draw calls.
    pub(crate) fn update_frustum_culling(&mut self) {
        // Skip if no sidechain data
        if self.sc.target_sidechain_positions.is_empty() {
            return;
        }

        let camera_eye = self.camera_controller.camera.eye;
        let camera_delta = (camera_eye - self.sc.last_cull_camera_eye).length();

        // Only update culling when camera moves more than 5 units
        // This prevents expensive updates on minor camera movements.
        // Exception: always update during animation so sidechain positions
        // reflect the interpolated state (not the final target uploaded by
        // apply_pending_scene).
        const CULL_UPDATE_THRESHOLD: f32 = 5.0;
        let animating =
            self.animator.is_animating() && self.animator.has_sidechain_data();
        if camera_delta < CULL_UPDATE_THRESHOLD && !animating {
            return;
        }

        self.sc.last_cull_camera_eye = camera_eye;

        // Get current frustum
        let frustum = self.camera_controller.frustum();

        // Get current sidechain positions (may be interpolated during
        // animation)
        let positions = if self.animator.is_animating()
            && self.animator.has_sidechain_data()
        {
            self.animator.get_sidechain_positions()
        } else {
            self.sc.target_sidechain_positions.clone()
        };

        // Get current backbone-sidechain bonds (may be interpolated)
        let bs_bonds = if self.animator.is_animating() {
            // Interpolate CA positions
            self.sc
                .target_backbone_sidechain_bonds
                .iter()
                .map(|(target_ca, cb_idx)| {
                    let res_idx =
                        self.sc
                            .cached_sidechain_residue_indices
                            .get(*cb_idx as usize)
                            .copied()
                            .unwrap_or(0) as usize;
                    let ca_pos = self
                        .animator
                        .get_ca_position(res_idx)
                        .unwrap_or(*target_ca);
                    (ca_pos, *cb_idx)
                })
                .collect::<Vec<_>>()
        } else {
            self.sc.target_backbone_sidechain_bonds.clone()
        };

        // Translate entire sidechains onto sheet surface
        let offset_map = self.sheet_offset_map();
        let res_indices = self.sc.cached_sidechain_residue_indices.clone();
        let adjusted_positions =
            crate::util::sheet_adjust::adjust_sidechains_for_sheet(
                &positions,
                &res_indices,
                &offset_map,
            );
        let adjusted_bonds = crate::util::sheet_adjust::adjust_bonds_for_sheet(
            &bs_bonds,
            &res_indices,
            &offset_map,
        );

        // Update sidechains with frustum culling
        self.sidechain_renderer.update_with_frustum(
            &self.context.device,
            &self.context.queue,
            &SidechainData {
                positions: &adjusted_positions,
                bonds: &self.sc.cached_sidechain_bonds,
                backbone_bonds: &adjusted_bonds,
                hydrophobicity: &self.sc.cached_sidechain_hydrophobicity,
                residue_indices: &self.sc.cached_sidechain_residue_indices,
            },
            Some(&frustum),
        );

        // Recreate picking bind group since buffer may have changed
        self.picking_groups.rebuild_capsule(
            &self.picking,
            &self.context.device,
            &self.sidechain_renderer,
        );
    }

    /// Refresh ball-and-stick renderer with current visibility flags.
    pub(crate) fn refresh_ball_and_stick(&mut self) {
        // Collect all non-protein entities from visible scene entities
        let entities: Vec<MoleculeEntity> = self
            .scene
            .entities()
            .iter()
            .filter(|se| se.visible)
            .map(|se| &se.entity)
            .filter(|e| {
                e.molecule_type != MoleculeType::Protein
                    && !matches!(
                        e.molecule_type,
                        MoleculeType::DNA | MoleculeType::RNA
                    )
            })
            .cloned()
            .collect();
        self.ball_and_stick_renderer.update_from_entities(
            &self.context.device,
            &self.context.queue,
            &entities,
            &self.options.display,
            Some(&self.options.colors),
        );
        // Recreate picking bind group
        self.picking_groups.rebuild_bns(
            &self.picking,
            &self.context.device,
            &self.ball_and_stick_renderer,
        );
    }

    /// Set SS override (from puzzle.toml annotation). Updates cached types
    /// and forces backbone renderer regeneration.
    pub fn set_ss_override(&mut self, ss_types: &[SSType]) {
        self.sc.cached_ss_types = ss_types.to_vec();
        self.backbone_renderer
            .set_ss_override(Some(ss_types.to_vec()));
        let camera_eye = self.camera_controller.camera.eye;
        self.submit_per_chain_lod_remesh(camera_eye);
    }

    /// Compute secondary structure types for all residues across all chains
    pub(crate) fn compute_ss_types(
        &self,
        backbone_chains: &[Vec<Vec3>],
    ) -> Vec<SSType> {
        use foldit_conv::secondary_structure::auto::detect as detect_ss;

        let mut all_ss_types = Vec::new();

        for chain in backbone_chains {
            // Extract CA positions (every 3rd atom starting at index 1: N, CA,
            // C pattern)
            let ca_positions: Vec<Vec3> = chain
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 3 == 1)
                .map(|(_, &pos)| pos)
                .collect();

            let ss_types = detect_ss(&ca_positions);
            all_ss_types.extend(ss_types);
        }

        all_ss_types
    }

    /// Build a map of sheet residue offsets (residue_idx -> offset vector).
    pub(crate) fn sheet_offset_map(&self) -> HashMap<u32, Vec3> {
        self.backbone_renderer
            .sheet_offsets()
            .iter()
            .copied()
            .collect()
    }

    /// Update the band visualization.
    /// Call this when bands are added, removed, or modified.
    pub fn update_bands(&mut self, bands: &[BandRenderInfo]) {
        self.band_renderer.update(
            &self.context.device,
            &self.context.queue,
            bands,
            Some(&self.options.colors),
        );
    }

    /// Update the pull visualization (only one pull at a time).
    /// Pass None to clear the pull visualization.
    pub fn update_pull(&mut self, pull: Option<&PullRenderInfo>) {
        self.pull_renderer.update(
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
