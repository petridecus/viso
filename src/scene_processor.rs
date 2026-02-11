//! Background scene processor for non-blocking geometry generation.
//!
//! Moves all CPU-heavy mesh/instance generation off the main thread.
//! The main thread only does GPU uploads (<1ms) and render passes.

use crate::ball_and_stick_renderer::BallAndStickRenderer;
use crate::capsule_sidechain_renderer::CapsuleSidechainRenderer;
use crate::engine::ViewMode;
use crate::nucleic_acid_renderer::NucleicAcidRenderer;
use crate::options::{ColorOptions, DisplayOptions};
use crate::ribbon_renderer::{RibbonParams, RibbonRenderer};
use crate::scene::AggregatedRenderData;
use crate::tube_renderer::TubeRenderer;
use foldit_conv::coords::entity::NucleotideRing;
use foldit_conv::coords::MoleculeEntity;
use foldit_conv::secondary_structure::SSType;
use crate::animation::AnimationAction;
use glam::Vec3;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::sync::Arc;

/// Sidechain data bundled for animation frame processing.
pub struct AnimationSidechainData {
    pub sidechain_positions: Vec<Vec3>,
    pub sidechain_bonds: Vec<(u32, u32)>,
    pub backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    pub sidechain_hydrophobicity: Vec<bool>,
    pub sidechain_residue_indices: Vec<u32>,
}

/// Request sent from main thread to scene processor.
pub enum SceneRequest {
    /// Full scene rebuild with all aggregated data.
    FullRebuild {
        aggregated: Arc<AggregatedRenderData>,
        action: Option<AnimationAction>,
        view_mode: ViewMode,
        display: DisplayOptions,
        colors: ColorOptions,
    },
    /// Per-frame animation mesh generation (tube + ribbon + optional sidechains).
    AnimationFrame {
        backbone_chains: Vec<Vec<Vec3>>,
        sidechains: Option<AnimationSidechainData>,
        view_mode: ViewMode,
        ss_types: Option<Vec<SSType>>,
    },
    /// Shut down the background thread.
    Shutdown,
}

/// All pre-computed CPU data, ready for GPU-only upload on the main thread.
#[derive(Clone)]
pub struct PreparedScene {
    // Tube mesh
    pub tube_vertices: Vec<u8>,
    pub tube_indices: Vec<u8>,
    pub tube_index_count: u32,

    // Ribbon mesh
    pub ribbon_vertices: Vec<u8>,
    pub ribbon_indices: Vec<u8>,
    pub ribbon_index_count: u32,
    pub sheet_offsets: Vec<(u32, Vec3)>,

    // Sidechain capsules
    pub sidechain_instances: Vec<u8>,
    pub sidechain_instance_count: u32,

    // Ball-and-stick instances
    pub bns_sphere_instances: Vec<u8>,
    pub bns_sphere_count: u32,
    pub bns_capsule_instances: Vec<u8>,
    pub bns_capsule_count: u32,
    pub bns_picking_capsules: Vec<u8>,
    pub bns_picking_count: u32,

    // Nucleic acid mesh
    pub na_vertices: Vec<u8>,
    pub na_indices: Vec<u8>,
    pub na_index_count: u32,

    // Passthrough data for animation setup (fast array copies on main thread)
    pub backbone_chains: Vec<Vec<Vec3>>,
    pub sidechain_positions: Vec<Vec3>,
    pub sidechain_bonds: Vec<(u32, u32)>,
    pub sidechain_hydrophobicity: Vec<bool>,
    pub sidechain_residue_indices: Vec<u32>,
    pub sidechain_atom_names: Vec<String>,
    pub backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    pub ss_types: Option<Vec<SSType>>,
    pub all_positions: Vec<Vec3>,
    pub action: Option<AnimationAction>,
    pub non_protein_entities: Vec<MoleculeEntity>,
    pub nucleic_acid_chains: Vec<Vec<Vec3>>,
    pub nucleic_acid_rings: Vec<NucleotideRing>,
}

/// Pre-computed animation frame data, ready for GPU upload.
#[derive(Clone)]
pub struct PreparedAnimationFrame {
    pub tube_vertices: Vec<u8>,
    pub tube_indices: Vec<u8>,
    pub tube_index_count: u32,
    pub ribbon_vertices: Vec<u8>,
    pub ribbon_indices: Vec<u8>,
    pub ribbon_index_count: u32,
    pub sheet_offsets: Vec<(u32, Vec3)>,
    pub sidechain_instances: Option<Vec<u8>>,
    pub sidechain_instance_count: u32,
}

pub struct SceneProcessor {
    request_tx: mpsc::Sender<SceneRequest>,
    scene_result: triple_buffer::Output<Option<PreparedScene>>,
    anim_result: triple_buffer::Output<Option<PreparedAnimationFrame>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SceneProcessor {
    /// Spawn the background scene processing thread.
    pub fn new() -> Self {
        let (request_tx, request_rx) = mpsc::channel::<SceneRequest>();
        let (scene_input, scene_output) = triple_buffer::triple_buffer(&None);
        let (anim_input, anim_output) = triple_buffer::triple_buffer(&None);

        let thread = std::thread::Builder::new()
            .name("scene-processor".into())
            .spawn(move || {
                Self::thread_loop(request_rx, scene_input, anim_input);
            })
            .expect("Failed to spawn scene processor thread");

        Self {
            request_tx,
            scene_result: scene_output,
            anim_result: anim_output,
            thread: Some(thread),
        }
    }

    /// Submit a scene request (non-blocking send).
    pub fn submit(&self, request: SceneRequest) {
        let _ = self.request_tx.send(request);
    }

    /// Non-blocking check for completed full scene rebuild.
    /// Always returns the most recent result, skipping any stale intermediates.
    pub fn try_recv_scene(&mut self) -> Option<PreparedScene> {
        self.scene_result.update();
        self.scene_result.output_buffer_mut().take()
    }

    /// Non-blocking check for completed animation frame.
    /// Always returns the most recent result, skipping any stale intermediates.
    pub fn try_recv_animation(&mut self) -> Option<PreparedAnimationFrame> {
        self.anim_result.update();
        self.anim_result.output_buffer_mut().take()
    }

    /// Shut down the background thread and wait for it to finish.
    pub fn shutdown(&mut self) {
        let _ = self.request_tx.send(SceneRequest::Shutdown);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }

    /// Background thread main loop.
    fn thread_loop(
        request_rx: mpsc::Receiver<SceneRequest>,
        mut scene_input: triple_buffer::Input<Option<PreparedScene>>,
        mut anim_input: triple_buffer::Input<Option<PreparedAnimationFrame>>,
    ) {
        loop {
            // Block waiting for the next request
            let request = match request_rx.recv() {
                Ok(r) => r,
                Err(_) => break, // Channel closed
            };

            // Drain any queued requests, keep only the latest.
            // FullRebuild always supersedes AnimationFrame (not vice versa).
            let mut latest = request;
            while let Ok(newer) = request_rx.try_recv() {
                match (&latest, &newer) {
                    // Don't let AnimationFrame supersede FullRebuild
                    (SceneRequest::FullRebuild { .. }, SceneRequest::AnimationFrame { .. }) => {}
                    _ => { latest = newer; }
                }
            }

            match latest {
                SceneRequest::Shutdown => break,
                SceneRequest::FullRebuild {
                    aggregated,
                    action,
                    view_mode,
                    display,
                    colors,
                } => {
                    let prepared = Self::process_full_rebuild(
                        &aggregated, action, view_mode, &display, &colors,
                    );
                    scene_input.write(Some(prepared));
                }
                SceneRequest::AnimationFrame {
                    backbone_chains,
                    sidechains,
                    view_mode,
                    ss_types,
                } => {
                    let prepared = Self::process_animation_frame(
                        backbone_chains, sidechains, view_mode, ss_types,
                    );
                    anim_input.write(Some(prepared));
                }
            }
        }
    }

    /// Generate all scene geometry on the background thread.
    ///
    /// Takes `&AggregatedRenderData` (borrowed through Arc) so that the expensive
    /// clone of passthrough fields happens here on the background thread rather
    /// than blocking the main render loop.
    fn process_full_rebuild(
        agg: &AggregatedRenderData,
        action: Option<AnimationAction>,
        view_mode: ViewMode,
        display: &DisplayOptions,
        colors: &ColorOptions,
    ) -> PreparedScene {
        // --- Tube mesh ---
        let tube_filter = match view_mode {
            ViewMode::Ribbon => {
                let mut coil_only = HashSet::new();
                coil_only.insert(SSType::Coil);
                Some(coil_only)
            }
            ViewMode::Tube => None,
        };
        let (tube_verts, tube_inds) = TubeRenderer::generate_tube_mesh(
            &agg.backbone_chains,
            &tube_filter,
            agg.ss_types.as_deref(),
        );
        let tube_index_count = tube_inds.len() as u32;
        let tube_vertices = bytemuck::cast_slice(&tube_verts).to_vec();
        let tube_indices = bytemuck::cast_slice(&tube_inds).to_vec();

        // --- Ribbon mesh ---
        let params = RibbonParams::default();
        let (ribbon_verts, ribbon_inds, sheet_offsets) = RibbonRenderer::generate_from_ca_only(
            &agg.backbone_chains,
            &params,
            agg.ss_types.as_deref(),
        );
        let ribbon_index_count = ribbon_inds.len() as u32;
        let ribbon_vertices = bytemuck::cast_slice(&ribbon_verts).to_vec();
        let ribbon_indices = bytemuck::cast_slice(&ribbon_inds).to_vec();

        // --- Sidechain capsules ---
        // Adjust sidechain positions/bonds for sheet surface offsets
        let offset_map: HashMap<u32, Vec3> = if view_mode == ViewMode::Ribbon {
            sheet_offsets.iter().copied().collect()
        } else {
            HashMap::new()
        };
        let adjusted_positions = adjust_sidechains_for_sheet(
            &agg.sidechain_positions,
            &agg.sidechain_residue_indices,
            &offset_map,
        );
        let adjusted_bonds = adjust_bonds_for_sheet(
            &agg.backbone_sidechain_bonds,
            &agg.sidechain_residue_indices,
            &offset_map,
        );
        let sidechain_insts = CapsuleSidechainRenderer::generate_instances(
            &adjusted_positions,
            &agg.sidechain_bonds,
            &adjusted_bonds,
            &agg.sidechain_hydrophobicity,
            &agg.sidechain_residue_indices,
            None, // no frustum culling on background thread
            Some((colors.hydrophobic_sidechain, colors.hydrophilic_sidechain)),
        );
        let sidechain_instance_count = sidechain_insts.len() as u32;
        let sidechain_instances = bytemuck::cast_slice(&sidechain_insts).to_vec();

        // --- Ball-and-stick instances ---
        let (bns_spheres, bns_capsules, bns_picking) =
            BallAndStickRenderer::generate_all_instances(
                &agg.non_protein_entities,
                display,
                Some(colors),
            );
        let bns_sphere_count = bns_spheres.len() as u32;
        let bns_capsule_count = bns_capsules.len() as u32;
        let bns_picking_count = bns_picking.len() as u32;
        let bns_sphere_instances = bytemuck::cast_slice(&bns_spheres).to_vec();
        let bns_capsule_instances = bytemuck::cast_slice(&bns_capsules).to_vec();
        let bns_picking_capsules = bytemuck::cast_slice(&bns_picking).to_vec();

        // --- Nucleic acid mesh ---
        let (na_verts, na_inds) = NucleicAcidRenderer::generate_mesh(
            &agg.nucleic_acid_chains,
            &agg.nucleic_acid_rings,
            Some(colors.nucleic_acid),
        );
        let na_index_count = na_inds.len() as u32;
        let na_vertices = bytemuck::cast_slice(&na_verts).to_vec();
        let na_indices = bytemuck::cast_slice(&na_inds).to_vec();

        PreparedScene {
            tube_vertices,
            tube_indices,
            tube_index_count,
            ribbon_vertices,
            ribbon_indices,
            ribbon_index_count,
            sheet_offsets,
            sidechain_instances,
            sidechain_instance_count,
            bns_sphere_instances,
            bns_sphere_count,
            bns_capsule_instances,
            bns_capsule_count,
            bns_picking_capsules,
            bns_picking_count,
            na_vertices,
            na_indices,
            na_index_count,
            // Passthrough data (cloned on the background thread, not the main thread)
            backbone_chains: agg.backbone_chains.clone(),
            sidechain_positions: agg.sidechain_positions.clone(),
            sidechain_bonds: agg.sidechain_bonds.clone(),
            sidechain_hydrophobicity: agg.sidechain_hydrophobicity.clone(),
            sidechain_residue_indices: agg.sidechain_residue_indices.clone(),
            sidechain_atom_names: agg.sidechain_atom_names.clone(),
            backbone_sidechain_bonds: agg.backbone_sidechain_bonds.clone(),
            ss_types: agg.ss_types.clone(),
            all_positions: agg.all_positions.clone(),
            action,
            non_protein_entities: agg.non_protein_entities.clone(),
            nucleic_acid_chains: agg.nucleic_acid_chains.clone(),
            nucleic_acid_rings: agg.nucleic_acid_rings.clone(),
        }
    }

    /// Generate tube + ribbon + optional sidechain mesh for an animation frame.
    ///
    /// Same mesh generation as `process_full_rebuild` but skips ball-and-stick
    /// and nucleic acid (those don't change during animation).
    fn process_animation_frame(
        backbone_chains: Vec<Vec<Vec3>>,
        sidechains: Option<AnimationSidechainData>,
        view_mode: ViewMode,
        ss_types: Option<Vec<SSType>>,
    ) -> PreparedAnimationFrame {
        // --- Tube mesh ---
        let tube_filter = match view_mode {
            ViewMode::Ribbon => {
                let mut coil_only = HashSet::new();
                coil_only.insert(SSType::Coil);
                Some(coil_only)
            }
            ViewMode::Tube => None,
        };
        let (tube_verts, tube_inds) = TubeRenderer::generate_tube_mesh(
            &backbone_chains,
            &tube_filter,
            ss_types.as_deref(),
        );
        let tube_index_count = tube_inds.len() as u32;
        let tube_vertices = bytemuck::cast_slice(&tube_verts).to_vec();
        let tube_indices = bytemuck::cast_slice(&tube_inds).to_vec();

        // --- Ribbon mesh ---
        let params = RibbonParams::default();
        let (ribbon_verts, ribbon_inds, sheet_offsets) = RibbonRenderer::generate_from_ca_only(
            &backbone_chains,
            &params,
            ss_types.as_deref(),
        );
        let ribbon_index_count = ribbon_inds.len() as u32;
        let ribbon_vertices = bytemuck::cast_slice(&ribbon_verts).to_vec();
        let ribbon_indices = bytemuck::cast_slice(&ribbon_inds).to_vec();

        // --- Optional sidechain capsules ---
        let (sidechain_instances, sidechain_instance_count) = if let Some(sc) = sidechains {
            let offset_map: HashMap<u32, Vec3> = if view_mode == ViewMode::Ribbon {
                sheet_offsets.iter().copied().collect()
            } else {
                HashMap::new()
            };
            let adjusted_positions = adjust_sidechains_for_sheet(
                &sc.sidechain_positions,
                &sc.sidechain_residue_indices,
                &offset_map,
            );
            let adjusted_bonds = adjust_bonds_for_sheet(
                &sc.backbone_sidechain_bonds,
                &sc.sidechain_residue_indices,
                &offset_map,
            );
            let insts = CapsuleSidechainRenderer::generate_instances(
                &adjusted_positions,
                &sc.sidechain_bonds,
                &adjusted_bonds,
                &sc.sidechain_hydrophobicity,
                &sc.sidechain_residue_indices,
                None,
                None, // animation frames use default colors
            );
            let count = insts.len() as u32;
            (Some(bytemuck::cast_slice(&insts).to_vec()), count)
        } else {
            (None, 0)
        };

        PreparedAnimationFrame {
            tube_vertices,
            tube_indices,
            tube_index_count,
            ribbon_vertices,
            ribbon_indices,
            ribbon_index_count,
            sheet_offsets,
            sidechain_instances,
            sidechain_instance_count,
        }
    }
}

impl Drop for SceneProcessor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// --- Sheet adjustment helpers (pure CPU, duplicated from engine for thread safety) ---

fn adjust_sidechains_for_sheet(
    positions: &[Vec3],
    sidechain_residue_indices: &[u32],
    offset_map: &HashMap<u32, Vec3>,
) -> Vec<Vec3> {
    if offset_map.is_empty() {
        return positions.to_vec();
    }
    positions
        .iter()
        .enumerate()
        .map(|(i, &pos)| {
            let res_idx = sidechain_residue_indices
                .get(i)
                .copied()
                .unwrap_or(u32::MAX);
            if let Some(&offset) = offset_map.get(&res_idx) {
                pos + offset
            } else {
                pos
            }
        })
        .collect()
}

fn adjust_bonds_for_sheet(
    bonds: &[(Vec3, u32)],
    sidechain_residue_indices: &[u32],
    offset_map: &HashMap<u32, Vec3>,
) -> Vec<(Vec3, u32)> {
    if offset_map.is_empty() {
        return bonds.to_vec();
    }
    bonds
        .iter()
        .map(|(ca_pos, cb_idx)| {
            let res_idx = sidechain_residue_indices
                .get(*cb_idx as usize)
                .copied()
                .unwrap_or(u32::MAX);
            if let Some(&offset) = offset_map.get(&res_idx) {
                (*ca_pos + offset, *cb_idx)
            } else {
                (*ca_pos, *cb_idx)
            }
        })
        .collect()
}
