//! Background scene processor for non-blocking geometry generation.
//!
//! Moves all CPU-heavy mesh/instance generation off the main thread.
//! The main thread only does GPU uploads (<1ms) and render passes.
//!
//! Supports **per-group mesh caching**: when a group's `mesh_version`
//! hasn't changed between frames, its cached mesh is reused instead of
//! being regenerated. Global settings changes (view mode, display,
//! colors) clear the entire cache.

use crate::renderer::molecular::ball_and_stick::BallAndStickRenderer;
use crate::renderer::molecular::capsule_sidechain::CapsuleSidechainRenderer;
use crate::renderer::molecular::nucleic_acid::NucleicAcidRenderer;
use crate::util::options::{ColorOptions, DisplayOptions};
use crate::renderer::molecular::ribbon::{RibbonParams, RibbonRenderer};
use super::{AggregatedRenderData, GroupId, PerGroupData};
use crate::util::score_color;
use crate::renderer::molecular::tube::TubeRenderer;
use foldit_conv::coords::entity::NucleotideRing;
use foldit_conv::coords::MoleculeEntity;
use foldit_conv::secondary_structure::SSType;
use crate::animation::AnimationAction;
use glam::Vec3;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::sync::Arc;

/// Fallback color for residues without score data (neutral gray).
const FALLBACK_RESIDUE_COLOR: [f32; 3] = [0.7, 0.7, 0.7];

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
    /// Full scene rebuild with per-group data + aggregated passthrough.
    FullRebuild {
        groups: Vec<PerGroupData>,
        aggregated: Arc<AggregatedRenderData>,
        /// Per-entity animation actions. Entities in the map animate with their
        /// action; entities not in the map snap. Empty map = snap all.
        entity_actions: HashMap<GroupId, AnimationAction>,
        display: DisplayOptions,
        colors: ColorOptions,
    },
    /// Per-frame animation mesh generation (tube + ribbon + optional sidechains).
    AnimationFrame {
        backbone_chains: Vec<Vec<Vec3>>,
        sidechains: Option<AnimationSidechainData>,
        ss_types: Option<Vec<SSType>>,
        per_residue_colors: Option<Vec<[f32; 3]>>,
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
    /// Concatenated per-residue colors (derived from scores, cached for animation).
    pub per_residue_colors: Option<Vec<[f32; 3]>>,
    pub all_positions: Vec<Vec3>,
    /// Per-entity animation actions. Entities in the map animate; others snap.
    /// Empty map = snap all (no animation).
    pub entity_actions: HashMap<GroupId, AnimationAction>,
    /// Where each entity's residues land in the flat concatenated arrays:
    /// `(entity_id, global_residue_start, residue_count)`.
    pub entity_residue_ranges: Vec<(GroupId, u32, u32)>,
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

// ---------------------------------------------------------------------------
// Per-group cached mesh
// ---------------------------------------------------------------------------

/// Cached mesh data for a single group. Stored as byte buffers ready for
/// concatenation, plus typed intermediates needed for index offsetting.
struct CachedGroupMesh {
    // Tube
    tube_verts: Vec<u8>,
    tube_inds: Vec<u32>,
    // Ribbon
    ribbon_verts: Vec<u8>,
    ribbon_inds: Vec<u32>,
    sheet_offsets: Vec<(u32, Vec3)>,
    // Sidechain capsules
    sidechain_instances: Vec<u8>,
    sidechain_instance_count: u32,
    // Ball-and-stick
    bns_sphere_instances: Vec<u8>,
    bns_sphere_count: u32,
    bns_capsule_instances: Vec<u8>,
    bns_capsule_count: u32,
    bns_picking_capsules: Vec<u8>,
    bns_picking_count: u32,
    // Nucleic acid
    na_verts: Vec<u8>,
    na_inds: Vec<u32>,
    // Counts for index offsetting
    tube_vert_count: u32,
    ribbon_vert_count: u32,
    na_vert_count: u32,
    residue_count: u32,
    // Passthrough per-group (for concatenation into global passthrough)
    backbone_chains: Vec<Vec<Vec3>>,
    sidechain_positions: Vec<Vec3>,
    sidechain_bonds: Vec<(u32, u32)>,
    sidechain_hydrophobicity: Vec<bool>,
    sidechain_residue_indices: Vec<u32>,
    sidechain_atom_names: Vec<String>,
    backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    ss_override: Option<Vec<SSType>>,
    /// Per-residue colors derived from scores (cached to avoid recomputation).
    per_residue_colors: Option<Vec<[f32; 3]>>,
    non_protein_entities: Vec<MoleculeEntity>,
    nucleic_acid_chains: Vec<Vec<Vec3>>,
    nucleic_acid_rings: Vec<NucleotideRing>,
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
    pub fn try_recv_scene(&mut self) -> Option<PreparedScene> {
        self.scene_result.update();
        self.scene_result.output_buffer_mut().take()
    }

    /// Non-blocking check for completed animation frame.
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

    /// Background thread main loop with per-group mesh caching.
    fn thread_loop(
        request_rx: mpsc::Receiver<SceneRequest>,
        mut scene_input: triple_buffer::Input<Option<PreparedScene>>,
        mut anim_input: triple_buffer::Input<Option<PreparedAnimationFrame>>,
    ) {
        let mut mesh_cache: HashMap<GroupId, (u64, CachedGroupMesh)> = HashMap::new();
        let mut last_display: Option<DisplayOptions> = None;
        let mut last_colors: Option<ColorOptions> = None;

        loop {
            // Block waiting for the next request
            let request = match request_rx.recv() {
                Ok(r) => r,
                Err(_) => break,
            };

            // Drain any queued requests, keep only the latest.
            let mut latest = request;
            while let Ok(newer) = request_rx.try_recv() {
                match (&latest, &newer) {
                    (SceneRequest::FullRebuild { .. }, SceneRequest::AnimationFrame { .. }) => {}
                    _ => { latest = newer; }
                }
            }

            match latest {
                SceneRequest::Shutdown => break,
                SceneRequest::FullRebuild {
                    groups,
                    aggregated: _,
                    entity_actions,
                    display,
                    colors,
                } => {
                    // Clear cache if global settings changed
                    let settings_changed =
                        last_display.as_ref() != Some(&display)
                        || last_colors.as_ref() != Some(&colors);

                    if settings_changed {
                        mesh_cache.clear();
                        last_display = Some(display.clone());
                        last_colors = Some(colors.clone());
                    }

                    // Generate or reuse per-group meshes
                    for g in &groups {
                        let needs_regen = match mesh_cache.get(&g.id) {
                            Some((cached_version, _)) => *cached_version != g.mesh_version,
                            None => true,
                        };

                        if needs_regen {
                            let mesh = Self::generate_group_mesh(g, &display, &colors);
                            mesh_cache.insert(g.id, (g.mesh_version, mesh));
                        }
                    }

                    // Evict removed groups
                    let active_ids: HashSet<GroupId> = groups.iter().map(|g| g.id).collect();
                    mesh_cache.retain(|id, _| active_ids.contains(id));

                    // Collect references in group order
                    let group_meshes: Vec<(GroupId, &CachedGroupMesh)> = groups.iter()
                        .filter_map(|g| mesh_cache.get(&g.id).map(|(_, mesh)| (g.id, mesh)))
                        .collect();

                    // Concatenate into PreparedScene
                    let prepared = Self::concatenate_meshes(&group_meshes, entity_actions);
                    scene_input.write(Some(prepared));
                }
                SceneRequest::AnimationFrame {
                    backbone_chains,
                    sidechains,
                    ss_types,
                    per_residue_colors,
                } => {
                    let prepared = Self::process_animation_frame(
                        backbone_chains, sidechains, ss_types,
                        per_residue_colors,
                    );
                    anim_input.write(Some(prepared));
                }
            }
        }
    }

    /// Generate mesh for a single group.
    fn generate_group_mesh(
        g: &PerGroupData,
        display: &DisplayOptions,
        colors: &ColorOptions,
    ) -> CachedGroupMesh {
        // Derive per-residue colors from scores when in score coloring mode
        use crate::util::options::BackboneColorMode;
        let per_residue_colors = match display.backbone_color_mode {
            BackboneColorMode::Score => g.per_residue_scores.as_ref().map(|s| score_color::per_residue_score_colors(s)),
            BackboneColorMode::ScoreRelative => g.per_residue_scores.as_ref().map(|s| score_color::per_residue_score_colors_relative(s)),
            BackboneColorMode::SecondaryStructure | BackboneColorMode::Chain => None,
        };

        // --- Tube mesh (coils only; ribbons handle helices/sheets) ---
        let tube_filter = {
            let mut coil_only = HashSet::new();
            coil_only.insert(SSType::Coil);
            Some(coil_only)
        };
        let (tube_verts_typed, tube_inds) = TubeRenderer::generate_tube_mesh_colored(
            &g.backbone_chains,
            &tube_filter,
            g.ss_override.as_deref(),
            per_residue_colors.as_deref(),
        );
        let tube_vert_count = tube_verts_typed.len() as u32;
        let tube_verts = bytemuck::cast_slice(&tube_verts_typed).to_vec();

        // --- Ribbon mesh ---
        let params = RibbonParams::default();
        let (ribbon_verts_typed, ribbon_inds, sheet_offsets) = RibbonRenderer::generate_from_ca_only_colored(
            &g.backbone_chains,
            &params,
            g.ss_override.as_deref(),
            per_residue_colors.as_deref(),
        );
        let ribbon_vert_count = ribbon_verts_typed.len() as u32;
        let ribbon_verts = bytemuck::cast_slice(&ribbon_verts_typed).to_vec();

        // --- Sidechain capsules ---
        let sidechain_positions: Vec<Vec3> = g.sidechain_atoms.iter().map(|a| a.position).collect();
        let sidechain_hydrophobicity: Vec<bool> = g.sidechain_atoms.iter().map(|a| a.is_hydrophobic).collect();
        let sidechain_residue_indices: Vec<u32> = g.sidechain_atoms.iter().map(|a| a.residue_idx).collect();

        let offset_map: HashMap<u32, Vec3> = sheet_offsets.iter().copied().collect();
        let adjusted_positions = adjust_sidechains_for_sheet(
            &sidechain_positions,
            &sidechain_residue_indices,
            &offset_map,
        );
        let adjusted_bonds = adjust_bonds_for_sheet(
            &g.backbone_sidechain_bonds,
            &sidechain_residue_indices,
            &offset_map,
        );
        let sidechain_insts = CapsuleSidechainRenderer::generate_instances(
            &adjusted_positions,
            &g.sidechain_bonds,
            &adjusted_bonds,
            &sidechain_hydrophobicity,
            &sidechain_residue_indices,
            None,
            Some((colors.hydrophobic_sidechain, colors.hydrophilic_sidechain)),
        );
        let sidechain_instance_count = sidechain_insts.len() as u32;
        let sidechain_instances = bytemuck::cast_slice(&sidechain_insts).to_vec();

        // --- Ball-and-stick instances ---
        let (bns_spheres, bns_capsules, bns_picking) =
            BallAndStickRenderer::generate_all_instances(
                &g.non_protein_entities,
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
        let (na_verts_typed, na_inds) = NucleicAcidRenderer::generate_mesh(
            &g.nucleic_acid_chains,
            &g.nucleic_acid_rings,
            Some(colors.nucleic_acid),
        );
        let na_vert_count = na_verts_typed.len() as u32;
        let na_verts = bytemuck::cast_slice(&na_verts_typed).to_vec();

        // --- Passthrough data ---
        let sidechain_atom_names: Vec<String> = g.sidechain_atoms.iter().map(|a| a.atom_name.clone()).collect();

        CachedGroupMesh {
            tube_verts,
            tube_inds,
            ribbon_verts,
            ribbon_inds,
            sheet_offsets,
            sidechain_instances,
            sidechain_instance_count,
            bns_sphere_instances,
            bns_sphere_count,
            bns_capsule_instances,
            bns_capsule_count,
            bns_picking_capsules,
            bns_picking_count,
            na_verts,
            na_inds,
            tube_vert_count,
            ribbon_vert_count,
            na_vert_count,
            residue_count: g.residue_count,
            backbone_chains: g.backbone_chains.clone(),
            sidechain_positions,
            sidechain_bonds: g.sidechain_bonds.clone(),
            sidechain_hydrophobicity,
            sidechain_residue_indices,
            sidechain_atom_names,
            backbone_sidechain_bonds: g.backbone_sidechain_bonds.clone(),
            ss_override: g.ss_override.clone(),
            per_residue_colors,
            non_protein_entities: g.non_protein_entities.clone(),
            nucleic_acid_chains: g.nucleic_acid_chains.clone(),
            nucleic_acid_rings: g.nucleic_acid_rings.clone(),
        }
    }

    /// Concatenate per-entity cached meshes into a single PreparedScene.
    fn concatenate_meshes(
        group_meshes: &[(GroupId, &CachedGroupMesh)],
        entity_actions: HashMap<GroupId, AnimationAction>,
    ) -> PreparedScene {
        // --- Tube ---
        let mut all_tube_verts: Vec<u8> = Vec::new();
        let mut all_tube_inds: Vec<u32> = Vec::new();
        let mut tube_vert_offset: u32 = 0;

        // --- Ribbon ---
        let mut all_ribbon_verts: Vec<u8> = Vec::new();
        let mut all_ribbon_inds: Vec<u32> = Vec::new();
        let mut ribbon_vert_offset: u32 = 0;
        let mut all_sheet_offsets: Vec<(u32, Vec3)> = Vec::new();

        // --- Sidechain ---
        let mut all_sidechain: Vec<u8> = Vec::new();
        let mut total_sidechain_count: u32 = 0;

        // --- BNS ---
        let mut all_bns_spheres: Vec<u8> = Vec::new();
        let mut total_bns_sphere_count: u32 = 0;
        let mut all_bns_capsules: Vec<u8> = Vec::new();
        let mut total_bns_capsule_count: u32 = 0;
        let mut all_bns_picking: Vec<u8> = Vec::new();
        let mut total_bns_picking_count: u32 = 0;

        // --- Nucleic acid ---
        let mut all_na_verts: Vec<u8> = Vec::new();
        let mut all_na_inds: Vec<u32> = Vec::new();
        let mut na_vert_offset: u32 = 0;

        // --- Passthrough ---
        let mut all_backbone_chains: Vec<Vec<Vec3>> = Vec::new();
        let mut all_sidechain_positions: Vec<Vec3> = Vec::new();
        let mut all_sidechain_bonds: Vec<(u32, u32)> = Vec::new();
        let mut all_sidechain_hydrophobicity: Vec<bool> = Vec::new();
        let mut all_sidechain_residue_indices: Vec<u32> = Vec::new();
        let mut all_sidechain_atom_names: Vec<String> = Vec::new();
        let mut all_backbone_sidechain_bonds: Vec<(Vec3, u32)> = Vec::new();
        let mut all_non_protein: Vec<MoleculeEntity> = Vec::new();
        let mut all_na_chains: Vec<Vec<Vec3>> = Vec::new();
        let mut all_na_rings: Vec<NucleotideRing> = Vec::new();
        let mut all_positions: Vec<Vec3> = Vec::new();

        // SS types: built from per-group overrides
        let mut has_any_ss = false;
        let mut ss_parts: Vec<(u32, Option<Vec<SSType>>, u32)> = Vec::new();
        let mut global_residue_offset: u32 = 0;

        // Per-residue colors: concatenated from cached group colors
        let mut has_any_colors = false;
        let mut all_per_residue_colors: Vec<[f32; 3]> = Vec::new();

        // Track where each entity's residues land in the flat arrays
        let mut entity_residue_ranges: Vec<(GroupId, u32, u32)> = Vec::new();

        for (group_id, mesh) in group_meshes {
            let sc_atom_offset = all_sidechain_positions.len() as u32;

            // Tube: offset indices
            all_tube_verts.extend_from_slice(&mesh.tube_verts);
            for &idx in &mesh.tube_inds {
                all_tube_inds.push(idx + tube_vert_offset);
            }
            tube_vert_offset += mesh.tube_vert_count;

            // Ribbon: offset indices
            all_ribbon_verts.extend_from_slice(&mesh.ribbon_verts);
            for &idx in &mesh.ribbon_inds {
                all_ribbon_inds.push(idx + ribbon_vert_offset);
            }
            // Sheet offsets: offset residue indices
            for &(res_idx, offset) in &mesh.sheet_offsets {
                all_sheet_offsets.push((res_idx + global_residue_offset, offset));
            }
            ribbon_vert_offset += mesh.ribbon_vert_count;

            // Sidechain: concatenate directly (instances are self-contained)
            all_sidechain.extend_from_slice(&mesh.sidechain_instances);
            total_sidechain_count += mesh.sidechain_instance_count;

            // BNS: concatenate directly
            all_bns_spheres.extend_from_slice(&mesh.bns_sphere_instances);
            total_bns_sphere_count += mesh.bns_sphere_count;
            all_bns_capsules.extend_from_slice(&mesh.bns_capsule_instances);
            total_bns_capsule_count += mesh.bns_capsule_count;
            all_bns_picking.extend_from_slice(&mesh.bns_picking_capsules);
            total_bns_picking_count += mesh.bns_picking_count;

            // NA: offset indices
            all_na_verts.extend_from_slice(&mesh.na_verts);
            for &idx in &mesh.na_inds {
                all_na_inds.push(idx + na_vert_offset);
            }
            na_vert_offset += mesh.na_vert_count;

            // Passthrough
            for chain in &mesh.backbone_chains {
                all_backbone_chains.push(chain.clone());
                all_positions.extend(chain);
            }
            all_sidechain_positions.extend(&mesh.sidechain_positions);
            for &(a, b) in &mesh.sidechain_bonds {
                all_sidechain_bonds.push((a + sc_atom_offset, b + sc_atom_offset));
            }
            all_sidechain_hydrophobicity.extend(&mesh.sidechain_hydrophobicity);
            for &ri in &mesh.sidechain_residue_indices {
                all_sidechain_residue_indices.push(ri + global_residue_offset);
            }
            all_sidechain_atom_names.extend(mesh.sidechain_atom_names.iter().cloned());
            for &(ca_pos, cb_idx) in &mesh.backbone_sidechain_bonds {
                all_backbone_sidechain_bonds.push((ca_pos, cb_idx + sc_atom_offset));
            }
            all_positions.extend(&mesh.sidechain_positions);
            all_non_protein.extend(mesh.non_protein_entities.iter().cloned());
            for chain in &mesh.nucleic_acid_chains {
                all_na_chains.push(chain.clone());
                all_positions.extend(chain);
            }
            all_na_rings.extend(mesh.nucleic_acid_rings.iter().cloned());

            // SS override tracking
            if mesh.ss_override.is_some() {
                has_any_ss = true;
            }
            ss_parts.push((global_residue_offset, mesh.ss_override.clone(), mesh.residue_count));

            // Per-residue color tracking
            if let Some(ref colors) = mesh.per_residue_colors {
                has_any_colors = true;
                all_per_residue_colors.extend_from_slice(colors);
            } else {
                // Pad with default so indices stay aligned
                for _ in 0..mesh.residue_count {
                    all_per_residue_colors.push(FALLBACK_RESIDUE_COLOR);
                }
            }

            // Track entity residue range
            entity_residue_ranges.push((*group_id, global_residue_offset, mesh.residue_count));

            global_residue_offset += mesh.residue_count;
        }

        // Build flat ss_types
        let ss_types = if has_any_ss {
            let total = global_residue_offset as usize;
            let mut ss = vec![SSType::Coil; total];
            for (offset, ss_override, count) in &ss_parts {
                if let Some(overrides) = ss_override {
                    let start = *offset as usize;
                    let end = (start + *count as usize).min(total);
                    for (i, &s) in overrides.iter().enumerate() {
                        if start + i < end {
                            ss[start + i] = s;
                        }
                    }
                }
            }
            Some(ss)
        } else {
            None
        };

        PreparedScene {
            tube_vertices: all_tube_verts,
            tube_indices: bytemuck::cast_slice(&all_tube_inds).to_vec(),
            tube_index_count: all_tube_inds.len() as u32,
            ribbon_vertices: all_ribbon_verts,
            ribbon_indices: bytemuck::cast_slice(&all_ribbon_inds).to_vec(),
            ribbon_index_count: all_ribbon_inds.len() as u32,
            sheet_offsets: all_sheet_offsets,
            sidechain_instances: all_sidechain,
            sidechain_instance_count: total_sidechain_count,
            bns_sphere_instances: all_bns_spheres,
            bns_sphere_count: total_bns_sphere_count,
            bns_capsule_instances: all_bns_capsules,
            bns_capsule_count: total_bns_capsule_count,
            bns_picking_capsules: all_bns_picking,
            bns_picking_count: total_bns_picking_count,
            na_vertices: all_na_verts,
            na_indices: bytemuck::cast_slice(&all_na_inds).to_vec(),
            na_index_count: all_na_inds.len() as u32,
            backbone_chains: all_backbone_chains,
            sidechain_positions: all_sidechain_positions,
            sidechain_bonds: all_sidechain_bonds,
            sidechain_hydrophobicity: all_sidechain_hydrophobicity,
            sidechain_residue_indices: all_sidechain_residue_indices,
            sidechain_atom_names: all_sidechain_atom_names,
            backbone_sidechain_bonds: all_backbone_sidechain_bonds,
            ss_types,
            per_residue_colors: if has_any_colors { Some(all_per_residue_colors) } else { None },
            all_positions,
            entity_actions,
            entity_residue_ranges,
            non_protein_entities: all_non_protein,
            nucleic_acid_chains: all_na_chains,
            nucleic_acid_rings: all_na_rings,
        }
    }

    /// Generate tube + ribbon + optional sidechain mesh for an animation frame.
    fn process_animation_frame(
        backbone_chains: Vec<Vec<Vec3>>,
        sidechains: Option<AnimationSidechainData>,
        ss_types: Option<Vec<SSType>>,
        per_residue_colors: Option<Vec<[f32; 3]>>,
    ) -> PreparedAnimationFrame {
        // --- Tube mesh (coils only; ribbons handle helices/sheets) ---
        let tube_filter = {
            let mut coil_only = HashSet::new();
            coil_only.insert(SSType::Coil);
            Some(coil_only)
        };
        let (tube_verts, tube_inds) = TubeRenderer::generate_tube_mesh_colored(
            &backbone_chains,
            &tube_filter,
            ss_types.as_deref(),
            per_residue_colors.as_deref(),
        );
        let tube_index_count = tube_inds.len() as u32;
        let tube_vertices = bytemuck::cast_slice(&tube_verts).to_vec();
        let tube_indices = bytemuck::cast_slice(&tube_inds).to_vec();

        // --- Ribbon mesh ---
        let params = RibbonParams::default();
        let (ribbon_verts, ribbon_inds, sheet_offsets) = RibbonRenderer::generate_from_ca_only_colored(
            &backbone_chains,
            &params,
            ss_types.as_deref(),
            per_residue_colors.as_deref(),
        );
        let ribbon_index_count = ribbon_inds.len() as u32;
        let ribbon_vertices = bytemuck::cast_slice(&ribbon_verts).to_vec();
        let ribbon_indices = bytemuck::cast_slice(&ribbon_inds).to_vec();

        // --- Optional sidechain capsules ---
        let (sidechain_instances, sidechain_instance_count) = if let Some(sc) = sidechains {
            let offset_map: HashMap<u32, Vec3> = sheet_offsets.iter().copied().collect();
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
                None,
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

use crate::util::sheet_adjust::{adjust_sidechains_for_sheet, adjust_bonds_for_sheet};
