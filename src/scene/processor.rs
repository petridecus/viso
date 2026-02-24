//! Background scene processor for non-blocking geometry generation.
//!
//! Moves all CPU-heavy mesh/instance generation off the main thread.
//! The main thread only does GPU uploads (<1ms) and render passes.
//!
//! Supports **per-entity mesh caching**: when an entity's `mesh_version`
//! hasn't changed between frames, its cached mesh is reused instead of
//! being regenerated. Global settings changes (view mode, display,
//! colors) clear the entire cache.

use std::{
    collections::{HashMap, HashSet},
    sync::mpsc,
};

use foldit_conv::{
    coords::{entity::NucleotideRing, MoleculeEntity},
    secondary_structure::SSType,
};
use glam::Vec3;

use super::PerEntityData;
use crate::{
    animation::Transition,
    options::{ColorOptions, DisplayOptions, GeometryOptions},
    renderer::molecular::{
        backbone::{BackboneRenderer, ChainRange},
        ball_and_stick::BallAndStickRenderer,
        capsule_sidechain::CapsuleSidechainRenderer,
        nucleic_acid::NucleicAcidRenderer,
    },
    util::score_color,
};

/// Fallback color for residues without score data (neutral gray).
const FALLBACK_RESIDUE_COLOR: [f32; 3] = [0.7, 0.7, 0.7];

/// Sidechain data bundled for animation frame processing.
pub struct AnimationSidechainData {
    /// Sidechain atom positions in world space.
    pub sidechain_positions: Vec<Vec3>,
    /// Intra-residue bonds as (atom_idx, atom_idx) pairs.
    pub sidechain_bonds: Vec<(u32, u32)>,
    /// Backbone-to-sidechain bonds as (CA position, atom idx).
    pub backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    /// Hydrophobicity flag per sidechain atom.
    pub sidechain_hydrophobicity: Vec<bool>,
    /// Residue index per sidechain atom.
    pub sidechain_residue_indices: Vec<u32>,
}

/// Request sent from main thread to scene processor.
pub enum SceneRequest {
    /// Full scene rebuild with per-entity data.
    FullRebuild {
        /// Per-entity data for mesh generation.
        entities: Vec<PerEntityData>,
        /// Per-entity transitions. Entities in the map animate with
        /// their transition; entities not in the map snap. Empty map =
        /// snap all.
        entity_transitions: HashMap<u32, Transition>,
        /// Current display options for mesh generation.
        display: DisplayOptions,
        /// Current color options for mesh generation.
        colors: ColorOptions,
        /// Current geometry options for mesh generation.
        geometry: GeometryOptions,
    },
    /// Per-frame animation mesh generation (backbone + optional sidechains).
    AnimationFrame {
        /// Interpolated backbone atom chains.
        backbone_chains: Vec<Vec<Vec3>>,
        /// Nucleic acid P-atom chains for backbone rendering.
        na_chains: Vec<Vec<Vec3>>,
        /// Optional interpolated sidechain data.
        sidechains: Option<AnimationSidechainData>,
        /// Secondary structure types for the current frame.
        ss_types: Option<Vec<SSType>>,
        /// Per-residue colors for the current frame.
        per_residue_colors: Option<Vec<[f32; 3]>>,
        /// Geometry options for mesh generation.
        geometry: GeometryOptions,
        /// Per-chain (spr, csv) overrides for LOD. When `Some`, each
        /// chain uses its own detail level instead of the global geo
        /// settings.
        per_chain_lod: Option<Vec<(usize, usize)>>,
    },
    /// Shut down the background thread.
    Shutdown,
}

/// All pre-computed CPU data, ready for GPU-only upload on the main thread.
#[derive(Clone)]
pub struct PreparedScene {
    /// Backbone mesh vertex bytes (shared by tube and ribbon passes).
    pub backbone_vertices: Vec<u8>,
    /// Backbone tube index bytes (back-face culled pass).
    pub backbone_tube_indices: Vec<u8>,
    /// Number of backbone tube indices.
    pub backbone_tube_index_count: u32,
    /// Backbone ribbon index bytes (no-cull pass).
    pub backbone_ribbon_indices: Vec<u8>,
    /// Number of backbone ribbon indices.
    pub backbone_ribbon_index_count: u32,
    /// Per-residue sheet normal offsets for sidechain adjustment.
    pub sheet_offsets: Vec<(u32, Vec3)>,
    /// Per-chain index ranges and bounding spheres for frustum culling.
    pub backbone_chain_ranges: Vec<ChainRange>,

    /// Sidechain capsule instance bytes.
    pub sidechain_instances: Vec<u8>,
    /// Number of sidechain capsule instances.
    pub sidechain_instance_count: u32,

    /// Ball-and-stick sphere instance bytes.
    pub bns_sphere_instances: Vec<u8>,
    /// Number of ball-and-stick spheres.
    pub bns_sphere_count: u32,
    /// Ball-and-stick capsule instance bytes.
    pub bns_capsule_instances: Vec<u8>,
    /// Number of ball-and-stick capsules.
    pub bns_capsule_count: u32,
    /// Ball-and-stick picking capsule bytes.
    pub bns_picking_capsules: Vec<u8>,
    /// Number of ball-and-stick picking capsules.
    pub bns_picking_count: u32,

    /// Nucleic acid mesh vertex bytes.
    pub na_vertices: Vec<u8>,
    /// Nucleic acid mesh index bytes.
    pub na_indices: Vec<u8>,
    /// Number of nucleic acid indices.
    pub na_index_count: u32,

    /// Backbone chains for animation setup.
    pub backbone_chains: Vec<Vec<Vec3>>,
    /// Nucleic acid P-atom chains.
    pub na_chains: Vec<Vec<Vec3>>,
    /// Sidechain atom positions.
    pub sidechain_positions: Vec<Vec3>,
    /// Sidechain intra-residue bonds.
    pub sidechain_bonds: Vec<(u32, u32)>,
    /// Hydrophobicity per sidechain atom.
    pub sidechain_hydrophobicity: Vec<bool>,
    /// Residue index per sidechain atom.
    pub sidechain_residue_indices: Vec<u32>,
    /// PDB atom names per sidechain atom.
    pub sidechain_atom_names: Vec<String>,
    /// Backbone-to-sidechain bonds.
    pub backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    /// Flat secondary structure types.
    pub ss_types: Option<Vec<SSType>>,
    /// Concatenated per-residue colors (derived from scores, cached for
    /// animation).
    pub per_residue_colors: Option<Vec<[f32; 3]>>,
    /// All atom positions for camera fitting.
    pub all_positions: Vec<Vec3>,
    /// Per-entity transitions. Entities in the map animate; others snap.
    /// Empty map = snap all (no animation).
    pub entity_transitions: HashMap<u32, Transition>,
    /// Where each entity's residues land in the flat concatenated arrays:
    /// `(entity_id, global_residue_start, residue_count)`.
    pub entity_residue_ranges: Vec<(u32, u32, u32)>,
    /// Non-protein entities for ball-and-stick rendering.
    pub non_protein_entities: Vec<MoleculeEntity>,
    /// Base ring geometry from DNA/RNA entities.
    pub nucleic_acid_rings: Vec<NucleotideRing>,
}

/// Pre-computed animation frame data, ready for GPU upload.
#[derive(Clone)]
pub struct PreparedAnimationFrame {
    /// Backbone mesh vertex bytes (shared by tube and ribbon passes).
    pub backbone_vertices: Vec<u8>,
    /// Backbone tube index bytes.
    pub backbone_tube_indices: Vec<u8>,
    /// Number of backbone tube indices.
    pub backbone_tube_index_count: u32,
    /// Backbone ribbon index bytes.
    pub backbone_ribbon_indices: Vec<u8>,
    /// Number of backbone ribbon indices.
    pub backbone_ribbon_index_count: u32,
    /// Per-residue sheet normal offsets.
    pub sheet_offsets: Vec<(u32, Vec3)>,
    /// Per-chain index ranges and bounding spheres for frustum culling.
    pub backbone_chain_ranges: Vec<ChainRange>,
    /// Optional sidechain capsule instance bytes.
    pub sidechain_instances: Option<Vec<u8>>,
    /// Number of sidechain capsule instances.
    pub sidechain_instance_count: u32,
}

// ---------------------------------------------------------------------------
// Per-group cached mesh
// ---------------------------------------------------------------------------

/// Cached mesh data for a single group. Stored as byte buffers ready for
/// concatenation, plus typed intermediates needed for index offsetting.
struct CachedEntityMesh {
    // Backbone (unified vertex buffer, partitioned index buffers)
    backbone_verts: Vec<u8>,
    backbone_tube_inds: Vec<u32>,
    backbone_ribbon_inds: Vec<u32>,
    backbone_vert_count: u32,
    sheet_offsets: Vec<(u32, Vec3)>,
    backbone_chain_ranges: Vec<ChainRange>,
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
    // Nucleic acid (ring + stem only; backbone handled by BackboneRenderer)
    na_verts: Vec<u8>,
    na_inds: Vec<u32>,
    na_vert_count: u32,
    residue_count: u32,
    // Passthrough per-group (for concatenation into global passthrough)
    backbone_chains: Vec<Vec<Vec3>>,
    nucleic_acid_chains: Vec<Vec<Vec3>>,
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
    nucleic_acid_rings: Vec<NucleotideRing>,
}

/// Background thread that generates CPU-side geometry from scene data.
pub struct SceneProcessor {
    request_tx: mpsc::Sender<SceneRequest>,
    scene_result: triple_buffer::Output<Option<PreparedScene>>,
    anim_result: triple_buffer::Output<Option<PreparedAnimationFrame>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SceneProcessor {
    /// Spawn the background scene processing thread.
    pub fn new() -> Result<Self, std::io::Error> {
        let (request_tx, request_rx) = mpsc::channel::<SceneRequest>();
        let (scene_input, scene_output) = triple_buffer::triple_buffer(&None);
        let (anim_input, anim_output) = triple_buffer::triple_buffer(&None);

        let thread = std::thread::Builder::new()
            .name("scene-processor".into())
            .spawn(move || {
                Self::thread_loop(request_rx, scene_input, anim_input);
            })?;

        Ok(Self {
            request_tx,
            scene_result: scene_output,
            anim_result: anim_output,
            thread: Some(thread),
        })
    }

    /// Submit a scene request (non-blocking send).
    pub fn submit(&self, request: SceneRequest) {
        let _ = self.request_tx.send(request);
    }

    /// Non-blocking check for completed full scene rebuild.
    pub fn try_recv_scene(&mut self) -> Option<PreparedScene> {
        let _ = self.scene_result.update();
        self.scene_result.output_buffer_mut().take()
    }

    /// Non-blocking check for completed animation frame.
    pub fn try_recv_animation(&mut self) -> Option<PreparedAnimationFrame> {
        let _ = self.anim_result.update();
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
        let mut mesh_cache: HashMap<u32, (u64, CachedEntityMesh)> =
            HashMap::new();
        let mut last_display: Option<DisplayOptions> = None;
        let mut last_colors: Option<ColorOptions> = None;
        let mut last_geometry: Option<GeometryOptions> = None;

        while let Ok(request) = request_rx.recv() {
            // Drain any queued requests, keep only the latest.
            let mut latest = request;
            while let Ok(newer) = request_rx.try_recv() {
                match (&latest, &newer) {
                    (
                        SceneRequest::FullRebuild { .. },
                        SceneRequest::AnimationFrame { .. },
                    ) => {}
                    _ => {
                        latest = newer;
                    }
                }
            }

            match latest {
                SceneRequest::Shutdown => break,
                SceneRequest::FullRebuild {
                    entities,
                    entity_transitions,
                    display,
                    colors,
                    geometry,
                } => {
                    // Clamp geometry detail so the concatenated vertex
                    // buffer stays under the wgpu 256 MB max.
                    let total_residues: usize = entities
                        .iter()
                        .map(|e| {
                            e.backbone_chains
                                .iter()
                                .map(|c| c.len() / 3)
                                .sum::<usize>()
                                + e.nucleic_acid_chains
                                    .iter()
                                    .map(|c| c.len())
                                    .sum::<usize>()
                        })
                        .sum();
                    let geometry =
                        geometry.clamped_for_residues(total_residues);

                    // Clear cache if global settings changed
                    let settings_changed = last_display.as_ref()
                        != Some(&display)
                        || last_colors.as_ref() != Some(&colors)
                        || last_geometry.as_ref() != Some(&geometry);

                    if settings_changed {
                        mesh_cache.clear();
                        last_display = Some(display.clone());
                        last_colors = Some(colors.clone());
                        last_geometry = Some(geometry.clone());
                    }

                    // Generate or reuse per-entity meshes
                    for e in &entities {
                        let needs_regen = match mesh_cache.get(&e.id) {
                            Some((cached_version, _)) => {
                                *cached_version != e.mesh_version
                            }
                            None => true,
                        };

                        if needs_regen {
                            let mesh = Self::generate_entity_mesh(
                                e, &display, &colors, &geometry,
                            );
                            let _ =
                                mesh_cache.insert(e.id, (e.mesh_version, mesh));
                        }
                    }

                    // Evict removed entities
                    let active_ids: HashSet<u32> =
                        entities.iter().map(|e| e.id).collect();
                    mesh_cache.retain(|id, _| active_ids.contains(id));

                    // Collect references in entity order
                    let entity_meshes: Vec<(u32, &CachedEntityMesh)> = entities
                        .iter()
                        .filter_map(|e| {
                            mesh_cache.get(&e.id).map(|(_, mesh)| (e.id, mesh))
                        })
                        .collect();

                    // Concatenate into PreparedScene
                    let prepared = Self::concatenate_meshes(
                        &entity_meshes,
                        entity_transitions,
                    );
                    scene_input.write(Some(prepared));
                }
                SceneRequest::AnimationFrame {
                    backbone_chains,
                    na_chains,
                    sidechains,
                    ss_types,
                    per_residue_colors,
                    geometry,
                    per_chain_lod,
                } => {
                    let prepared = Self::process_animation_frame(
                        backbone_chains,
                        na_chains,
                        sidechains,
                        ss_types,
                        per_residue_colors,
                        &geometry,
                        per_chain_lod,
                    );
                    anim_input.write(Some(prepared));
                }
            }
        }
    }

    /// Generate mesh for a single entity.
    fn generate_entity_mesh(
        g: &PerEntityData,
        display: &DisplayOptions,
        colors: &ColorOptions,
        geometry: &GeometryOptions,
    ) -> CachedEntityMesh {
        // Derive per-residue colors from scores when in score coloring mode
        use crate::options::BackboneColorMode;
        let per_residue_colors = match display.backbone_color_mode {
            BackboneColorMode::Score => g
                .per_residue_scores
                .as_ref()
                .map(|s| score_color::per_residue_score_colors(s)),
            BackboneColorMode::ScoreRelative => g
                .per_residue_scores
                .as_ref()
                .map(|s| score_color::per_residue_score_colors_relative(s)),
            BackboneColorMode::SecondaryStructure
            | BackboneColorMode::Chain => None,
        };

        // --- Backbone mesh (protein + nucleic acid, unified) ---
        let (
            backbone_verts_typed,
            backbone_tube_inds,
            backbone_ribbon_inds,
            sheet_offsets,
            backbone_chain_ranges,
        ) = BackboneRenderer::generate_mesh_colored(
            &g.backbone_chains,
            &g.nucleic_acid_chains,
            g.ss_override.as_deref(),
            per_residue_colors.as_deref(),
            geometry,
            None,
        );
        let backbone_vert_count = backbone_verts_typed.len() as u32;
        let backbone_verts =
            bytemuck::cast_slice(&backbone_verts_typed).to_vec();

        // --- Sidechain capsules ---
        let sidechain_positions: Vec<Vec3> =
            g.sidechain_atoms.iter().map(|a| a.position).collect();
        let sidechain_hydrophobicity: Vec<bool> =
            g.sidechain_atoms.iter().map(|a| a.is_hydrophobic).collect();
        let sidechain_residue_indices: Vec<u32> =
            g.sidechain_atoms.iter().map(|a| a.residue_idx).collect();

        let offset_map: HashMap<u32, Vec3> =
            sheet_offsets.iter().copied().collect();
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
        let sidechain_instances =
            bytemuck::cast_slice(&sidechain_insts).to_vec();

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
        let bns_capsule_instances =
            bytemuck::cast_slice(&bns_capsules).to_vec();
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
        let sidechain_atom_names: Vec<String> = g
            .sidechain_atoms
            .iter()
            .map(|a| a.atom_name.clone())
            .collect();

        CachedEntityMesh {
            backbone_verts,
            backbone_tube_inds,
            backbone_ribbon_inds,
            backbone_vert_count,
            sheet_offsets,
            backbone_chain_ranges,
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
            na_vert_count,
            residue_count: g.residue_count,
            backbone_chains: g.backbone_chains.clone(),
            nucleic_acid_chains: g.nucleic_acid_chains.clone(),
            sidechain_positions,
            sidechain_bonds: g.sidechain_bonds.clone(),
            sidechain_hydrophobicity,
            sidechain_residue_indices,
            sidechain_atom_names,
            backbone_sidechain_bonds: g.backbone_sidechain_bonds.clone(),
            ss_override: g.ss_override.clone(),
            per_residue_colors,
            non_protein_entities: g.non_protein_entities.clone(),
            nucleic_acid_rings: g.nucleic_acid_rings.clone(),
        }
    }

    /// Offset the `residue_idx` field embedded in raw vertex bytes.
    ///
    /// `BackboneVertex` and `NaVertex` share the same 52-byte
    /// layout with a `u32 residue_idx` at byte offset 36. When concatenating
    /// vertices from multiple entities, each entity's local residue indices
    /// must be shifted by the global offset so the GPU's per-residue color
    /// buffer is indexed correctly.
    fn offset_vertex_residue_idx(dst: &mut Vec<u8>, src: &[u8], offset: u32) {
        const VERTEX_SIZE: usize = 52;
        const RESIDUE_IDX_OFFSET: usize = 36;

        if offset == 0 {
            dst.extend_from_slice(src);
            return;
        }

        let start = dst.len();
        dst.extend_from_slice(src);

        // Patch each vertex's residue_idx in-place
        let mut pos = start + RESIDUE_IDX_OFFSET;
        while pos + 4 <= dst.len() {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&dst[pos..pos + 4]);
            let patched = u32::from_ne_bytes(bytes) + offset;
            dst[pos..pos + 4].copy_from_slice(&patched.to_ne_bytes());
            pos += VERTEX_SIZE;
        }
    }

    /// Concatenate per-entity cached meshes into a single PreparedScene.
    fn concatenate_meshes(
        entity_meshes: &[(u32, &CachedEntityMesh)],
        entity_transitions: HashMap<u32, Transition>,
    ) -> PreparedScene {
        // --- Backbone (unified vertex buffer, partitioned index buffers) ---
        let mut all_backbone_verts: Vec<u8> = Vec::new();
        let mut all_backbone_tube_inds: Vec<u32> = Vec::new();
        let mut all_backbone_ribbon_inds: Vec<u32> = Vec::new();
        let mut backbone_vert_offset: u32 = 0;
        let mut all_sheet_offsets: Vec<(u32, Vec3)> = Vec::new();
        let mut all_chain_ranges: Vec<ChainRange> = Vec::new();

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
        let mut entity_residue_ranges: Vec<(u32, u32, u32)> = Vec::new();

        for (entity_id, mesh) in entity_meshes {
            let sc_atom_offset = all_sidechain_positions.len() as u32;

            // Backbone: offset vertex residue_idx and indices
            Self::offset_vertex_residue_idx(
                &mut all_backbone_verts,
                &mesh.backbone_verts,
                global_residue_offset,
            );
            for &idx in &mesh.backbone_tube_inds {
                all_backbone_tube_inds.push(idx + backbone_vert_offset);
            }
            for &idx in &mesh.backbone_ribbon_inds {
                all_backbone_ribbon_inds.push(idx + backbone_vert_offset);
            }
            // Sheet offsets: offset residue indices
            for &(res_idx, offset) in &mesh.sheet_offsets {
                all_sheet_offsets
                    .push((res_idx + global_residue_offset, offset));
            }
            // Chain ranges: offset index ranges into the global buffers
            let tube_idx_offset = all_backbone_tube_inds.len() as u32
                - mesh.backbone_tube_inds.len() as u32;
            let ribbon_idx_offset = all_backbone_ribbon_inds.len() as u32
                - mesh.backbone_ribbon_inds.len() as u32;
            for r in &mesh.backbone_chain_ranges {
                all_chain_ranges.push(ChainRange {
                    tube_index_start: r.tube_index_start + tube_idx_offset,
                    tube_index_end: r.tube_index_end + tube_idx_offset,
                    ribbon_index_start: r.ribbon_index_start
                        + ribbon_idx_offset,
                    ribbon_index_end: r.ribbon_index_end + ribbon_idx_offset,
                    bounding_center: r.bounding_center,
                    bounding_radius: r.bounding_radius,
                });
            }
            backbone_vert_offset += mesh.backbone_vert_count;

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

            // NA: offset vertex indices and embedded residue_idx
            Self::offset_vertex_residue_idx(
                &mut all_na_verts,
                &mesh.na_verts,
                global_residue_offset,
            );
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
                all_sidechain_bonds
                    .push((a + sc_atom_offset, b + sc_atom_offset));
            }
            all_sidechain_hydrophobicity.extend(&mesh.sidechain_hydrophobicity);
            for &ri in &mesh.sidechain_residue_indices {
                all_sidechain_residue_indices.push(ri + global_residue_offset);
            }
            all_sidechain_atom_names
                .extend(mesh.sidechain_atom_names.iter().cloned());
            for &(ca_pos, cb_idx) in &mesh.backbone_sidechain_bonds {
                all_backbone_sidechain_bonds
                    .push((ca_pos, cb_idx + sc_atom_offset));
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
            ss_parts.push((
                global_residue_offset,
                mesh.ss_override.clone(),
                mesh.residue_count,
            ));

            // Per-residue color tracking
            if let Some(ref colors) = mesh.per_residue_colors {
                has_any_colors = true;
                all_per_residue_colors.extend_from_slice(colors);
            } else {
                // Pad with default so indices stay aligned
                all_per_residue_colors.extend(std::iter::repeat_n(
                    FALLBACK_RESIDUE_COLOR,
                    mesh.residue_count as usize,
                ));
            }

            // Track entity residue range
            entity_residue_ranges.push((
                *entity_id,
                global_residue_offset,
                mesh.residue_count,
            ));

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
            backbone_vertices: all_backbone_verts,
            backbone_tube_indices: bytemuck::cast_slice(
                &all_backbone_tube_inds,
            )
            .to_vec(),
            backbone_tube_index_count: all_backbone_tube_inds.len() as u32,
            backbone_ribbon_indices: bytemuck::cast_slice(
                &all_backbone_ribbon_inds,
            )
            .to_vec(),
            backbone_ribbon_index_count: all_backbone_ribbon_inds.len() as u32,
            sheet_offsets: all_sheet_offsets,
            backbone_chain_ranges: all_chain_ranges,
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
            na_chains: all_na_chains,
            sidechain_positions: all_sidechain_positions,
            sidechain_bonds: all_sidechain_bonds,
            sidechain_hydrophobicity: all_sidechain_hydrophobicity,
            sidechain_residue_indices: all_sidechain_residue_indices,
            sidechain_atom_names: all_sidechain_atom_names,
            backbone_sidechain_bonds: all_backbone_sidechain_bonds,
            ss_types,
            per_residue_colors: if has_any_colors {
                Some(all_per_residue_colors)
            } else {
                None
            },
            all_positions,
            entity_transitions,
            entity_residue_ranges,
            non_protein_entities: all_non_protein,
            nucleic_acid_rings: all_na_rings,
        }
    }

    /// Generate backbone + optional sidechain mesh for an animation frame.
    fn process_animation_frame(
        backbone_chains: Vec<Vec<Vec3>>,
        na_chains: Vec<Vec<Vec3>>,
        sidechains: Option<AnimationSidechainData>,
        ss_types: Option<Vec<SSType>>,
        per_residue_colors: Option<Vec<[f32; 3]>>,
        geometry: &GeometryOptions,
        per_chain_lod: Option<Vec<(usize, usize)>>,
    ) -> PreparedAnimationFrame {
        // --- Backbone mesh (protein + nucleic acid, unified) ---
        let total_residues: usize =
            backbone_chains.iter().map(|c| c.len() / 3).sum::<usize>()
                + na_chains.iter().map(|c| c.len()).sum::<usize>();
        let safe_geo = geometry.clamped_for_residues(total_residues);
        let (verts, tube_inds, ribbon_inds, sheet_offsets, chain_ranges) =
            BackboneRenderer::generate_mesh_colored(
                &backbone_chains,
                &na_chains,
                ss_types.as_deref(),
                per_residue_colors.as_deref(),
                &safe_geo,
                per_chain_lod.as_deref(),
            );
        let backbone_tube_index_count = tube_inds.len() as u32;
        let backbone_ribbon_index_count = ribbon_inds.len() as u32;
        let backbone_vertices = bytemuck::cast_slice(&verts).to_vec();
        let backbone_tube_indices = bytemuck::cast_slice(&tube_inds).to_vec();
        let backbone_ribbon_indices =
            bytemuck::cast_slice(&ribbon_inds).to_vec();

        // --- Optional sidechain capsules ---
        let (sidechain_instances, sidechain_instance_count) =
            if let Some(sc) = sidechains {
                let offset_map: HashMap<u32, Vec3> =
                    sheet_offsets.iter().copied().collect();
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
            backbone_vertices,
            backbone_tube_indices,
            backbone_tube_index_count,
            backbone_ribbon_indices,
            backbone_ribbon_index_count,
            sheet_offsets,
            backbone_chain_ranges: chain_ranges,
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

use crate::util::sheet_adjust::{
    adjust_bonds_for_sheet, adjust_sidechains_for_sheet,
};
