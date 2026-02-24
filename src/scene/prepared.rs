use std::collections::HashMap;

use foldit_conv::{
    coords::{entity::NucleotideRing, MoleculeEntity},
    secondary_structure::SSType,
};
use glam::Vec3;

use super::PerEntityData;
use crate::{
    animation::transition::Transition,
    options::{ColorOptions, DisplayOptions, GeometryOptions},
    renderer::molecular::backbone::ChainRange,
};

/// Fallback color for residues without score data (neutral gray).
pub const FALLBACK_RESIDUE_COLOR: [f32; 3] = [0.7, 0.7, 0.7];

// ---------------------------------------------------------------------------
// Shared sub-structs
// ---------------------------------------------------------------------------

/// Backbone mesh data ready for GPU upload (byte buffers).
#[derive(Clone)]
pub struct BackboneMeshData {
    /// Backbone mesh vertex bytes (shared by tube and ribbon passes).
    pub vertices: Vec<u8>,
    /// Backbone tube index bytes.
    pub tube_indices: Vec<u8>,
    /// Number of backbone tube indices.
    pub tube_index_count: u32,
    /// Backbone ribbon index bytes.
    pub ribbon_indices: Vec<u8>,
    /// Number of backbone ribbon indices.
    pub ribbon_index_count: u32,
    /// Per-residue sheet normal offsets for sidechain adjustment.
    pub sheet_offsets: Vec<(u32, Vec3)>,
    /// Per-chain index ranges and bounding spheres for frustum culling.
    pub chain_ranges: Vec<ChainRange>,
}

/// Ball-and-stick instance data (GPU-ready byte buffers).
#[derive(Clone)]
pub struct BallAndStickInstances {
    /// Sphere instance bytes.
    pub sphere_instances: Vec<u8>,
    /// Number of spheres.
    pub sphere_count: u32,
    /// Capsule (bond) instance bytes.
    pub capsule_instances: Vec<u8>,
    /// Number of capsules.
    pub capsule_count: u32,
    /// Picking capsule bytes.
    pub picking_capsules: Vec<u8>,
    /// Number of picking capsules.
    pub picking_count: u32,
}

/// Nucleic acid instance data (GPU-ready byte buffers).
#[derive(Clone)]
pub struct NucleicAcidInstances {
    /// Stem capsule instance bytes.
    pub stem_instances: Vec<u8>,
    /// Number of stem instances.
    pub stem_count: u32,
    /// Ring polygon instance bytes.
    pub ring_instances: Vec<u8>,
    /// Number of ring instances.
    pub ring_count: u32,
}

/// CPU-side sidechain atom data for animation and interaction.
#[derive(Clone)]
pub struct SidechainCpuData {
    /// Sidechain atom positions.
    pub positions: Vec<Vec3>,
    /// Intra-residue bonds as (atom_idx, atom_idx) pairs.
    pub bonds: Vec<(u32, u32)>,
    /// Backbone-to-sidechain bonds as (CA position, atom idx).
    pub backbone_bonds: Vec<(Vec3, u32)>,
    /// Hydrophobicity flag per sidechain atom.
    pub hydrophobicity: Vec<bool>,
    /// Residue index per sidechain atom.
    pub residue_indices: Vec<u32>,
    /// PDB atom names per sidechain atom.
    pub atom_names: Vec<String>,
}

// ---------------------------------------------------------------------------
// CachedEntityMesh sub-struct
// ---------------------------------------------------------------------------

/// Cached backbone data for a single entity with typed indices for offsetting.
pub(super) struct CachedBackbone {
    pub verts: Vec<u8>,
    pub tube_inds: Vec<u32>,
    pub ribbon_inds: Vec<u32>,
    pub vert_count: u32,
    pub sheet_offsets: Vec<(u32, Vec3)>,
    pub chain_ranges: Vec<ChainRange>,
}

// ---------------------------------------------------------------------------
// Existing types
// ---------------------------------------------------------------------------

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
    /// Backbone mesh data.
    pub backbone: BackboneMeshData,
    /// Sidechain capsule instance bytes.
    pub sidechain_instances: Vec<u8>,
    /// Number of sidechain capsule instances.
    pub sidechain_instance_count: u32,
    /// Ball-and-stick instance data.
    pub bns: BallAndStickInstances,
    /// Nucleic acid instance data.
    pub na: NucleicAcidInstances,
    /// Backbone chains for animation setup.
    pub backbone_chains: Vec<Vec<Vec3>>,
    /// Nucleic acid P-atom chains.
    pub na_chains: Vec<Vec<Vec3>>,
    /// CPU-side sidechain data.
    pub sidechain: SidechainCpuData,
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
    /// Backbone mesh data.
    pub backbone: BackboneMeshData,
    /// Optional sidechain capsule instance bytes.
    pub sidechain_instances: Option<Vec<u8>>,
    /// Number of sidechain capsule instances.
    pub sidechain_instance_count: u32,
}

// ---------------------------------------------------------------------------
// Per-entity cached mesh
// ---------------------------------------------------------------------------

/// Cached mesh data for a single entity. Stored as byte buffers ready for
/// concatenation, plus typed intermediates needed for index offsetting.
pub(super) struct CachedEntityMesh {
    /// Backbone data with typed indices for concatenation.
    pub backbone: CachedBackbone,
    /// Sidechain capsule instance bytes.
    pub sidechain_instances: Vec<u8>,
    /// Number of sidechain instances.
    pub sidechain_instance_count: u32,
    /// Ball-and-stick instance data.
    pub bns: BallAndStickInstances,
    /// Nucleic acid instance data.
    pub na: NucleicAcidInstances,
    /// Number of residues in this entity.
    pub residue_count: u32,
    /// Backbone chains (passthrough for animation).
    pub backbone_chains: Vec<Vec<Vec3>>,
    /// Nucleic acid chains (passthrough).
    pub nucleic_acid_chains: Vec<Vec<Vec3>>,
    /// CPU-side sidechain data.
    pub sidechain: SidechainCpuData,
    /// Secondary structure override.
    pub ss_override: Option<Vec<SSType>>,
    /// Per-residue colors derived from scores (cached to avoid recomputation).
    pub per_residue_colors: Option<Vec<[f32; 3]>>,
    /// Non-protein entities.
    pub non_protein_entities: Vec<MoleculeEntity>,
    /// Nucleotide ring geometry.
    pub nucleic_acid_rings: Vec<NucleotideRing>,
}
