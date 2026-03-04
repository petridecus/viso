use foldit_conv::render::sidechain::SidechainAtoms;
use foldit_conv::secondary_structure::SSType;
use foldit_conv::types::entity::MoleculeEntity;
use glam::Vec3;

use crate::engine::scene_data::PerEntityData;
use crate::options::{ColorOptions, DisplayOptions, GeometryOptions};
use crate::renderer::geometry::backbone::ChainRange;
use crate::renderer::picking::PickMap;

// ---------------------------------------------------------------------------
// Shared sub-structs
// ---------------------------------------------------------------------------

/// Backbone mesh data ready for GPU upload (byte buffers).
#[derive(Clone)]
pub(crate) struct BackboneMeshData {
    /// Backbone mesh vertex bytes (shared by tube and ribbon passes).
    pub(crate) vertices: Vec<u8>,
    /// Backbone tube index bytes.
    pub(crate) tube_indices: Vec<u8>,
    /// Number of backbone tube indices.
    pub(crate) tube_index_count: u32,
    /// Backbone ribbon index bytes.
    pub(crate) ribbon_indices: Vec<u8>,
    /// Number of backbone ribbon indices.
    pub(crate) ribbon_index_count: u32,
    /// Per-residue sheet normal offsets for sidechain adjustment.
    pub(crate) sheet_offsets: Vec<(u32, Vec3)>,
    /// Per-chain index ranges and bounding spheres for frustum culling.
    pub(crate) chain_ranges: Vec<ChainRange>,
}

/// Ball-and-stick instance data (GPU-ready byte buffers).
#[derive(Clone)]
pub(crate) struct BallAndStickInstances {
    /// Sphere instance bytes.
    pub(crate) sphere_instances: Vec<u8>,
    /// Number of spheres.
    pub(crate) sphere_count: u32,
    /// Capsule (bond) instance bytes.
    pub(crate) capsule_instances: Vec<u8>,
    /// Number of capsules.
    pub(crate) capsule_count: u32,
}

/// Nucleic acid instance data (GPU-ready byte buffers).
#[derive(Clone)]
pub(crate) struct NucleicAcidInstances {
    /// Stem capsule instance bytes.
    pub(crate) stem_instances: Vec<u8>,
    /// Number of stem instances.
    pub(crate) stem_count: u32,
    /// Ring polygon instance bytes.
    pub(crate) ring_instances: Vec<u8>,
    /// Number of ring instances.
    pub(crate) ring_count: u32,
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

/// Request sent from main thread to scene processor.
pub enum SceneRequest {
    /// Full scene rebuild with per-entity data.
    FullRebuild {
        /// Per-entity data for mesh generation.
        entities: Vec<PerEntityData>,
        /// Current display options for mesh generation.
        display: DisplayOptions,
        /// Current color options for mesh generation.
        colors: ColorOptions,
        /// Current geometry options for mesh generation.
        geometry: GeometryOptions,
    },
    /// Per-frame animation mesh generation (backbone + optional sidechains).
    ///
    /// `na_chains`, `ss_types`, and `per_residue_colors` are optional; when
    /// `None` the background thread uses values cached from the last
    /// `FullRebuild`, avoiding per-frame clones of stable data.
    AnimationFrame {
        /// Interpolated backbone atom chains.
        backbone_chains: Vec<Vec<Vec3>>,
        /// Nucleic acid P-atom chains. `None` = use cached from last rebuild.
        na_chains: Option<Vec<Vec<Vec3>>>,
        /// Optional interpolated sidechain data.
        sidechains: Option<SidechainAtoms>,
        /// Secondary structure types. `None` = use cached from last rebuild.
        ss_types: Option<Vec<SSType>>,
        /// Per-residue colors. `None` = use cached from last rebuild.
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
pub(crate) struct PreparedScene {
    /// Backbone mesh data.
    pub(crate) backbone: BackboneMeshData,
    /// Sidechain capsule instance bytes.
    pub(crate) sidechain_instances: Vec<u8>,
    /// Number of sidechain capsule instances.
    pub(crate) sidechain_instance_count: u32,
    /// Ball-and-stick instance data.
    pub(crate) bns: BallAndStickInstances,
    /// Nucleic acid instance data.
    pub(crate) na: NucleicAcidInstances,
    /// Mapping from raw GPU pick IDs to typed pick targets.
    pub(crate) pick_map: PickMap,
}

/// Pre-computed animation frame data, ready for GPU upload.
#[derive(Clone)]
pub(crate) struct PreparedAnimationFrame {
    /// Backbone mesh data.
    pub(crate) backbone: BackboneMeshData,
    /// Optional sidechain capsule instance bytes.
    pub(crate) sidechain_instances: Option<Vec<u8>>,
    /// Number of sidechain capsule instances.
    pub(crate) sidechain_instance_count: u32,
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
    /// Non-protein entities (for BnS pick ID offset calculation).
    pub non_protein_entities: Vec<MoleculeEntity>,
}
