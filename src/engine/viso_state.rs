//! Viso-side derived state rederived from [`molex::Assembly`] on every
//! sync. The render path reads only these types; [`molex::Assembly`] and
//! [`molex::MoleculeEntity`] do not appear in any render-path signature
//! (decision #16 of the assembly migration).
//!
//! - [`VisoEntityState`] — per-entity drawing mode, SS override, topology,
//!   and mesh version.
//! - [`EntityTopology`] — self-sufficient render-ready view of a single
//!   entity: backbone layout, sidechain layout, NA ring layout, sheet
//!   plane normals, per-residue colors, SS types, plus the structural
//!   atom-element / bond / residue metadata the renderer consumes.
//! - [`EntityPositions`] — animator write surface + renderer read surface,
//!   keyed by entity id.
//! - [`SceneRenderState`] — cross-entity renderable data (disulfide
//!   endpoints + H-bond endpoints), rederived on sync.

use std::ops::Range;

use glam::Vec3;
use molex::entity::molecule::id::EntityId;
use molex::{AtomId, CovalentBond, Element, SSType};
use rustc_hash::FxHashMap;

use crate::options::DrawingMode;

// ---------------------------------------------------------------------------
// VisoEntityState
// ---------------------------------------------------------------------------

/// Per-entity state held by [`VisoEngine`].
///
/// Has four fields: the drawing mode chosen for this entity, an optional
/// SS override (used by the viewer to pin SS assignments), the rederived
/// [`EntityTopology`], and a monotonically increasing `mesh_version`
/// used by the mesh cache to detect when geometry needs to be regenerated.
///
/// [`VisoEngine`]: super::VisoEngine
pub struct VisoEntityState {
    /// Drawing mode for this entity (Cartoon / Stick / `BallAndStick`).
    pub drawing_mode: DrawingMode,
    /// Optional secondary-structure override. When present, takes
    /// priority over [`EntityTopology::ss_types`] at render time.
    pub ss_override: Option<Vec<SSType>>,
    /// Rederived render-ready view of this entity.
    pub topology: EntityTopology,
    /// Bumped whenever this entity's geometry needs to be regenerated.
    pub mesh_version: u64,
}

// ---------------------------------------------------------------------------
// EntityTopology
// ---------------------------------------------------------------------------

/// Self-sufficient render-ready view of a single entity.
///
/// Duplicates the structural metadata the renderer needs (atom elements,
/// bond list, residue ranges) at sync time so the render path is a pure
/// function of `(&EntityTopology, &[Vec3])` for per-entity mesh-gen.
/// Neither `&Assembly` nor `&MoleculeEntity` appear in render-path
/// signatures (decision #16).
pub struct EntityTopology {
    /// Per-backbone-chain N/CA/C atom indices, interleaved as `[N0, CA0,
    /// C0, N1, CA1, C1, …]`. One outer vec per backbone chain. Empty for
    /// non-polymer entities.
    pub backbone_chain_layout: Vec<Vec<usize>>,

    /// Sidechain atom indices and bond topology for ball-and-stick /
    /// sidechain-capsule rendering. Empty for non-protein entities.
    pub sidechain_layout: SidechainLayout,

    /// Nucleotide ring atom-index layout per residue, for DNA/RNA
    /// rendering. Empty for non-NA entities.
    pub ring_topology: Vec<NucleotideRingLayout>,

    /// Fitted β-sheet plane normals `(residue_idx, normal)`. Empty when
    /// no multi-strand sheets were detected or fit. See
    /// `renderer/geometry/backbone/sheet_fit.rs`.
    pub sheet_plane_normals: Vec<(u32, Vec3)>,

    /// Per-residue vertex colors (Cartoon-mode). `None` when the current
    /// color scheme doesn't produce per-residue colors.
    pub per_residue_colors: Option<Vec<[f32; 3]>>,

    /// Per-residue secondary structure from
    /// [`molex::Assembly::ss_types`]. Empty for non-protein entities.
    pub ss_types: Vec<SSType>,

    // -- Structural metadata duplicated from MoleculeEntity so mesh-gen
    // can work without an Assembly ref. Memory cost is modest (~30 KB
    // per typical protein). See decision #16. --
    /// Element of each atom, in entity-local index order.
    pub atom_elements: Vec<Element>,
    /// Which residue each atom belongs to (index into
    /// [`residue_atom_ranges`](Self::residue_atom_ranges)).
    pub atom_residue_index: Vec<u32>,
    /// 3-byte residue name (`b"ALA"`, `b"GLY"`, …) per residue.
    pub residue_names: Vec<[u8; 3]>,
    /// Atom-index range per residue, in entity-local indices.
    pub residue_atom_ranges: Vec<Range<u32>>,
    /// Every intra-entity covalent bond. Endpoints use
    /// [`AtomId`](molex::AtomId) so the renderer can map back to
    /// positions via the owning entity.
    pub bonds: Vec<CovalentBond>,
}

/// Sidechain atom-index layout for a single entity.
///
/// All indices are entity-local (into the entity's atom positions slice
/// in [`EntityPositions`]).
pub struct SidechainLayout {
    /// Atom index (entity-local) of each sidechain atom.
    pub atom_indices: Vec<u32>,
    /// Residue index (entity-local) of each sidechain atom. Parallel to
    /// [`atom_indices`](Self::atom_indices).
    pub residue_indices: Vec<u32>,
    /// Packed atom name for each sidechain atom (`"CB"`, `"CG"`, …).
    /// Trimmed of trailing padding; always valid UTF-8. Parallel to
    /// [`atom_indices`](Self::atom_indices).
    pub atom_names: Vec<String>,
    /// Hydrophobicity flag per sidechain atom. Parallel to
    /// [`atom_indices`](Self::atom_indices).
    pub hydrophobicity: Vec<bool>,
    /// Intra-sidechain bonds as `(a, b)` indices into
    /// [`atom_indices`](Self::atom_indices).
    pub bonds: Vec<(u32, u32)>,
    /// Backbone → sidechain bonds as `(ca_atom_idx, cb_layout_idx)` where
    /// `ca_atom_idx` is an entity-local atom index of CA and
    /// `cb_layout_idx` is the index into
    /// [`atom_indices`](Self::atom_indices) of the CB that CA connects
    /// to.
    pub backbone_bonds: Vec<(u32, u32)>,
}

impl SidechainLayout {
    /// Empty layout (no sidechain atoms).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            atom_indices: Vec::new(),
            residue_indices: Vec::new(),
            atom_names: Vec::new(),
            hydrophobicity: Vec::new(),
            bonds: Vec::new(),
            backbone_bonds: Vec::new(),
        }
    }
}

/// Atom-index layout for a single nucleotide's ring(s).
///
/// All indices are entity-local.
pub struct NucleotideRingLayout {
    /// Residue index (entity-local) this ring belongs to.
    pub residue_index: u32,
    /// Six atom indices for the hexagonal ring (N1, C2, N3, C4, C5, C6).
    pub hex_ring: [u32; 6],
    /// Optional five atom indices for the pentagonal ring on purines
    /// (C4, C5, N7, C8, N9). `None` for pyrimidines.
    pub pent_ring: Option<[u32; 5]>,
    /// Atom index of C1' (sugar anchor for stem → backbone connection).
    pub c1_prime: Option<u32>,
    /// NDB base color.
    pub color: [f32; 3],
}

// ---------------------------------------------------------------------------
// EntityPositions
// ---------------------------------------------------------------------------

/// Per-entity animator write surface and renderer read surface.
///
/// Animator writes `per_entity[id]` every frame. Renderer reads. Never
/// touches [`molex::Assembly`] directly. Reconciled on every sync: new
/// entities get an initial reference snapshot inserted; removed entities
/// are dropped.
#[derive(Default)]
pub struct EntityPositions {
    /// Per-entity atom positions, keyed by entity id.
    pub per_entity: FxHashMap<EntityId, Vec<Vec3>>,
}

impl EntityPositions {
    /// Empty positions map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            per_entity: FxHashMap::default(),
        }
    }

    /// Read-only position slice for an entity.
    #[must_use]
    pub fn get(&self, id: EntityId) -> Option<&[Vec3]> {
        self.per_entity.get(&id).map(Vec::as_slice)
    }

    /// Mutable position slice for an entity.
    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut Vec<Vec3>> {
        self.per_entity.get_mut(&id)
    }

    /// Replace the positions for an entity (overwrites existing slot).
    pub fn set(&mut self, id: EntityId, positions: Vec<Vec3>) {
        let _ = self.per_entity.insert(id, positions);
    }

    /// Insert a new entity from a reference position snapshot if absent.
    ///
    /// Used on sync when a new entity joined the assembly: the initial
    /// positions are copied from the assembly's reference positions so
    /// the animator has a visual state to interpolate from.
    pub fn insert_from_reference(&mut self, id: EntityId, reference: &[Vec3]) {
        let _ = self
            .per_entity
            .entry(id)
            .or_insert_with(|| reference.to_vec());
    }

    /// Drop positions for an entity that is no longer in the assembly.
    pub fn remove(&mut self, id: EntityId) {
        let _ = self.per_entity.remove(&id);
    }

    /// Keep only entities for which `keep` returns true.
    pub fn retain(&mut self, mut keep: impl FnMut(EntityId) -> bool) {
        self.per_entity.retain(|&id, _| keep(id));
    }
}

// ---------------------------------------------------------------------------
// SceneRenderState
// ---------------------------------------------------------------------------

/// Cross-entity rendering data derived from [`molex::Assembly`] at sync.
///
/// Disulfides and backbone H-bonds live at scene level because their
/// endpoints span two entities. Both are `(AtomId, AtomId)` pairs so the
/// renderer resolves positions through [`EntityPositions`] at render
/// time without touching [`molex::Assembly`].
#[derive(Default)]
pub struct SceneRenderState {
    /// Disulfide endpoints (SG–SG pairs). Populated from
    /// [`molex::Assembly::disulfides`] on sync.
    pub disulfide_endpoints: Vec<(AtomId, AtomId)>,
    /// Backbone H-bond endpoints (donor N/carbonyl-C heavy atom pairs).
    /// Populated from [`molex::Assembly::hbonds`] on sync.
    pub hbond_endpoints: Vec<(AtomId, AtomId)>,
}

impl SceneRenderState {
    /// Empty scene render state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}
