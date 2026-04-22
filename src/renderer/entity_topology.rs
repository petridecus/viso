//! Render-ready per-entity contract.
//!
//! [`EntityTopology`] is the self-sufficient snapshot the mesh worker
//! and main-thread render consumers read from. It's Arc-shared with
//! the background worker so mesh-gen is a pure function of
//! `(&EntityTopology, &[Vec3])` — neither `&Assembly` nor
//! `&MoleculeEntity` appear in any render-path signature.
//!
//! [`SidechainLayout`] and [`NucleotideRingLayout`] are entity-internal
//! geometry descriptions consumed by the sidechain and NA renderers.
//!
//! The `derive_topology` factory that produces an `EntityTopology`
//! from a `MoleculeEntity` lives on the engine side in
//! [`crate::engine::entity_view`] because deriving it is engine-side
//! sync work; this module only defines the contract.

use std::ops::Range;

use glam::Vec3;
use molex::{
    CovalentBond, Element, MoleculeType, NucleotideRing, SSType,
};

// ---------------------------------------------------------------------------
// EntityTopology
// ---------------------------------------------------------------------------

/// Self-sufficient render-ready view of a single entity.
///
/// Duplicates the structural metadata the renderer needs (atom elements,
/// bond list, residue ranges) at sync time so the render path is a pure
/// function of `(&EntityTopology, &[Vec3])` for per-entity mesh-gen.
#[derive(Clone)]
pub struct EntityTopology {
    /// Molecule type of the source entity, so the render dispatcher can
    /// pick the right mesh-gen path without looking at the `Assembly`.
    pub molecule_type: MoleculeType,

    /// Polymer-backbone atom indices, split into continuous chain
    /// segments. Semantics depend on [`molecule_type`](Self::molecule_type):
    ///
    /// - **Protein:** each inner `Vec` is `[N₀, CA₀, C₀, N₁, CA₁, C₁, …]`
    ///   (stride 3), as consumed by the protein backbone renderer.
    /// - **Nucleic acid:** each inner `Vec` is `[P₀, P₁, …]` (stride 1),
    ///   as consumed by the NA renderer.
    /// - **Other:** empty.
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

impl EntityTopology {
    /// Whether this entity renders through the protein backbone path.
    #[must_use]
    pub fn is_protein(&self) -> bool {
        self.molecule_type == MoleculeType::Protein
    }

    /// Whether this entity renders through the nucleic acid path.
    #[must_use]
    pub fn is_nucleic_acid(&self) -> bool {
        matches!(self.molecule_type, MoleculeType::DNA | MoleculeType::RNA)
    }

    /// Reconstruct the backbone chains the protein / NA renderers expect
    /// by resolving atom indices in [`backbone_chain_layout`] against the
    /// provided positions slice.
    ///
    /// Per [`backbone_chain_layout`](Self::backbone_chain_layout) semantics:
    /// protein chains come out as interleaved `[N, CA, C, …]` triplets;
    /// nucleic-acid chains come out as `[P₀, P₁, …]` P-atom sequences.
    #[must_use]
    pub fn backbone_chain_positions(&self, positions: &[Vec3]) -> Vec<Vec<Vec3>> {
        self.backbone_chain_layout
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .filter_map(|&idx| positions.get(idx).copied())
                    .collect()
            })
            .collect()
    }

    /// Resolve the nucleotide ring layouts into renderer-facing
    /// `NucleotideRing` instances, pulling atom positions from the
    /// provided slice.
    #[must_use]
    pub fn resolve_rings(&self, positions: &[Vec3]) -> Vec<NucleotideRing> {
        self.ring_topology
            .iter()
            .filter_map(|layout| layout.resolve(positions))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SidechainLayout
// ---------------------------------------------------------------------------

/// Sidechain atom-index layout for a single entity.
///
/// All indices are entity-local (into the entity's atom positions slice
/// in [`EntityPositions`](crate::engine::positions::EntityPositions)).
#[derive(Clone)]
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

// ---------------------------------------------------------------------------
// NucleotideRingLayout
// ---------------------------------------------------------------------------

/// Atom-index layout for a single nucleotide's ring(s).
///
/// All indices are entity-local.
#[derive(Clone)]
pub struct NucleotideRingLayout {
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

impl NucleotideRingLayout {
    /// Resolve atom indices to world positions using the provided slice.
    /// Returns `None` if any hex-ring atom is out of range.
    #[must_use]
    pub fn resolve(&self, positions: &[Vec3]) -> Option<NucleotideRing> {
        let hex_ring: Vec<Vec3> = self
            .hex_ring
            .iter()
            .map(|&idx| positions.get(idx as usize).copied())
            .collect::<Option<Vec<_>>>()?;
        let pent_ring = self.pent_ring.as_ref().map_or(Vec::new(), |pent| {
            pent.iter()
                .filter_map(|&idx| positions.get(idx as usize).copied())
                .collect()
        });
        let c1_prime = self
            .c1_prime
            .and_then(|idx| positions.get(idx as usize).copied());
        Some(NucleotideRing {
            hex_ring,
            pent_ring,
            c1_prime,
            color: self.color,
        })
    }
}
