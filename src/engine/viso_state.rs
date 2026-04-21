//! Viso-side derived state rederived from [`molex::Assembly`] on every
//! sync. The render path reads only these types; [`molex::Assembly`] and
//! [`molex::MoleculeEntity`] do not appear in any render-path signature.
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
use std::sync::Arc;

use glam::Vec3;
use molex::entity::molecule::id::EntityId;
use molex::{
    Assembly, AtomId, CovalentBond, Element, MoleculeEntity, MoleculeType,
    NucleotideRing, SSType,
};
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
    /// Rederived render-ready view of this entity. Arc-wrapped so the
    /// background mesh worker can snapshot a request without cloning
    /// the underlying buffers.
    pub topology: Arc<EntityTopology>,
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
/// signatures.
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

/// Sidechain atom-index layout for a single entity.
///
/// All indices are entity-local (into the entity's atom positions slice
/// in [`EntityPositions`]).
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

/// Atom-index layout for a single nucleotide's ring(s).
///
/// All indices are entity-local.
#[derive(Clone)]
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

// ---------------------------------------------------------------------------
// EntityPositions
// ---------------------------------------------------------------------------

/// Per-entity animator write surface and renderer read surface.
///
/// Animator writes `per_entity[id]` every frame. Renderer reads. Never
/// touches [`molex::Assembly`] directly. Reconciled on every sync: new
/// entities get an initial reference snapshot inserted; removed entities
/// are dropped.
#[derive(Default, Clone)]
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

    /// Rederive cross-entity rendering data from an `Assembly` snapshot.
    #[must_use]
    pub fn from_assembly(assembly: &Assembly) -> Self {
        Self {
            disulfide_endpoints: assembly
                .disulfides()
                .map(|b| (b.a, b.b))
                .collect(),
            hbond_endpoints: hbond_endpoints(assembly),
        }
    }
}

/// Resolve the `Assembly`'s flat-backbone H-bond list back to per-entity
/// `AtomId` pairs.
///
/// `Assembly::hbonds` indexes into the flattened backbone sequence (one
/// `ResidueBackbone` per kept protein residue, concatenated in entity
/// order). To get atom identities the renderer can resolve through
/// [`EntityPositions`], we walk the same concatenation and remember which
/// entity and which local atom index each flat residue slot maps to.
fn hbond_endpoints(assembly: &Assembly) -> Vec<(AtomId, AtomId)> {
    let mut flat_to_atoms: Vec<[AtomId; 4]> = Vec::new();
    for entity in assembly.entities() {
        let Some(protein) = entity.as_protein() else {
            continue;
        };
        let eid = protein.id;
        for residue in &protein.residues {
            let start = residue.atom_range.start as u32;
            flat_to_atoms.push([
                AtomId {
                    entity: eid,
                    index: start,
                },
                AtomId {
                    entity: eid,
                    index: start + 1,
                },
                AtomId {
                    entity: eid,
                    index: start + 2,
                },
                AtomId {
                    entity: eid,
                    index: start + 3,
                },
            ]);
        }
    }

    assembly
        .hbonds()
        .iter()
        .filter_map(|h| {
            let donor = flat_to_atoms.get(h.donor)?;
            let acceptor = flat_to_atoms.get(h.acceptor)?;
            // Donor N → acceptor C=O carbonyl carbon.
            Some((donor[0], acceptor[2]))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// EntityTopology builder
// ---------------------------------------------------------------------------

impl EntityTopology {
    /// Rederive the render-ready view of a single entity.
    ///
    /// `ss` is the per-residue secondary structure for the entity, as
    /// produced by [`Assembly::ss_types`]. For non-protein entities it
    /// should be an empty slice.
    #[must_use]
    pub fn from_entity(entity: &MoleculeEntity, ss: &[SSType]) -> Self {
        let molecule_type = entity.molecule_type();
        match entity {
            MoleculeEntity::Protein(protein) => {
                let backbone_chain_layout =
                    protein_backbone_chain_layout(protein);
                let sidechain_layout = protein_sidechain_layout(protein);
                let (residue_names, residue_atom_ranges, atom_residue_index) =
                    residue_tables(
                        protein.residues.iter().map(|r| {
                            (r.name, r.atom_range.clone())
                        }),
                        protein.atoms.len(),
                    );
                Self {
                    molecule_type,
                    backbone_chain_layout,
                    sidechain_layout,
                    ring_topology: Vec::new(),
                    sheet_plane_normals: Vec::new(),
                    per_residue_colors: None,
                    ss_types: ss.to_vec(),
                    atom_elements: atom_elements(&protein.atoms),
                    atom_residue_index,
                    residue_names,
                    residue_atom_ranges,
                    bonds: protein.bonds.clone(),
                }
            }
            MoleculeEntity::NucleicAcid(na) => {
                let (residue_names, residue_atom_ranges, atom_residue_index) =
                    residue_tables(
                        na.residues
                            .iter()
                            .map(|r| (r.name, r.atom_range.clone())),
                        na.atoms.len(),
                    );
                Self {
                    molecule_type,
                    backbone_chain_layout: na_backbone_chain_layout(na),
                    sidechain_layout: SidechainLayout::empty(),
                    ring_topology: na_ring_topology(na),
                    sheet_plane_normals: Vec::new(),
                    per_residue_colors: None,
                    ss_types: Vec::new(),
                    atom_elements: atom_elements(&na.atoms),
                    atom_residue_index,
                    residue_names,
                    residue_atom_ranges,
                    bonds: na.bonds.clone(),
                }
            }
            MoleculeEntity::SmallMolecule(sm) => Self {
                molecule_type,
                backbone_chain_layout: Vec::new(),
                sidechain_layout: SidechainLayout::empty(),
                ring_topology: Vec::new(),
                sheet_plane_normals: Vec::new(),
                per_residue_colors: None,
                ss_types: Vec::new(),
                atom_elements: atom_elements(&sm.atoms),
                atom_residue_index: vec![0; sm.atoms.len()],
                residue_names: vec![sm.residue_name],
                residue_atom_ranges: std::iter::once(
                    0..sm.atoms.len() as u32,
                )
                .collect(),
                bonds: sm.bonds.clone(),
            },
            MoleculeEntity::Bulk(bulk) => Self {
                molecule_type,
                backbone_chain_layout: Vec::new(),
                sidechain_layout: SidechainLayout::empty(),
                ring_topology: Vec::new(),
                sheet_plane_normals: Vec::new(),
                per_residue_colors: None,
                ss_types: Vec::new(),
                atom_elements: atom_elements(&bulk.atoms),
                atom_residue_index: Vec::new(),
                residue_names: Vec::new(),
                residue_atom_ranges: Vec::new(),
                bonds: Vec::new(),
            },
        }
    }

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
// Builder helpers
// ---------------------------------------------------------------------------

fn atom_elements(atoms: &[molex::Atom]) -> Vec<Element> {
    atoms.iter().map(|a| a.element).collect()
}

/// Build `(residue_names, residue_atom_ranges, atom_residue_index)` from
/// an iterator of `(name, atom_range)` residue metadata.
///
/// The atom-residue index is a flat `Vec<u32>` of length `atom_count`
/// where entry `i` is the residue index containing atom `i`. Atoms not
/// covered by any residue range (e.g. canonicalized-but-dropped residues
/// whose atoms remain in `atoms` but are unreferenced) default to `0`.
fn residue_tables<I>(
    residues: I,
    atom_count: usize,
) -> (Vec<[u8; 3]>, Vec<Range<u32>>, Vec<u32>)
where
    I: IntoIterator<Item = ([u8; 3], Range<usize>)>,
{
    let mut names = Vec::new();
    let mut ranges = Vec::new();
    let mut atom_residue_index = vec![0u32; atom_count];
    for (res_idx, (name, range)) in residues.into_iter().enumerate() {
        names.push(name);
        let start = range.start as u32;
        let end = range.end as u32;
        ranges.push(start..end);
        for slot in &mut atom_residue_index[range] {
            *slot = res_idx as u32;
        }
    }
    (names, ranges, atom_residue_index)
}

/// Interleaved `[N, CA, C]` atom indices per continuous backbone segment.
fn protein_backbone_chain_layout(
    protein: &molex::entity::molecule::protein::ProteinEntity,
) -> Vec<Vec<usize>> {
    use molex::entity::molecule::traits::Polymer;
    let n_segments = protein.segment_count();
    (0..n_segments)
        .map(|seg_idx| {
            let range = protein.segment_range(seg_idx);
            let mut indices = Vec::with_capacity(range.len() * 3);
            for residue in &protein.residues[range] {
                // ProteinEntity::new enforces canonical order: N, CA, C, O
                // as the first four atoms of each kept residue.
                indices.push(residue.atom_range.start);
                indices.push(residue.atom_range.start + 1);
                indices.push(residue.atom_range.start + 2);
            }
            indices
        })
        .collect()
}

/// Flatten protein sidechain atoms with their metadata and topology.
///
/// Walks every kept residue's sidechain heavy atoms (canonical positions
/// `[4..]`, excluding hydrogens). Collects entity-local atom indices,
/// residue indices, names, and hydrophobicity; rebuilds intra-sidechain
/// and backbone→sidechain bond pairs from the entity's bond list.
fn protein_sidechain_layout(
    protein: &molex::entity::molecule::protein::ProteinEntity,
) -> SidechainLayout {
    use molex::chemistry::amino_acids::AminoAcid;

    let mut atom_indices: Vec<u32> = Vec::new();
    let mut residue_indices: Vec<u32> = Vec::new();
    let mut atom_names: Vec<String> = Vec::new();
    let mut hydrophobicity: Vec<bool> = Vec::new();
    // Map entity-local atom index → layout index, so we can resolve
    // bond endpoints back to positions within this layout.
    let mut atom_to_layout: FxHashMap<u32, u32> = FxHashMap::default();

    for (res_idx, residue) in protein.residues.iter().enumerate() {
        let start = residue.atom_range.start;
        let end = residue.atom_range.end;
        if end.saturating_sub(start) < 4 {
            continue;
        }
        let is_hydrophobic =
            AminoAcid::from_code(residue.name).is_some_and(AminoAcid::is_hydrophobic);
        for atom_idx in (start + 4)..end {
            let atom = &protein.atoms[atom_idx];
            if atom.element == Element::H {
                continue;
            }
            let layout_idx = atom_indices.len() as u32;
            atom_indices.push(atom_idx as u32);
            residue_indices.push(res_idx as u32);
            atom_names.push(atom_name_string(atom.name));
            hydrophobicity.push(is_hydrophobic);
            let _ = atom_to_layout.insert(atom_idx as u32, layout_idx);
        }
    }

    let mut bonds: Vec<(u32, u32)> = Vec::new();
    let mut backbone_bonds: Vec<(u32, u32)> = Vec::new();
    for bond in &protein.bonds {
        let a_local = bond.a.index;
        let b_local = bond.b.index;
        let a_side = atom_to_layout.get(&a_local).copied();
        let b_side = atom_to_layout.get(&b_local).copied();
        match (a_side, b_side) {
            (Some(a), Some(b)) => bonds.push((a, b)),
            (Some(side), None) => {
                if let Some(ca) = backbone_anchor(protein, b_local) {
                    backbone_bonds.push((ca, side));
                }
            }
            (None, Some(side)) => {
                if let Some(ca) = backbone_anchor(protein, a_local) {
                    backbone_bonds.push((ca, side));
                }
            }
            (None, None) => {}
        }
    }

    SidechainLayout {
        atom_indices,
        residue_indices,
        atom_names,
        hydrophobicity,
        bonds,
        backbone_bonds,
    }
}

/// If `atom_local` is the CA atom of its residue (canonical offset 1),
/// return its entity-local atom index for wiring as a backbone anchor.
fn backbone_anchor(
    protein: &molex::entity::molecule::protein::ProteinEntity,
    atom_local: u32,
) -> Option<u32> {
    let idx = atom_local as usize;
    let residue = protein
        .residues
        .iter()
        .find(|r| r.atom_range.contains(&idx))?;
    if idx == residue.atom_range.start + 1 {
        Some(atom_local)
    } else {
        None
    }
}

/// Trim padding/null bytes from a 4-byte atom name and return an owned
/// `String`. Falls back to an empty string if the name is not valid
/// UTF-8.
fn atom_name_string(raw: [u8; 4]) -> String {
    std::str::from_utf8(&raw)
        .unwrap_or("")
        .trim_matches(|c: char| c == ' ' || c == '\0')
        .to_owned()
}

// ---------------------------------------------------------------------------
// NA topology helpers
// ---------------------------------------------------------------------------

/// Per-segment P-atom indices for an NA entity, reusing the canonical
/// atom ordering (P is at `residue.atom_range.start` on every kept
/// residue) and `segment_breaks` (residue indices where a new segment
/// starts) populated at `NAEntity::new`.
fn na_backbone_chain_layout(
    na: &molex::entity::molecule::nucleic_acid::NAEntity,
) -> Vec<Vec<usize>> {
    if na.residues.is_empty() {
        return Vec::new();
    }
    let mut segments: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for (res_idx, residue) in na.residues.iter().enumerate() {
        if na.segment_breaks.contains(&res_idx) && !current.is_empty() {
            segments.push(std::mem::take(&mut current));
        }
        current.push(residue.atom_range.start);
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

/// Canonical hexagonal-ring atom names (all bases).
const HEX_RING_NAMES: &[&[u8]] =
    &[b"N1", b"C2", b"N3", b"C4", b"C5", b"C6"];
/// Canonical pentagonal-ring atom names (purines only).
const PENT_RING_NAMES: &[&[u8]] = &[b"C4", b"C5", b"N7", b"C8", b"N9"];

fn is_purine(res_name: [u8; 3]) -> bool {
    let name = std::str::from_utf8(&res_name).unwrap_or("").trim();
    matches!(
        name,
        "DA" | "DG" | "DI" | "A" | "G" | "ADE" | "GUA" | "I" | "RAD" | "RGU"
    )
}

fn ndb_base_color(res_name: [u8; 3]) -> Option<[f32; 3]> {
    let name = std::str::from_utf8(&res_name).unwrap_or("").trim();
    match name {
        "DA" | "A" | "ADE" | "RAD" => Some([0.85, 0.20, 0.20]),
        "DG" | "G" | "GUA" | "RGU" => Some([0.20, 0.80, 0.20]),
        "DC" | "C" | "CYT" | "RCY" => Some([0.90, 0.90, 0.20]),
        "DT" | "THY" => Some([0.20, 0.20, 0.85]),
        "DU" | "U" | "URA" => Some([0.20, 0.85, 0.85]),
        _ => None,
    }
}

fn trim_atom_name(raw: &[u8; 4]) -> &[u8] {
    let end = raw
        .iter()
        .position(|&b| b == 0 || b == b' ')
        .unwrap_or(raw.len());
    &raw[..end]
}

/// Per-residue ring atom indices for an NA entity.
fn na_ring_topology(
    na: &molex::entity::molecule::nucleic_acid::NAEntity,
) -> Vec<NucleotideRingLayout> {
    let mut rings = Vec::new();
    for (res_idx, residue) in na.residues.iter().enumerate() {
        let Some(color) = ndb_base_color(residue.name) else {
            continue;
        };
        let mut name_to_idx: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        for idx in residue.atom_range.clone() {
            let trimmed = trim_atom_name(&na.atoms[idx].name);
            let _ = name_to_idx.insert(trimmed.to_vec(), idx as u32);
        }
        let hex_ring: Option<[u32; 6]> = {
            let mut out = [0u32; 6];
            let mut ok = true;
            for (slot, name) in out.iter_mut().zip(HEX_RING_NAMES) {
                if let Some(&idx) = name_to_idx.get(*name) {
                    *slot = idx;
                } else {
                    ok = false;
                    break;
                }
            }
            ok.then_some(out)
        };
        let Some(hex_ring) = hex_ring else {
            continue;
        };
        let pent_ring: Option<[u32; 5]> = if is_purine(residue.name) {
            let mut out = [0u32; 5];
            let mut ok = true;
            for (slot, name) in out.iter_mut().zip(PENT_RING_NAMES) {
                if let Some(&idx) = name_to_idx.get(*name) {
                    *slot = idx;
                } else {
                    ok = false;
                    break;
                }
            }
            ok.then_some(out)
        } else {
            None
        };
        let c1_prime = name_to_idx
            .get(b"C1'".as_slice())
            .or_else(|| name_to_idx.get(b"C1*".as_slice()))
            .copied();
        rings.push(NucleotideRingLayout {
            residue_index: res_idx as u32,
            hex_ring,
            pent_ring,
            c1_prime,
            color,
        });
    }
    rings
}
