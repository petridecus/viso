//! Cross-entity scene render state.
//!
//! [`SceneRenderState`] is the scene-wide derived output of an
//! [`Assembly`] snapshot. Endpoint lists (H-bonds, disulfides) come
//! directly from assembly analyses; resolved [`StructuralBond`]
//! instances are produced per-frame by
//! [`update_structural_bonds`](SceneRenderState::update_structural_bonds)
//! using per-entity annotations (visibility, drawing mode) the engine
//! passes in via [`BondResolveInput`].

use glam::Vec3;
use molex::entity::molecule::id::EntityId;
use molex::{Assembly, AtomId};
use rustc_hash::FxHashMap;

use super::entity_view::{EntityView, RibbonBackbone};
use super::positions::EntityPositions;
use crate::options::{BondOptions, ColorOptions, DrawingMode};
use crate::renderer::geometry::bond::StructuralBond;

// ---------------------------------------------------------------------------
// SceneRenderState
// ---------------------------------------------------------------------------

/// Cross-entity rendering data derived from [`Assembly`].
///
/// Populated in two phases per frame:
/// 1. [`from_assembly`](Self::from_assembly) — rederives the endpoint lists
///    from the latest Assembly snapshot.
/// 2. [`update_structural_bonds`](Self::update_structural_bonds) — turns
///    endpoints into render-ready [`StructuralBond`] capsules using per-entity
///    annotations (visibility, Cartoon ribbon anchors). Runs each sync; needs
///    engine state the pure Assembly derivation doesn't see.
#[derive(Clone, Default)]
pub(crate) struct SceneRenderState {
    /// Disulfide endpoints (SG–SG pairs). Populated from
    /// [`Assembly::disulfides`](molex::Assembly::disulfides) on sync.
    pub(crate) disulfide_endpoints: Vec<(AtomId, AtomId)>,
    /// Backbone H-bond endpoints (donor N / carbonyl C heavy atom
    /// pairs). Populated from [`Assembly::hbonds`](molex::Assembly::hbonds)
    /// on sync.
    pub(crate) hbond_endpoints: Vec<(AtomId, AtomId)>,
    /// Render-ready structural bond capsules. Repopulated on every
    /// sync via [`update_structural_bonds`](Self::update_structural_bonds).
    structural_bonds: Vec<StructuralBond>,
}

impl SceneRenderState {
    /// Empty scene render state.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Rederive the Assembly-pure portion of scene state (endpoint
    /// lists). Does not produce [`structural_bonds`] — call
    /// [`update_structural_bonds`](Self::update_structural_bonds) after
    /// the engine has built its per-entity views.
    ///
    /// [`structural_bonds`]: Self::structural_bonds
    #[must_use]
    pub(crate) fn from_assembly(assembly: &Assembly) -> Self {
        Self {
            disulfide_endpoints: assembly
                .disulfides()
                .map(|b| (b.a, b.b))
                .collect(),
            hbond_endpoints: hbond_endpoints(assembly),
            structural_bonds: Vec::new(),
        }
    }

    /// Current render-ready bond list. Read by the GPU bond renderer.
    #[must_use]
    pub(crate) fn structural_bonds(&self) -> &[StructuralBond] {
        &self.structural_bonds
    }

    /// Repopulate [`structural_bonds`](Self::structural_bonds) for the
    /// current frame.
    ///
    /// Filters endpoints whose entity is toggled off, and re-anchors
    /// H-bond endpoints to the ribbon for Cartoon-mode protein
    /// entities so dashed capsules land on the rendered curve rather
    /// than on raw N/C atom positions.
    pub(crate) fn update_structural_bonds(
        &mut self,
        input: &BondResolveInput<'_>,
    ) {
        let mut bonds = Vec::with_capacity(
            self.hbond_endpoints.len() + self.disulfide_endpoints.len(),
        );

        if input.options.hydrogen_bonds.visible {
            push_hbonds(&mut bonds, &self.hbond_endpoints, input);
        }
        if input.options.disulfide_bonds.visible {
            push_disulfides(&mut bonds, &self.disulfide_endpoints, input);
        }

        self.structural_bonds = bonds;
    }
}

// ---------------------------------------------------------------------------
// BondResolveInput
// ---------------------------------------------------------------------------

/// Complete per-frame input to structural-bond resolution. References
/// existing engine state directly — no per-entity bundle layer — plus
/// a precomputed [`RibbonBackbone`] cache that's only worth building
/// once per Cartoon-mode protein entity per sync.
pub(crate) struct BondResolveInput<'a> {
    /// Per-entity atom positions. Provides the fallback raw atom
    /// coordinates when ribbon projection isn't applicable.
    pub(crate) positions: &'a EntityPositions,
    /// Engine per-entity state (for `drawing_mode` + topology lookups).
    pub(crate) entity_views: &'a FxHashMap<EntityId, EntityView>,
    /// Engine visibility overlay. Missing → visible.
    pub(crate) entity_visibility: &'a FxHashMap<EntityId, bool>,
    /// Precomputed ribbon projections for Cartoon-mode protein entities
    /// that had enough residues to project. Missing → raw atom
    /// positions are used instead.
    pub(crate) ribbons: &'a FxHashMap<EntityId, RibbonBackbone>,
    /// Bond visibility + style + radii.
    pub(crate) options: &'a BondOptions,
    /// Palette of reference colors (H-bond / disulfide tints).
    pub(crate) colors: &'a ColorOptions,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the `Assembly`'s flat-backbone H-bond list to per-entity
/// `AtomId` pairs.
///
/// `Assembly::hbonds` indexes into the flattened backbone sequence
/// (one `ResidueBackbone` per kept protein residue, concatenated in
/// entity order). To get atom identities the renderer can resolve
/// through [`EntityPositions`], walk the same concatenation and
/// remember which entity and which local atom index each flat residue
/// slot maps to.
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

fn push_hbonds(
    out: &mut Vec<StructuralBond>,
    endpoints: &[(AtomId, AtomId)],
    input: &BondResolveInput<'_>,
) {
    let color = tinted(input.colors.band_hbond);
    let opts = &input.options.hydrogen_bonds;

    for &(donor, acceptor) in endpoints {
        if !endpoint_visible(donor, input) || !endpoint_visible(acceptor, input)
        {
            continue;
        }
        let Some(pos_a) =
            hbond_endpoint_position(donor, HBondSide::Donor, input)
        else {
            continue;
        };
        let Some(pos_b) =
            hbond_endpoint_position(acceptor, HBondSide::Acceptor, input)
        else {
            continue;
        };
        out.push(StructuralBond {
            pos_a,
            pos_b,
            color,
            radius: opts.radius,
            residue_idx: donor.index,
            style: opts.style,
            emissive: 0.6,
            opacity: 0.5,
        });
    }
}

fn push_disulfides(
    out: &mut Vec<StructuralBond>,
    endpoints: &[(AtomId, AtomId)],
    input: &BondResolveInput<'_>,
) {
    let color = tinted(input.colors.band_disulfide);
    let opts = &input.options.disulfide_bonds;

    for &(a, b) in endpoints {
        if !endpoint_visible(a, input) || !endpoint_visible(b, input) {
            continue;
        }
        let Some(pos_a) = atom_position(a, input.positions) else {
            continue;
        };
        let Some(pos_b) = atom_position(b, input.positions) else {
            continue;
        };
        out.push(StructuralBond {
            pos_a,
            pos_b,
            color,
            radius: opts.radius,
            residue_idx: 0,
            style: opts.style,
            emissive: 0.6,
            opacity: 0.5,
        });
    }
}

/// Whether an endpoint's entity is currently visible. Entities without
/// a visibility entry are assumed visible (new-entity safety).
fn endpoint_visible(atom: AtomId, input: &BondResolveInput<'_>) -> bool {
    input
        .entity_visibility
        .get(&atom.entity)
        .copied()
        .unwrap_or(true)
}

/// Raw world-space atom position via [`EntityPositions`].
fn atom_position(atom: AtomId, positions: &EntityPositions) -> Option<Vec3> {
    positions
        .get(atom.entity)
        .and_then(|slice| slice.get(atom.index as usize).copied())
}

/// H-bond endpoint position.
///
/// For Cartoon-mode entities with a ribbon projection, uses the
/// per-residue projected position (N for donor, C for acceptor) so the
/// capsule attaches to the rendered curve. Otherwise falls back to the
/// raw atom position.
fn hbond_endpoint_position(
    atom: AtomId,
    side: HBondSide,
    input: &BondResolveInput<'_>,
) -> Option<Vec3> {
    input
        .entity_views
        .get(&atom.entity)
        .filter(|view| view.drawing_mode == DrawingMode::Cartoon)
        .and_then(|view| {
            let ribbon = input.ribbons.get(&atom.entity)?;
            let residue = view
                .topology
                .atom_residue_index
                .get(atom.index as usize)
                .copied()?;
            match side {
                HBondSide::Donor => ribbon.n_at(residue),
                HBondSide::Acceptor => ribbon.c_at(residue),
            }
        })
        .or_else(|| atom_position(atom, input.positions))
}

#[derive(Clone, Copy)]
enum HBondSide {
    Donor,
    Acceptor,
}

/// Lighten a base color 50% toward white for on-screen pop.
fn tinted(base: [f32; 3]) -> [f32; 3] {
    let w = 0.5;
    [
        base[0] + (1.0 - base[0]) * w,
        base[1] + (1.0 - base[1]) * w,
        base[2] + (1.0 - base[2]) * w,
    ]
}
