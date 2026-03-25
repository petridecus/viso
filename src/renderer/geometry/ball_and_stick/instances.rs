//! Per-molecule-type instance generators for ball-and-stick rendering.
//!
//! Each function produces sphere and/or capsule impostor instances for a
//! specific molecule type (polymer, ligand, lipid, ion, water, solvent).

use molex::analysis::bonds::{
    infer_bonds, BondOrder, InferredBond, DEFAULT_TOLERANCE,
};
use molex::{Element, MoleculeEntity};

use super::{
    atom_color, build_atom_residue_map, find_perpendicular, InstanceCollector,
    BALL_RADIUS_SCALE, BOND_RADIUS, DOUBLE_BOND_OFFSET, ION_RADIUS_SCALE,
    STICK_BOND_RADIUS, STICK_SPHERE_RADIUS, WATER_RADIUS,
};
use crate::options::{ColorOptions, DrawingMode};
use crate::renderer::impostor::{CapsuleInstance, SphereInstance};

/// Bond context for coarse-grained bond emission.
struct BondContext<'a> {
    bonds: &'a [InferredBond],
    atoms: &'a [molex::Atom],
}

/// Generate ball-and-stick or stick instances for a polymer
/// (protein/DNA/RNA) entity using full bond inference.
///
/// Unlike `generate_ligand_instances` (which uses per-residue bond
/// inference and misses inter-residue peptide bonds), this uses
/// `infer_bonds` on the full atom set to capture the backbone
/// connectivity.
///
/// When `per_residue_colors` is provided, atoms are colored by their
/// residue's backbone color (chain, SS, score, etc.) instead of CPK
/// element colors.
pub(super) fn generate_polymer_bns_instances(
    entity: &MoleculeEntity,
    atom_offset: u32,
    mode: DrawingMode,
    per_residue_colors: Option<&[[f32; 3]]>,
    out: &mut InstanceCollector,
) {
    let atoms = entity.atom_set();
    let is_stick = matches!(mode, DrawingMode::Stick | DrawingMode::ThinStick);
    let bond_radius = match mode {
        DrawingMode::Stick => STICK_BOND_RADIUS,
        _ => BOND_RADIUS,
    };

    // Build atom-index -> sequential-residue-index map for color lookup.
    // For proteins/NA with residue structure, use residues directly.
    // For others, fall back to no per-residue mapping.
    let atom_residue_map: Option<Vec<usize>> = per_residue_colors
        .and_then(|_| build_atom_residue_map(entity, atoms.len()));

    // Resolve color for an atom: per-residue if available, else CPK.
    let atom_color_fn = |atom_idx: usize, elem: Element| -> [f32; 3] {
        if let (Some(colors), Some(ref map)) =
            (per_residue_colors, &atom_residue_map)
        {
            let seq_idx = map.get(atom_idx).copied().unwrap_or(0);
            if let Some(&c) = colors.get(seq_idx) {
                return c;
            }
        }
        elem.cpk_color()
    };

    // Generate atom spheres
    for (i, atom) in atoms.iter().enumerate() {
        let elem = atom.element;
        if is_stick && elem == Element::H {
            continue;
        }
        let color = atom_color_fn(i, elem);
        let radius = match mode {
            DrawingMode::Stick => STICK_SPHERE_RADIUS,
            DrawingMode::ThinStick => STICK_SPHERE_RADIUS * 0.7,
            _ => elem.vdw_radius() * BALL_RADIUS_SCALE,
        };
        let pick_id = atom_offset + i as u32;

        out.spheres.push(SphereInstance {
            center: [atom.position.x, atom.position.y, atom.position.z, radius],
            color: [color[0], color[1], color[2], pick_id as f32],
        });
    }

    // Full bond inference (captures inter-residue peptide bonds)
    let inferred_bonds = infer_bonds(atoms, DEFAULT_TOLERANCE);
    for bond in &inferred_bonds {
        let elem_a = atoms
            .get(bond.atom_a)
            .map_or(Element::Unknown, |a| a.element);
        let elem_b = atoms
            .get(bond.atom_b)
            .map_or(Element::Unknown, |a| a.element);
        if is_stick && (elem_a == Element::H || elem_b == Element::H) {
            continue;
        }
        let pos_a = atoms[bond.atom_a].position;
        let pos_b = atoms[bond.atom_b].position;
        let color_a = atom_color_fn(bond.atom_a, elem_a);
        let color_b = atom_color_fn(bond.atom_b, elem_b);
        let pick_id = atom_offset + bond.atom_a as u32;
        let a = [pos_a.x, pos_a.y, pos_a.z];
        let b = [pos_b.x, pos_b.y, pos_b.z];

        if !is_stick && bond.order == BondOrder::Double {
            let axis = (pos_b - pos_a).normalize_or_zero();
            let perp = find_perpendicular(axis);
            let offset = perp * DOUBLE_BOND_OFFSET;
            let thin_radius = bond_radius * 0.7;

            let a1 = pos_a + offset;
            let b1 = pos_b + offset;
            out.push_bond(
                [[a1.x, a1.y, a1.z], [b1.x, b1.y, b1.z]],
                thin_radius,
                [color_a, color_b],
                pick_id,
            );
            let a2 = pos_a - offset;
            let b2 = pos_b - offset;
            out.push_bond(
                [[a2.x, a2.y, a2.z], [b2.x, b2.y, b2.z]],
                thin_radius,
                [color_a, color_b],
                pick_id,
            );
        } else {
            out.push_bond([a, b], bond_radius, [color_a, color_b], pick_id);
        }
    }
}

/// Generate ball-and-stick instances for a small molecule entity.
///
/// When `carbon_tint` is `Some(color)`, carbon atoms and their bond
/// endpoints use the tint color instead of CPK gray; heteroatoms (N, O,
/// S, P, ...) keep standard CPK coloring. Pass `None` for plain ligand
/// rendering.
pub(super) fn generate_ligand_instances(
    atoms: &[molex::Atom],
    carbon_tint: Option<[f32; 3]>,
    atom_offset: u32,
    out: &mut InstanceCollector,
) {
    // Generate atom spheres
    for (i, atom) in atoms.iter().enumerate() {
        let elem = atom.element;
        let color = atom_color(elem, carbon_tint);
        let radius = elem.vdw_radius() * BALL_RADIUS_SCALE;
        let pick_id = atom_offset + i as u32;

        out.spheres.push(SphereInstance {
            center: [atom.position.x, atom.position.y, atom.position.z, radius],
            color: [color[0], color[1], color[2], pick_id as f32],
        });
    }

    // Infer bonds per-residue (avoids O(n^2) on large entities)
    let inferred_bonds = infer_bonds_for_atoms(atoms);
    for bond in &inferred_bonds {
        let pos_a = atoms[bond.atom_a].position;
        let pos_b = atoms[bond.atom_b].position;
        let color_a = atom_color(
            atoms
                .get(bond.atom_a)
                .map_or(Element::Unknown, |a| a.element),
            carbon_tint,
        );
        let color_b = atom_color(
            atoms
                .get(bond.atom_b)
                .map_or(Element::Unknown, |a| a.element),
            carbon_tint,
        );
        let pick_id = atom_offset + bond.atom_a as u32;
        let a = [pos_a.x, pos_a.y, pos_a.z];
        let b = [pos_b.x, pos_b.y, pos_b.z];

        if bond.order == BondOrder::Double {
            let axis = (pos_b - pos_a).normalize_or_zero();
            let perp = find_perpendicular(axis);
            let offset = perp * DOUBLE_BOND_OFFSET;
            let thin_radius = BOND_RADIUS * 0.7;

            let a1 = pos_a + offset;
            let b1 = pos_b + offset;
            out.push_bond(
                [[a1.x, a1.y, a1.z], [b1.x, b1.y, b1.z]],
                thin_radius,
                [color_a, color_b],
                pick_id,
            );

            let a2 = pos_a - offset;
            let b2 = pos_b - offset;
            out.push_bond(
                [[a2.x, a2.y, a2.z], [b2.x, b2.y, b2.z]],
                thin_radius,
                [color_a, color_b],
                pick_id,
            );
        } else {
            out.push_bond([a, b], BOND_RADIUS, [color_a, color_b], pick_id);
        }
    }
}

/// Generate coarse-grained lipid instances (phosphorus + head-group
/// spheres, thin tail bonds).
pub(super) fn generate_coarse_lipid_instances(
    atoms: &[molex::Atom],
    lipid_carbon_tint: [f32; 3],
    atom_offset: u32,
    out: &mut InstanceCollector,
) {
    // Generate atom spheres (skip H entirely, CG radii for others)
    for (i, atom) in atoms.iter().enumerate() {
        let elem = atom.element;
        let pick_id = atom_offset + i as u32;

        match elem {
            Element::P => {
                // Phosphorus: medium sphere, CPK orange
                let color = elem.cpk_color();
                out.spheres.push(SphereInstance {
                    center: [
                        atom.position.x,
                        atom.position.y,
                        atom.position.z,
                        1.0,
                    ],
                    color: [color[0], color[1], color[2], pick_id as f32],
                });
            }
            Element::O | Element::N => {
                // Head-group: small spheres, element-colored
                let color = elem.cpk_color();
                out.spheres.push(SphereInstance {
                    center: [
                        atom.position.x,
                        atom.position.y,
                        atom.position.z,
                        0.35,
                    ],
                    color: [color[0], color[1], color[2], pick_id as f32],
                });
            }
            // H, C, and everything else: no sphere in CG mode
            _ => {}
        }
    }

    let bonds = infer_bonds_for_atoms(atoms);
    emit_coarse_bonds(
        &BondContext {
            bonds: &bonds,
            atoms,
        },
        Some(lipid_carbon_tint),
        atom_offset,
        out,
    );
}

/// Emit coarse-grained bond capsules from pre-inferred bonds, skipping
/// hydrogen atoms and using thinner radii for C-C tail bonds.
fn emit_coarse_bonds(
    ctx: &BondContext,
    carbon_tint: Option<[f32; 3]>,
    atom_offset: u32,
    out: &mut InstanceCollector,
) {
    for bond in ctx.bonds {
        let elem_a = ctx
            .atoms
            .get(bond.atom_a)
            .map_or(Element::Unknown, |a| a.element);
        let elem_b = ctx
            .atoms
            .get(bond.atom_b)
            .map_or(Element::Unknown, |a| a.element);
        if elem_a == Element::H || elem_b == Element::H {
            continue;
        }

        let pos_a = ctx.atoms[bond.atom_a].position;
        let pos_b = ctx.atoms[bond.atom_b].position;
        let pick_id = atom_offset + bond.atom_a as u32;

        // C-C tail bonds thin, head-group bonds thicker
        let both_carbon = elem_a == Element::C && elem_b == Element::C;
        let radius = if both_carbon { 0.06 } else { 0.10 };
        let color_a = atom_color(elem_a, carbon_tint);
        let color_b = atom_color(elem_b, carbon_tint);

        out.bonds.push(CapsuleInstance {
            endpoint_a: [pos_a.x, pos_a.y, pos_a.z, radius],
            endpoint_b: [pos_b.x, pos_b.y, pos_b.z, pick_id as f32],
            color_a: [color_a[0], color_a[1], color_a[2], 0.0],
            color_b: [color_b[0], color_b[1], color_b[2], 0.0],
        });
    }
}

/// Generate ion instances (element-colored spheres, no bonds).
pub(super) fn generate_ion_instances(
    atoms: &[molex::Atom],
    atom_offset: u32,
    out: &mut InstanceCollector,
) {
    for (i, atom) in atoms.iter().enumerate() {
        let elem = atom.element;
        let color = elem.cpk_color();
        let radius = elem.vdw_radius() * ION_RADIUS_SCALE;
        let pick_id = atom_offset + i as u32;

        out.spheres.push(SphereInstance {
            center: [atom.position.x, atom.position.y, atom.position.z, radius],
            color: [color[0], color[1], color[2], pick_id as f32],
        });
    }
}

/// Generate water instances (blue oxygen spheres, optional O-H bonds).
pub(super) fn generate_water_instances(
    atoms: &[molex::Atom],
    atom_offset: u32,
    out: &mut InstanceCollector,
) {
    // Water oxygen: light blue sphere
    let water_color: [f32; 3] = [0.5, 0.7, 1.0];

    for (i, atom) in atoms.iter().enumerate() {
        let elem = atom.element;

        // Only render oxygens as visible spheres, skip hydrogens in water
        if elem == Element::O || elem == Element::Unknown {
            let pick_id = atom_offset + i as u32;

            out.spheres.push(SphereInstance {
                center: [
                    atom.position.x,
                    atom.position.y,
                    atom.position.z,
                    WATER_RADIUS,
                ],
                color: [
                    water_color[0],
                    water_color[1],
                    water_color[2],
                    pick_id as f32,
                ],
            });
        }
    }

    // Optionally infer O-H bonds if hydrogens are present
    if atoms.len() > 1 {
        let inferred_bonds = infer_bonds(atoms, DEFAULT_TOLERANCE);
        for bond in &inferred_bonds {
            let pos_a = atoms[bond.atom_a].position;
            let pos_b = atoms[bond.atom_b].position;
            let pick_id = atom_offset + bond.atom_a as u32;
            let radius = BOND_RADIUS * 0.7;

            out.push_bond(
                [[pos_a.x, pos_a.y, pos_a.z], [pos_b.x, pos_b.y, pos_b.z]],
                radius,
                [water_color, [0.9, 0.9, 0.9]], // White for hydrogen end
                pick_id,
            );
        }
    }
}

/// Resolve the carbon tint for a cofactor entity, consulting user color
/// options when available and falling back to built-in per-residue tints.
pub(super) fn resolve_cofactor_tint(
    entity: &MoleculeEntity,
    colors: Option<&ColorOptions>,
) -> [f32; 3] {
    entity.as_small_molecule().map_or_else(
        || [0.5, 0.5, 0.5],
        |sm| {
            colors.map_or_else(
                || cofactor_carbon_tint(sm.residue_name),
                |c| {
                    c.cofactor_tint(
                        std::str::from_utf8(&sm.residue_name)
                            .unwrap_or("")
                            .trim(),
                    )
                },
            )
        },
    )
}

/// Carbon tint color for cofactor rendering, keyed by 3-letter residue
/// name. Applied to carbon atoms and their bond endpoints; heteroatoms
/// keep CPK.
fn cofactor_carbon_tint(res_name: [u8; 3]) -> [f32; 3] {
    let name = std::str::from_utf8(&res_name).unwrap_or("").trim();
    match name {
        "CLA" => [0.2, 0.7, 0.3],         // green
        "CHL" => [0.2, 0.6, 0.35],        // green (chlorophyll B)
        "BCR" | "BCB" => [0.9, 0.5, 0.1], // orange
        "HEM" | "HEC" | "HEA" | "HEB" => [0.7, 0.15, 0.15], // dark red
        "PHO" => [0.5, 0.7, 0.3],         // yellow-green
        "PL9" | "PLQ" => [0.6, 0.5, 0.2], // amber
        _ => [0.5, 0.5, 0.5],             // neutral fallback
    }
}

/// Generate instances for solvent entities.
/// Tiny gray spheres (no bonds), skip H atoms.
pub(super) fn generate_solvent_instances(
    atoms: &[molex::Atom],
    solvent_color: [f32; 3],
    atom_offset: u32,
    out: &mut InstanceCollector,
) {
    const SOLVENT_RADIUS: f32 = 0.15;

    for (i, atom) in atoms.iter().enumerate() {
        let elem = atom.element;
        if elem == Element::H {
            continue;
        }
        let pick_id = atom_offset + i as u32;

        out.spheres.push(SphereInstance {
            center: [
                atom.position.x,
                atom.position.y,
                atom.position.z,
                SOLVENT_RADIUS,
            ],
            color: [
                solvent_color[0],
                solvent_color[1],
                solvent_color[2],
                pick_id as f32,
            ],
        });
    }
}

/// Infer bonds from an atom slice, using pairwise distance check for
/// multi-molecule groups to avoid O(n^2) on large entities. For
/// single-molecule entities (typical ligands), this delegates directly
/// to `infer_bonds`.
fn infer_bonds_for_atoms(atoms: &[molex::Atom]) -> Vec<InferredBond> {
    if atoms.len() < 2 {
        return Vec::new();
    }

    // For small atom sets, just use the standard function
    if atoms.len() <= 500 {
        return infer_bonds(atoms, DEFAULT_TOLERANCE);
    }

    // For large atom sets (e.g. all lipids lumped together),
    // use pairwise distance check which skips H atoms.
    let all_indices: Vec<usize> = (0..atoms.len()).collect();
    let mut bonds = Vec::new();
    infer_bonds_pairwise(&all_indices, atoms, &mut bonds);
    bonds
}

/// Infer bonds by pairwise distance check over a subset of atom indices.
/// Skips hydrogen atoms entirely.
fn infer_bonds_pairwise(
    atom_indices: &[usize],
    atoms: &[molex::Atom],
    bonds: &mut Vec<InferredBond>,
) {
    for (ai, &i) in atom_indices.iter().enumerate() {
        let elem_i = atoms.get(i).map_or(Element::Unknown, |a| a.element);
        if elem_i == Element::H {
            continue;
        }
        let cov_i = elem_i.covalent_radius();

        for &j in &atom_indices[ai + 1..] {
            let elem_j = atoms.get(j).map_or(Element::Unknown, |a| a.element);
            if elem_j == Element::H {
                continue;
            }
            let cov_j = elem_j.covalent_radius();

            let dist = atoms[i].position.distance(atoms[j].position);
            let sum_cov = cov_i + cov_j;
            let threshold = sum_cov + DEFAULT_TOLERANCE;

            if dist <= threshold && dist > 0.4 {
                let order = if dist < sum_cov * 0.9 {
                    BondOrder::Double
                } else {
                    BondOrder::Single
                };
                bonds.push(InferredBond {
                    atom_a: i,
                    atom_b: j,
                    order,
                });
            }
        }
    }
}
