//! Per-molecule-type instance generators for ball-and-stick rendering.
//!
//! Each function produces sphere and/or capsule impostor instances for a
//! specific molecule type (polymer, ligand, lipid, ion, water, solvent).
//! All generators read from `(EntityTopology, positions)` — the render
//! path never sees `&MoleculeEntity` or `&Assembly`.

use glam::Vec3;
use molex::{BondOrder, CovalentBond, Element};

use super::{
    atom_color, find_perpendicular, InstanceCollector, BALL_RADIUS_SCALE,
    BOND_RADIUS, DOUBLE_BOND_OFFSET, ION_RADIUS_SCALE, STICK_BOND_RADIUS,
    STICK_SPHERE_RADIUS, WATER_RADIUS,
};
use crate::engine::viso_state::EntityTopology;
use crate::options::{ColorOptions, DrawingMode};
use crate::renderer::impostor::{CapsuleInstance, SphereInstance};

/// Generate ball-and-stick or stick instances for a polymer
/// (protein/DNA/RNA) entity using the entity's pre-populated bond graph.
///
/// When `per_residue_colors` is provided, atoms are colored by their
/// residue's backbone color (chain, SS, score, etc.) instead of CPK
/// element colors.
pub(super) fn generate_polymer_bns_instances(
    topology: &EntityTopology,
    positions: &[Vec3],
    atom_offset: u32,
    mode: DrawingMode,
    per_residue_colors: Option<&[[f32; 3]]>,
    out: &mut InstanceCollector,
) {
    let elements = &topology.atom_elements;
    let is_stick = matches!(mode, DrawingMode::Stick | DrawingMode::ThinStick);
    let bond_radius = match mode {
        DrawingMode::Stick => STICK_BOND_RADIUS,
        _ => BOND_RADIUS,
    };

    // Resolve color for an atom: per-residue if available, else CPK.
    let atom_color_fn = |atom_idx: usize, elem: Element| -> [f32; 3] {
        if let Some(colors) = per_residue_colors {
            let res = topology
                .atom_residue_index
                .get(atom_idx)
                .copied()
                .unwrap_or(0) as usize;
            if let Some(&c) = colors.get(res) {
                return c;
            }
        }
        elem.cpk_color()
    };

    // Generate atom spheres
    for (i, (&elem, &pos)) in elements.iter().zip(positions.iter()).enumerate() {
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
            center: [pos.x, pos.y, pos.z, radius],
            color: [color[0], color[1], color[2], pick_id as f32],
        });
    }

    emit_topology_bonds(
        &topology.bonds,
        elements,
        positions,
        atom_offset,
        &atom_color_fn,
        is_stick,
        bond_radius,
        out,
    );
}

/// Generate ball-and-stick instances for a small molecule entity.
///
/// When `carbon_tint` is `Some(color)`, carbon atoms and their bond
/// endpoints use the tint color instead of CPK gray; heteroatoms (N, O,
/// S, P, ...) keep standard CPK coloring. Pass `None` for plain ligand
/// rendering.
pub(super) fn generate_ligand_instances(
    topology: &EntityTopology,
    positions: &[Vec3],
    carbon_tint: Option<[f32; 3]>,
    atom_offset: u32,
    out: &mut InstanceCollector,
) {
    let elements = &topology.atom_elements;
    // Generate atom spheres
    for (i, (&elem, &pos)) in elements.iter().zip(positions.iter()).enumerate() {
        let color = atom_color(elem, carbon_tint);
        let radius = elem.vdw_radius() * BALL_RADIUS_SCALE;
        let pick_id = atom_offset + i as u32;

        out.spheres.push(SphereInstance {
            center: [pos.x, pos.y, pos.z, radius],
            color: [color[0], color[1], color[2], pick_id as f32],
        });
    }

    let color_fn = |_atom_idx: usize, elem: Element| atom_color(elem, carbon_tint);
    emit_topology_bonds(
        &topology.bonds,
        elements,
        positions,
        atom_offset,
        &color_fn,
        false,
        BOND_RADIUS,
        out,
    );
}

/// Generate coarse-grained lipid instances (phosphorus + head-group
/// spheres, thin tail bonds).
pub(super) fn generate_coarse_lipid_instances(
    topology: &EntityTopology,
    positions: &[Vec3],
    lipid_carbon_tint: [f32; 3],
    atom_offset: u32,
    out: &mut InstanceCollector,
) {
    let elements = &topology.atom_elements;
    // Generate atom spheres (skip H entirely, CG radii for others)
    for (i, (&elem, &pos)) in elements.iter().zip(positions.iter()).enumerate() {
        let pick_id = atom_offset + i as u32;

        match elem {
            Element::P => {
                let color = elem.cpk_color();
                out.spheres.push(SphereInstance {
                    center: [pos.x, pos.y, pos.z, 1.0],
                    color: [color[0], color[1], color[2], pick_id as f32],
                });
            }
            Element::O | Element::N => {
                let color = elem.cpk_color();
                out.spheres.push(SphereInstance {
                    center: [pos.x, pos.y, pos.z, 0.35],
                    color: [color[0], color[1], color[2], pick_id as f32],
                });
            }
            _ => {}
        }
    }

    // Thin tail bonds for C-C, thicker head-group bonds.
    for bond in &topology.bonds {
        let a = bond.a.index as usize;
        let b = bond.b.index as usize;
        let elem_a = elements.get(a).copied().unwrap_or(Element::Unknown);
        let elem_b = elements.get(b).copied().unwrap_or(Element::Unknown);
        if elem_a == Element::H || elem_b == Element::H {
            continue;
        }
        let (Some(&pos_a), Some(&pos_b)) = (positions.get(a), positions.get(b))
        else {
            continue;
        };
        let pick_id = atom_offset + a as u32;
        let both_carbon = elem_a == Element::C && elem_b == Element::C;
        let radius = if both_carbon { 0.06 } else { 0.10 };
        let color_a = atom_color(elem_a, Some(lipid_carbon_tint));
        let color_b = atom_color(elem_b, Some(lipid_carbon_tint));
        out.bonds.push(CapsuleInstance {
            endpoint_a: [pos_a.x, pos_a.y, pos_a.z, radius],
            endpoint_b: [pos_b.x, pos_b.y, pos_b.z, pick_id as f32],
            color_a: [color_a[0], color_a[1], color_a[2], 0.0],
            color_b: [color_b[0], color_b[1], color_b[2], 0.0],
        });
    }
}

/// Resolve the carbon tint for a cofactor entity, consulting user color
/// options when available and falling back to built-in per-residue tints.
pub(super) fn resolve_cofactor_tint(
    topology: &EntityTopology,
    colors: Option<&ColorOptions>,
) -> [f32; 3] {
    let Some(&res_name) = topology.residue_names.first() else {
        return [0.5, 0.5, 0.5];
    };
    colors.map_or_else(
        || cofactor_carbon_tint(res_name),
        |c| {
            c.cofactor_tint(
                std::str::from_utf8(&res_name).unwrap_or("").trim(),
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
        "CLA" => [0.2, 0.7, 0.3],
        "CHL" => [0.2, 0.6, 0.35],
        "BCR" | "BCB" => [0.9, 0.5, 0.1],
        "HEM" | "HEC" | "HEA" | "HEB" => [0.7, 0.15, 0.15],
        "PHO" => [0.5, 0.7, 0.3],
        "PL9" | "PLQ" => [0.6, 0.5, 0.2],
        _ => [0.5, 0.5, 0.5],
    }
}

/// Generate ion instances (element-colored spheres, no bonds).
pub(super) fn generate_ion_instances(
    topology: &EntityTopology,
    positions: &[Vec3],
    atom_offset: u32,
    out: &mut InstanceCollector,
) {
    for (i, (&elem, &pos)) in
        topology.atom_elements.iter().zip(positions.iter()).enumerate()
    {
        let color = elem.cpk_color();
        let radius = elem.vdw_radius() * ION_RADIUS_SCALE;
        let pick_id = atom_offset + i as u32;

        out.spheres.push(SphereInstance {
            center: [pos.x, pos.y, pos.z, radius],
            color: [color[0], color[1], color[2], pick_id as f32],
        });
    }
}

/// Generate water instances (blue oxygen spheres, optional O-H bonds).
pub(super) fn generate_water_instances(
    topology: &EntityTopology,
    positions: &[Vec3],
    atom_offset: u32,
    out: &mut InstanceCollector,
) {
    let water_color: [f32; 3] = [0.5, 0.7, 1.0];
    let elements = &topology.atom_elements;

    for (i, (&elem, &pos)) in elements.iter().zip(positions.iter()).enumerate() {
        if elem == Element::O || elem == Element::Unknown {
            let pick_id = atom_offset + i as u32;
            out.spheres.push(SphereInstance {
                center: [pos.x, pos.y, pos.z, WATER_RADIUS],
                color: [
                    water_color[0],
                    water_color[1],
                    water_color[2],
                    pick_id as f32,
                ],
            });
        }
    }

    // Emit explicit O-H bonds when topology records them.
    for bond in &topology.bonds {
        let a = bond.a.index as usize;
        let b = bond.b.index as usize;
        let (Some(&pos_a), Some(&pos_b)) = (positions.get(a), positions.get(b))
        else {
            continue;
        };
        let pick_id = atom_offset + a as u32;
        let radius = BOND_RADIUS * 0.7;
        out.push_bond(
            [[pos_a.x, pos_a.y, pos_a.z], [pos_b.x, pos_b.y, pos_b.z]],
            radius,
            [water_color, [0.9, 0.9, 0.9]],
            pick_id,
        );
    }
}

/// Generate instances for solvent entities.
/// Tiny gray spheres (no bonds), skip H atoms.
pub(super) fn generate_solvent_instances(
    topology: &EntityTopology,
    positions: &[Vec3],
    solvent_color: [f32; 3],
    atom_offset: u32,
    out: &mut InstanceCollector,
) {
    const SOLVENT_RADIUS: f32 = 0.15;

    for (i, (&elem, &pos)) in
        topology.atom_elements.iter().zip(positions.iter()).enumerate()
    {
        if elem == Element::H {
            continue;
        }
        let pick_id = atom_offset + i as u32;

        out.spheres.push(SphereInstance {
            center: [pos.x, pos.y, pos.z, SOLVENT_RADIUS],
            color: [
                solvent_color[0],
                solvent_color[1],
                solvent_color[2],
                pick_id as f32,
            ],
        });
    }
}

/// Emit bond capsules from a topology bond list, handling double bonds
/// via parallel-capsule offset and skipping stick-mode hydrogens.
#[allow(clippy::too_many_arguments)]
fn emit_topology_bonds(
    bonds: &[CovalentBond],
    elements: &[Element],
    positions: &[Vec3],
    atom_offset: u32,
    color_fn: &impl Fn(usize, Element) -> [f32; 3],
    skip_hydrogens: bool,
    bond_radius: f32,
    out: &mut InstanceCollector,
) {
    for bond in bonds {
        let a = bond.a.index as usize;
        let b = bond.b.index as usize;
        let elem_a = elements.get(a).copied().unwrap_or(Element::Unknown);
        let elem_b = elements.get(b).copied().unwrap_or(Element::Unknown);
        if skip_hydrogens && (elem_a == Element::H || elem_b == Element::H) {
            continue;
        }
        let (Some(&pos_a), Some(&pos_b)) = (positions.get(a), positions.get(b))
        else {
            continue;
        };
        let color_a = color_fn(a, elem_a);
        let color_b = color_fn(b, elem_b);
        let pick_id = atom_offset + a as u32;
        let ap = [pos_a.x, pos_a.y, pos_a.z];
        let bp = [pos_b.x, pos_b.y, pos_b.z];

        if !skip_hydrogens && bond.order == BondOrder::Double {
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
            out.push_bond([ap, bp], bond_radius, [color_a, color_b], pick_id);
        }
    }
}
