//! Minimal entity construction helpers for unit tests.

use molex::entity::EntityIdAllocator;
use molex::ops::codec::{split_into_entities, Coords, CoordsAtom};
use molex::{Element, MoleculeEntity};

fn res_name(s: &str) -> [u8; 3] {
    let bytes = s.as_bytes();
    let mut out = [b' '; 3];
    for (i, &b) in bytes.iter().take(3).enumerate() {
        out[i] = b;
    }
    out
}

fn atom_name(s: &str) -> [u8; 4] {
    let bytes = s.as_bytes();
    let mut out = [b' '; 4];
    for (i, &b) in bytes.iter().take(4).enumerate() {
        out[i] = b;
    }
    out
}

/// Build a minimal protein entity with `residue_count` residues, each
/// having N/CA/C at deterministic positions along the X axis.
#[allow(clippy::expect_used)]
pub fn make_protein_entity(
    entity_id: u32,
    chain_id: u8,
    residue_count: u32,
) -> MoleculeEntity {
    let atom_count = residue_count as usize * 3;
    let mut atoms = Vec::with_capacity(atom_count);
    let mut chain_ids = Vec::with_capacity(atom_count);
    let mut res_names = Vec::with_capacity(atom_count);
    let mut res_nums = Vec::with_capacity(atom_count);
    let mut atom_names = Vec::with_capacity(atom_count);
    let mut elements = Vec::with_capacity(atom_count);

    for r in 0..residue_count {
        let base_x = r as f32 * 3.8;
        for (offset, name, elem) in [
            (0.0, "N", Element::N),
            (1.5, "CA", Element::C),
            (3.0, "C", Element::C),
        ] {
            atoms.push(CoordsAtom {
                x: base_x + offset,
                y: 0.0,
                z: 0.0,
                occupancy: 1.0,
                b_factor: 0.0,
            });
            chain_ids.push(chain_id);
            res_names.push(res_name("ALA"));
            res_nums.push(r as i32 + 1);
            atom_names.push(atom_name(name));
            elements.push(elem);
        }
    }

    let coords = Coords {
        num_atoms: atom_count,
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
        elements,
    };
    let mut entities = split_into_entities(&coords);
    let mut entity = entities.pop().expect("split produced no entities");
    let mut alloc = EntityIdAllocator::new();
    entity.set_id(alloc.from_raw(entity_id));
    entity
}

/// Build a minimal water entity (non-focusable).
#[allow(clippy::expect_used)]
pub fn make_water_entity(entity_id: u32) -> MoleculeEntity {
    let coords = Coords {
        num_atoms: 3,
        atoms: vec![
            CoordsAtom {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                occupancy: 1.0,
                b_factor: 0.0,
            },
            CoordsAtom {
                x: 0.96,
                y: 0.0,
                z: 0.0,
                occupancy: 1.0,
                b_factor: 0.0,
            },
            CoordsAtom {
                x: -0.24,
                y: 0.93,
                z: 0.0,
                occupancy: 1.0,
                b_factor: 0.0,
            },
        ],
        chain_ids: vec![b'W'; 3],
        res_names: vec![res_name("HOH"); 3],
        res_nums: vec![1; 3],
        atom_names: vec![atom_name("O"), atom_name("H1"), atom_name("H2")],
        elements: vec![Element::O, Element::H, Element::H],
    };
    let mut entities = split_into_entities(&coords);
    let mut entity = entities.pop().expect("split produced no entities");
    let mut alloc = EntityIdAllocator::new();
    entity.set_id(alloc.from_raw(entity_id));
    entity
}
