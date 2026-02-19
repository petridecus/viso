use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GeometryOptions {
    pub tube_radius: f32,
    pub tube_radial_segments: u32,
    pub solvent_radius: f32,
    pub ligand_sphere_radius: f32,
    pub ligand_bond_radius: f32,
}

impl Default for GeometryOptions {
    fn default() -> Self {
        Self {
            tube_radius: 0.3,
            tube_radial_segments: 8,
            solvent_radius: 0.15,
            ligand_sphere_radius: 0.3,
            ligand_bond_radius: 0.12,
        }
    }
}
