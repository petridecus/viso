use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
/// Geometry detail options for molecular rendering primitives.
pub struct GeometryOptions {
    /// Backbone tube radius in angstroms.
    pub tube_radius: f32,
    /// Number of radial segments around tubes.
    pub tube_radial_segments: u32,
    /// Solvent sphere radius in angstroms.
    pub solvent_radius: f32,
    /// Ligand atom sphere radius.
    pub ligand_sphere_radius: f32,
    /// Ligand bond cylinder radius.
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
