use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[schemars(title = "Geometry", inline)]
#[serde(default)]
/// Geometry detail options for molecular rendering primitives.
pub struct GeometryOptions {
    /// Helix ribbon half-width in angstroms.
    #[schemars(title = "Helix Width", range(min = 0.2, max = 3.0), extend("step" = 0.1))]
    pub helix_width: f32,
    /// Helix ribbon thickness.
    #[schemars(title = "Helix Thickness", range(min = 0.05, max = 1.0), extend("step" = 0.05))]
    pub helix_thickness: f32,
    /// Helix cross-section roundness (0 = flat ribbon, 1 = circular tube).
    #[schemars(title = "Helix Roundness", range(min = 0.0, max = 1.0), extend("step" = 0.05))]
    pub helix_roundness: f32,

    /// Sheet ribbon half-width in angstroms.
    #[schemars(title = "Sheet Width", range(min = 0.2, max = 3.0), extend("step" = 0.1))]
    pub sheet_width: f32,
    /// Sheet ribbon thickness.
    #[schemars(title = "Sheet Thickness", range(min = 0.05, max = 1.0), extend("step" = 0.05))]
    pub sheet_thickness: f32,
    /// Sheet cross-section roundness (0 = flat ribbon, 1 = circular tube).
    #[schemars(title = "Sheet Roundness", range(min = 0.0, max = 1.0), extend("step" = 0.05))]
    pub sheet_roundness: f32,

    /// Coil tube width (diameter).
    #[schemars(title = "Coil Width", range(min = 0.1, max = 1.5), extend("step" = 0.05))]
    pub coil_width: f32,
    /// Coil tube thickness.
    #[schemars(title = "Coil Thickness", range(min = 0.1, max = 1.5), extend("step" = 0.05))]
    pub coil_thickness: f32,
    /// Coil cross-section roundness (0 = flat ribbon, 1 = circular tube).
    #[schemars(title = "Coil Roundness", range(min = 0.0, max = 1.0), extend("step" = 0.05))]
    pub coil_roundness: f32,

    /// Nucleic acid backbone ribbon width.
    #[schemars(title = "NA Width", range(min = 0.2, max = 3.0), extend("step" = 0.1))]
    pub na_width: f32,
    /// Nucleic acid backbone ribbon thickness.
    #[schemars(title = "NA Thickness", range(min = 0.05, max = 1.0), extend("step" = 0.05))]
    pub na_thickness: f32,
    /// Nucleic acid backbone roundness (0 = flat ribbon, 1 = circular tube).
    #[schemars(title = "NA Roundness", range(min = 0.0, max = 1.0), extend("step" = 0.05))]
    pub na_roundness: f32,

    /// Spline segments per residue (higher = smoother curves, more GPU cost).
    #[schemars(title = "Spline Detail", range(min = 4, max = 32), extend("step" = 4))]
    pub segments_per_residue: usize,
    /// Vertices per cross-section ring (higher = rounder tubes, more GPU cost).
    #[schemars(title = "Cross-Section Detail", range(min = 4, max = 16), extend("step" = 2))]
    pub cross_section_verts: usize,

    /// Solvent sphere radius in angstroms.
    #[schemars(skip)]
    pub solvent_radius: f32,
    /// Ligand atom sphere radius.
    #[schemars(skip)]
    pub ligand_sphere_radius: f32,
    /// Ligand bond cylinder radius.
    #[schemars(skip)]
    pub ligand_bond_radius: f32,
}

impl Default for GeometryOptions {
    fn default() -> Self {
        Self {
            helix_width: 1.4,
            helix_thickness: 0.25,
            helix_roundness: 0.0,
            sheet_width: 1.6,
            sheet_thickness: 0.25,
            sheet_roundness: 0.0,
            coil_width: 0.4,
            coil_thickness: 0.4,
            coil_roundness: 1.0,
            na_width: 1.2,
            na_thickness: 0.25,
            na_roundness: 0.0,
            segments_per_residue: 16,
            cross_section_verts: 8,
            solvent_radius: 0.15,
            ligand_sphere_radius: 0.3,
            ligand_bond_radius: 0.12,
        }
    }
}
