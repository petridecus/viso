use glam::Vec3;
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
    /// Vertices per cross-section ring (higher = rounder tubes, more GPU
    /// cost).
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

impl GeometryOptions {
    /// Return a copy with detail scaled down for the given LOD tier.
    ///
    /// The user's `segments_per_residue` and `cross_section_verts` are treated
    /// as the maximum detail (tier 0). Higher tiers scale down:
    /// - Tier 0: 100% of user settings (32 spr, 16 csv at defaults)
    /// - Tier 1: 50% (16 spr, 8 csv)
    /// - Tier 2: 25% (8 spr, 4 csv)
    /// - Tier 3: 12.5% (4 spr, 4 csv)
    pub fn with_lod_tier(&self, tier: u8) -> Self {
        if tier == 0 {
            return self.clone();
        }
        let (spr, csv) = lod_scaled(
            self.segments_per_residue,
            self.cross_section_verts,
            tier,
        );
        Self {
            segments_per_residue: spr,
            cross_section_verts: csv,
            ..self.clone()
        }
    }

    /// Clamp detail so the estimated vertex buffer stays under the wgpu 256 MB
    /// max buffer size. Returns `self` unchanged for small structures.
    pub fn clamped_for_residues(&self, total_residues: usize) -> Self {
        const MAX_BUFFER_BYTES: usize = 256 * 1024 * 1024;
        const VERTEX_BYTES: usize = 52; // size_of::<BackboneVertex>()
        const OVERHEAD: f64 = 1.15;

        let est = |spr: usize, csv: usize| -> usize {
            (total_residues as f64 * spr as f64 * csv as f64 * OVERHEAD)
                as usize
                * VERTEX_BYTES
        };

        // Try current settings first
        if est(self.segments_per_residue, self.cross_section_verts)
            <= MAX_BUFFER_BYTES
        {
            return self.clone();
        }

        // Progressively reduce until it fits
        for tier in 1..=3u8 {
            let (spr, csv) = lod_scaled(
                self.segments_per_residue,
                self.cross_section_verts,
                tier,
            );
            if est(spr, csv) <= MAX_BUFFER_BYTES {
                return self.with_lod_tier(tier);
            }
        }
        self.with_lod_tier(3)
    }
}

/// Scale user detail settings down for an LOD tier.
///
/// Only `spr` (segments per residue) is reduced — `csv` (cross-section
/// vertices) is kept at the user's setting. Reducing csv changes the
/// cross-section shape and normals, making flat ribbons look rounded and
/// incorrectly lit. Reducing spr only makes curves slightly less smooth,
/// which is barely visible at distance.
///
/// - Tier 0: spr 100% (unchanged)
/// - Tier 1: spr 50%
/// - Tier 2: spr 25%
/// - Tier 3: spr 12.5%
pub fn lod_scaled(max_spr: usize, max_csv: usize, tier: u8) -> (usize, usize) {
    let spr = match tier {
        0 => max_spr,
        1 => (max_spr / 2).max(4),
        2 => (max_spr / 4).max(4),
        _ => (max_spr / 8).max(4),
    };
    (spr, max_csv)
}

/// Convenience: LOD tier params from defaults (for backward compat).
pub fn lod_params(tier: u8) -> (usize, usize) {
    let defaults = GeometryOptions::default();
    lod_scaled(
        defaults.segments_per_residue,
        defaults.cross_section_verts,
        tier,
    )
}

/// Select LOD tier based on absolute camera distance in angstroms.
///
/// Uses absolute distance thresholds — screen-space occupancy is what
/// matters, not relative bounding-radius ratios.
///
/// - Close (< 150 A):    tier 0 — 32 spr (full detail)
/// - Medium (150–250 A): tier 1 — 16 spr (half detail)
/// - Far (250–400 A):    tier 2 — 8 spr (quarter detail)
/// - Very far (> 400 A): tier 3 — 4 spr (minimum detail)
pub fn select_lod_tier(camera_distance: f32, _bounding_radius: f32) -> u8 {
    if camera_distance < 150.0 {
        0
    } else if camera_distance < 250.0 {
        1
    } else if camera_distance < 400.0 {
        2
    } else {
        3
    }
}

/// Select LOD tier for a single chain based on its bounding center distance
/// to the camera eye.
pub fn select_chain_lod_tier(chain_center: Vec3, camera_eye: Vec3) -> u8 {
    let distance = (chain_center - camera_eye).length();
    select_lod_tier(distance, 0.0)
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
            segments_per_residue: 32,
            cross_section_verts: 16,
            solvent_radius: 0.15,
            ligand_sphere_radius: 0.3,
            ligand_bond_radius: 0.12,
        }
    }
}
