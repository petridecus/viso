//! Cross-section profiles and extrusion for backbone geometry.
//!
//! Maps secondary structure types to shape parameters (width, thickness,
//! roundness) and extrudes 3D cross-sections at each spline point.

use foldit_conv::secondary_structure::SSType;
use glam::Vec3;

use super::spline::SplinePoint;
use super::BackboneVertex;
use crate::options::GeometryOptions;

// ==================== CROSS-SECTION PROFILE ====================

/// Interpolated per-spline-point geometry parameters.
#[derive(Clone, Copy)]
pub(crate) struct CrossSectionProfile {
    pub width: f32,
    pub thickness: f32,
    pub roundness: f32,
    pub radial_blend: f32,
    pub color: [f32; 3],
    pub residue_idx: u32,
}

impl CrossSectionProfile {
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            width: self.width + (other.width - self.width) * t,
            thickness: self.thickness
                + (other.thickness - self.thickness) * t,
            roundness: self.roundness
                + (other.roundness - self.roundness) * t,
            radial_blend: self.radial_blend
                + (other.radial_blend - self.radial_blend) * t,
            color: [
                self.color[0] + (other.color[0] - self.color[0]) * t,
                self.color[1] + (other.color[1] - self.color[1]) * t,
                self.color[2] + (other.color[2] - self.color[2]) * t,
            ],
            residue_idx: if t < 0.5 {
                self.residue_idx
            } else {
                other.residue_idx
            },
        }
    }
}

/// Build a `CrossSectionProfile` for a single residue from the geometry
/// options and SS type.
pub(crate) fn resolve_profile(
    ss: SSType,
    residue_idx: u32,
    color: [f32; 3],
    geo: &GeometryOptions,
) -> CrossSectionProfile {
    let (width, thickness, roundness, radial_blend) = match ss {
        SSType::Helix => (
            geo.helix_width,
            geo.helix_thickness,
            geo.helix_roundness,
            1.0_f32,
        ),
        SSType::Sheet => (
            geo.sheet_width,
            geo.sheet_thickness,
            geo.sheet_roundness,
            0.0,
        ),
        SSType::Coil => (
            geo.coil_width,
            geo.coil_thickness,
            geo.coil_roundness,
            0.0,
        ),
    };
    CrossSectionProfile {
        width,
        thickness,
        roundness,
        radial_blend,
        color,
        residue_idx,
    }
}

/// Build a `CrossSectionProfile` for a nucleic acid residue.
pub(crate) fn resolve_na_profile(
    residue_idx: u32,
    color: [f32; 3],
    geo: &GeometryOptions,
) -> CrossSectionProfile {
    CrossSectionProfile {
        width: geo.na_width,
        thickness: geo.na_thickness,
        roundness: geo.na_roundness,
        radial_blend: 0.0,
        color,
        residue_idx,
    }
}

// ==================== PROFILE INTERPOLATION ====================

/// Interpolate per-residue profiles to spline resolution with smooth
/// transitions at SS boundaries.
pub(crate) fn interpolate_profiles(
    profiles: &[CrossSectionProfile],
    total_spline: usize,
    n_residues: usize,
) -> Vec<CrossSectionProfile> {
    if n_residues < 2 || total_spline < 2 {
        return vec![profiles[0]; total_spline];
    }

    // Profile lerp between neighboring residues produces smooth
    // interpolation through SS boundaries automatically.
    (0..total_spline)
        .map(|i| {
            let frac = i as f32 / (total_spline - 1) as f32;
            let rf = frac * (n_residues - 1) as f32;
            let r0 = (rf.floor() as usize).min(n_residues - 1);
            let r1 = (r0 + 1).min(n_residues - 1);
            let t = rf - r0 as f32;

            let mut result = profiles[r0].lerp(&profiles[r1], t);
            result.residue_idx = if t < 0.5 {
                profiles[r0].residue_idx
            } else {
                profiles[r1].residue_idx
            };
            result
        })
        .collect()
}

// ==================== CROSS-SECTION EXTRUSION ====================

/// Extrude a cross-section ring at a single spline point.
pub(crate) fn extrude_cross_section(
    frame: &SplinePoint,
    hw: f32,
    ht: f32,
    roundness: f32,
    color: [f32; 3],
    residue_idx: u32,
    cross_section_verts: usize,
    vertices: &mut Vec<BackboneVertex>,
) {
    for k in 0..cross_section_verts {
        let angle = (k as f32 / cross_section_verts as f32)
            * std::f32::consts::TAU;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Rectangular corner position
        let rect_x = cos_a.signum() * hw;
        let rect_y = sin_a.signum() * ht;

        // Circular position
        let circ_x = cos_a * hw;
        let circ_y = sin_a * ht;

        // Blend between rectangular and circular
        let x = rect_x + (circ_x - rect_x) * roundness;
        let y = rect_y + (circ_y - rect_y) * roundness;

        let offset = frame.binormal * x + frame.normal * y;
        let pos = frame.pos + offset;

        if roundness > 0.5 {
            // Round: smooth cylindrical normals via center_pos encoding
            let smooth_normal = offset.normalize_or_zero();
            vertices.push(BackboneVertex {
                position: pos.into(),
                normal: smooth_normal.into(),
                color,
                residue_idx,
                center_pos: frame.pos.into(),
            });
        } else {
            // Flat: face normals via center_pos for shader reconstruction
            let face_normal = if sin_a.abs() > cos_a.abs() {
                frame.normal * sin_a.signum()
            } else {
                frame.binormal * cos_a.signum()
            };
            vertices.push(BackboneVertex {
                position: pos.into(),
                normal: face_normal.into(),
                color,
                residue_idx,
                center_pos: (pos - face_normal).into(),
            });
        }
    }
}

/// Compute a cross-section offset for end cap vertices.
pub(crate) fn cap_offset(
    frame: &SplinePoint,
    hw: f32,
    ht: f32,
    roundness: f32,
    cross_section_verts: usize,
    k: usize,
) -> Vec3 {
    let angle = (k as f32 / cross_section_verts as f32)
        * std::f32::consts::TAU;
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let rect_x = cos_a.signum() * hw;
    let rect_y = sin_a.signum() * ht;
    let circ_x = cos_a * hw;
    let circ_y = sin_a * ht;

    let x = rect_x + (circ_x - rect_x) * roundness;
    let y = rect_y + (circ_y - rect_y) * roundness;

    frame.binormal * x + frame.normal * y
}
