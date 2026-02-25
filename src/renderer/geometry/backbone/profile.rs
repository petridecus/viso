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
            thickness: self.thickness + (other.thickness - self.thickness) * t,
            roundness: self.roundness + (other.roundness - self.roundness) * t,
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
        SSType::Coil => {
            (geo.coil_width, geo.coil_thickness, geo.coil_roundness, 0.0)
        }
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
    profile: &CrossSectionProfile,
    cross_section_verts: usize,
    vertices: &mut Vec<BackboneVertex>,
) {
    let hw = profile.width * 0.5;
    let ht = profile.thickness * 0.5;
    let roundness = profile.roundness;
    let color = profile.color;
    let residue_idx = profile.residue_idx;
    for k in 0..cross_section_verts {
        let angle =
            (k as f32 / cross_section_verts as f32) * std::f32::consts::TAU;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Rectangular corner position
        let rect_x = cos_a.signum() * hw;
        let rect_y = sin_a.signum() * ht;

        // Elliptical position
        let circ_x = cos_a * hw;
        let circ_y = sin_a * ht;

        // Blend between rectangular and elliptical
        let x = rect_x + (circ_x - rect_x) * roundness;
        let y = rect_y + (circ_y - rect_y) * roundness;

        let offset = frame.binormal * x + frame.normal * y;
        let pos = frame.pos + offset;

        // Surface normal from the elliptical gradient: (cos/hw, sin/ht).
        let grad = frame.binormal * (cos_a / hw) + frame.normal * (sin_a / ht);
        let mut surface_normal = grad.normalize_or_zero();

        // Ensure the normal points outward (same hemisphere as offset).
        if surface_normal.dot(offset) < 0.0 {
            surface_normal = -surface_normal;
        }

        let r = offset.length();

        vertices.push(BackboneVertex {
            position: pos.into(),
            normal: surface_normal.into(),
            color,
            residue_idx,
            center_pos: (pos - surface_normal * r).into(),
        });
    }
}

/// Compute a cross-section offset for end cap vertices.
pub(crate) fn cap_offset(
    frame: &SplinePoint,
    profile: &CrossSectionProfile,
    cross_section_verts: usize,
    k: usize,
) -> Vec3 {
    let hw = profile.width * 0.5;
    let ht = profile.thickness * 0.5;
    let roundness = profile.roundness;
    let angle = (k as f32 / cross_section_verts as f32) * std::f32::consts::TAU;
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

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    fn make_frame(pos: Vec3, tangent: Vec3, normal: Vec3) -> SplinePoint {
        let binormal = tangent.cross(normal).normalize();
        SplinePoint {
            pos,
            tangent,
            normal,
            binormal,
        }
    }

    /// Circular tube: verify normals are radial and center_pos == frame.pos.
    #[test]
    fn circular_tube_normals_are_radial() {
        let frame = make_frame(Vec3::ZERO, Vec3::Z, Vec3::Y);
        let hw = 0.2_f32;
        let csv = 8;
        let profile = CrossSectionProfile {
            width: hw * 2.0,
            thickness: hw * 2.0,
            roundness: 1.0,
            radial_blend: 0.0,
            color: [1.0, 0.0, 0.0],
            residue_idx: 0,
        };
        let mut verts = Vec::new();
        extrude_cross_section(&frame, &profile, csv, &mut verts);
        assert_eq!(verts.len(), csv);

        for (k, v) in verts.iter().enumerate() {
            let pos = Vec3::from(v.position);
            let nrm = Vec3::from(v.normal);
            let cp = Vec3::from(v.center_pos);

            // Position should be on a circle of radius hw in XY plane
            let dist = pos.length();
            assert!(
                (dist - hw).abs() < 1e-5,
                "k={k}: position dist {dist} != hw {hw}",
            );

            // Normal should be the radial direction (unit vector from origin to
            // pos)
            let expected_normal = pos.normalize();
            let dot = nrm.dot(expected_normal);
            assert!(
                dot > 0.9999,
                "k={k}: normal {nrm:?} not radial (dot with expected \
                 {expected_normal:?} = {dot})",
            );

            // center_pos should equal frame.pos (origin) for circles
            assert!(
                cp.length() < 1e-5,
                "k={k}: center_pos {cp:?} should be ~origin for circle",
            );
        }
    }

    /// Elliptical cross-section: verify gradient normals differ from radial.
    #[test]
    fn elliptical_normals_differ_from_radial() {
        let frame = make_frame(Vec3::ZERO, Vec3::Z, Vec3::Y);
        let csv = 8;
        let profile = CrossSectionProfile {
            width: 0.8,
            thickness: 0.2,
            roundness: 1.0,
            radial_blend: 0.0,
            color: [1.0, 0.0, 0.0],
            residue_idx: 0,
        };
        let mut verts = Vec::new();
        extrude_cross_section(&frame, &profile, csv, &mut verts);

        // At 45 degrees, elliptical gradient normal should differ from radial
        let v = &verts[1]; // k=1, angle=π/4
        let pos = Vec3::from(v.position);
        let nrm = Vec3::from(v.normal);
        let radial = pos.normalize();

        // For hw >> ht, the gradient normal tilts toward the short axis
        let dot = nrm.dot(radial);
        assert!(
            dot < 0.999,
            "ellipse normal should differ from radial (dot={dot})",
        );

        // Verify it's a unit vector
        assert!(
            (nrm.length() - 1.0).abs() < 1e-5,
            "normal should be unit: len={}",
            nrm.length(),
        );

        // Verify center_pos reconstruction gives same normal as stored
        let cp = Vec3::from(v.center_pos);
        let reconstructed = (pos - cp).normalize();
        let recon_dot = reconstructed.dot(nrm);
        assert!(
            recon_dot > 0.999,
            "center_pos reconstruction should match stored normal \
             (dot={recon_dot}, stored={nrm:?}, \
             reconstructed={reconstructed:?})",
        );
    }

    /// Verify center_pos reconstruction matches stored normal for all vertices.
    #[test]
    fn center_pos_reconstruction_matches_stored_normal() {
        let frame = make_frame(
            Vec3::new(5.0, 3.0, -2.0),
            Vec3::new(0.0, 0.3, 0.95).normalize(),
            Vec3::Y,
        );
        // Re-orthogonalize normal to tangent
        let t = frame.tangent;
        let n = (Vec3::Y - t * t.dot(Vec3::Y)).normalize();
        let b = t.cross(n).normalize();
        let frame = SplinePoint {
            pos: frame.pos,
            tangent: t,
            normal: n,
            binormal: b,
        };

        for &(hw, ht, roundness) in &[
            (0.2, 0.2, 1.0), // circle
            (0.4, 0.1, 1.0), // wide ellipse
            (0.1, 0.3, 1.0), // tall ellipse
            (0.2, 0.2, 0.5), // half-round circle
        ] {
            let profile = CrossSectionProfile {
                width: hw * 2.0,
                thickness: ht * 2.0,
                roundness,
                radial_blend: 0.0,
                color: [1.0, 0.0, 0.0],
                residue_idx: 0,
            };
            let mut verts = Vec::new();
            extrude_cross_section(&frame, &profile, 8, &mut verts);

            for (k, v) in verts.iter().enumerate() {
                let pos = Vec3::from(v.position);
                let nrm = Vec3::from(v.normal);
                let cp = Vec3::from(v.center_pos);

                let diff = pos - cp;
                if diff.length() < 1e-6 {
                    continue; // degenerate case
                }
                let reconstructed = diff.normalize();
                let dot = reconstructed.dot(nrm);
                assert!(
                    dot > 0.999,
                    "hw={hw} ht={ht} r={roundness} k={k}: reconstruction \
                     mismatch (dot={dot:.6}, stored={nrm:?}, \
                     reconstructed={reconstructed:?})",
                );
            }
        }
    }
}
