//! Visual state types for animation.

use glam::Vec3;

/// The visual state of a residue at a point in time.
#[derive(Debug, Clone, Copy)]
pub struct ResidueVisualState {
    /// Backbone atom positions: N, CA, C
    pub backbone: [Vec3; 3],
    /// Chi angles (up to 4)
    pub chis: [f32; 4],
    /// Number of valid chi angles
    pub num_chis: usize,
}

impl ResidueVisualState {
    /// Residue state from backbone position and chi angles.
    pub fn new(backbone: [Vec3; 3], chis: [f32; 4], num_chis: usize) -> Self {
        Self {
            backbone,
            chis,
            num_chis,
        }
    }

    /// Create a state with only backbone (no sidechains).
    pub fn backbone_only(backbone: [Vec3; 3]) -> Self {
        Self {
            backbone,
            chis: [0.0; 4],
            num_chis: 0,
        }
    }

    /// Linear interpolation between two states.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let backbone = [
            lerp_vec3(t, self.backbone[0], other.backbone[0]),
            lerp_vec3(t, self.backbone[1], other.backbone[1]),
            lerp_vec3(t, self.backbone[2], other.backbone[2]),
        ];

        let num_chis = self.num_chis.max(other.num_chis);
        let mut chis = [0.0f32; 4];
        for (i, chi) in chis.iter_mut().enumerate().take(num_chis) {
            *chi = lerp_angle(t, self.chis[i], other.chis[i]);
        }

        Self {
            backbone,
            chis,
            num_chis,
        }
    }

    /// Get the CA (alpha carbon) position.
    #[inline]
    pub fn ca_position(&self) -> Vec3 {
        self.backbone[1]
    }
}

/// Linear interpolation between two Vec3 positions.
#[inline]
pub fn lerp_vec3(t: f32, start: Vec3, end: Vec3) -> Vec3 {
    start + (end - start) * t
}

/// Angle wrapping for chi interpolation (from Rosetta).
/// Adjusts the initial angle to be within 180 degrees of the final angle.
#[inline]
pub fn fix_angle(initial: f32, final_angle: f32) -> f32 {
    let mut adjusted = initial;
    while adjusted > final_angle + 180.0 {
        adjusted -= 360.0;
    }
    while adjusted < final_angle - 180.0 {
        adjusted += 360.0;
    }
    adjusted
}

/// Linear interpolation between two angles with proper wrapping.
#[inline]
pub fn lerp_angle(t: f32, start: f32, end: f32) -> f32 {
    let adjusted_start = fix_angle(start, end);
    adjusted_start + (end - adjusted_start) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residue_visual_state_lerp() {
        let a = ResidueVisualState::new(
            [Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)],
            [0.0, 0.0, 0.0, 0.0],
            1,
        );
        let b = ResidueVisualState::new(
            [Vec3::Y, Vec3::new(1.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.0)],
            [90.0, 0.0, 0.0, 0.0],
            1,
        );

        let mid = a.lerp(&b, 0.5);
        assert!((mid.backbone[1].y - 0.5).abs() < 0.001);
        assert!((mid.chis[0] - 45.0).abs() < 0.001);
    }

    #[test]
    fn test_backbone_only() {
        let state = ResidueVisualState::backbone_only([Vec3::ZERO, Vec3::X, Vec3::Y]);
        assert_eq!(state.num_chis, 0);
        assert_eq!(state.chis, [0.0; 4]);
    }

    #[test]
    fn test_lerp_vec3() {
        let a = Vec3::ZERO;
        let b = Vec3::new(10.0, 20.0, 30.0);
        let mid = lerp_vec3(0.5, a, b);
        assert!((mid - Vec3::new(5.0, 10.0, 15.0)).length() < 0.001);
    }

    #[test]
    fn test_lerp_angle_wrap() {
        // Should take short path from 350 to 10 (through 0)
        let result = lerp_angle(0.5, 350.0, 10.0);
        assert!((result - 0.0).abs() < 0.001);
    }
}
