use glam::Vec3;

/// State for animating a single residue between two conformations.
#[derive(Debug, Clone)]
pub struct ResidueAnimationState {
    pub residue_idx: usize,
    pub start_backbone: [Vec3; 3], // N, CA, C
    pub end_backbone: [Vec3; 3],
    pub start_chis: [f32; 4],
    pub end_chis: [f32; 4],
    pub num_chis: usize,
    pub needs_animation: bool,
}

/// Result of interpolating a residue at a specific time t.
#[derive(Debug, Clone, Copy)]
pub struct InterpolatedResidue {
    pub backbone: [Vec3; 3],
    pub chis: [f32; 4],
    pub num_chis: usize,
    pub residue_idx: usize,
}

impl ResidueAnimationState {
    /// Create a new residue animation state from start and end conformations.
    pub fn new(
        residue_idx: usize,
        start_backbone: [Vec3; 3],
        end_backbone: [Vec3; 3],
        start_chis: &[f32],
        end_chis: &[f32],
    ) -> Self {
        let num_chis = start_chis.len().min(end_chis.len()).min(4);

        let mut start_chis_arr = [0.0f32; 4];
        let mut end_chis_arr = [0.0f32; 4];

        for i in 0..num_chis {
            start_chis_arr[i] = start_chis[i];
            end_chis_arr[i] = end_chis[i];
        }

        Self {
            residue_idx,
            start_backbone,
            end_backbone,
            start_chis: start_chis_arr,
            end_chis: end_chis_arr,
            num_chis,
            needs_animation: true, // Will be validated by check_needs_animation
        }
    }

    /// Interpolate the residue state at time t (0.0 to 1.0).
    /// Target: <1μs
    #[inline]
    pub fn interpolate(&self, t: f32) -> InterpolatedResidue {
        // Clamp t to valid range
        let t = t.clamp(0.0, 1.0);

        // Interpolate backbone positions
        let backbone = [
            lerp_vec3(t, self.start_backbone[0], self.end_backbone[0]),
            lerp_vec3(t, self.start_backbone[1], self.end_backbone[1]),
            lerp_vec3(t, self.start_backbone[2], self.end_backbone[2]),
        ];

        // Interpolate chi angles with proper angle wrapping
        let mut chis = [0.0f32; 4];
        for i in 0..self.num_chis {
            chis[i] = lerp_angle(t, self.start_chis[i], self.end_chis[i]);
        }

        InterpolatedResidue {
            backbone,
            chis,
            num_chis: self.num_chis,
            residue_idx: self.residue_idx,
        }
    }

    /// Check if this residue actually needs animation based on the difference
    /// between start and end states. Sets `needs_animation` flag.
    pub fn check_needs_animation(&mut self, epsilon: f32) {
        // Check backbone positions
        for i in 0..3 {
            let diff = (self.end_backbone[i] - self.start_backbone[i]).length();
            if diff > epsilon {
                self.needs_animation = true;
                return;
            }
        }

        // Check chi angles
        for i in 0..self.num_chis {
            let adjusted_start = fix_angle(self.start_chis[i], self.end_chis[i]);
            let diff = (self.end_chis[i] - adjusted_start).abs();
            if diff > epsilon {
                self.needs_animation = true;
                return;
            }
        }

        self.needs_animation = false;
    }
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

/// Linear interpolation between two Vec3 positions.
#[inline]
pub fn lerp_vec3(t: f32, start: Vec3, end: Vec3) -> Vec3 {
    start + (end - start) * t
}

/// Linear interpolation between two angles with proper wrapping.
/// Uses fix_angle to ensure the shortest path is taken.
#[inline]
pub fn lerp_angle(t: f32, start: f32, end: f32) -> f32 {
    let adjusted_start = fix_angle(start, end);
    adjusted_start + (end - adjusted_start) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    // =========================================================================
    // fix_angle tests
    // =========================================================================

    #[test]
    fn test_fix_angle_no_wrap() {
        // Angles already close, no wrapping needed
        let result = fix_angle(45.0, 50.0);
        assert!((result - 45.0).abs() < EPSILON, "Expected 45.0, got {}", result);

        let result = fix_angle(-30.0, -20.0);
        assert!((result - (-30.0)).abs() < EPSILON, "Expected -30.0, got {}", result);
    }

    #[test]
    fn test_fix_angle_positive_wrap() {
        // Initial angle is more than 180 degrees above final, needs to wrap down
        let result = fix_angle(350.0, 10.0);
        // 350 should wrap to -10 to be within 180 of 10
        assert!(
            (result - (-10.0)).abs() < EPSILON,
            "Expected -10.0, got {}",
            result
        );

        let result = fix_angle(270.0, 0.0);
        // 270 should wrap to -90 to be within 180 of 0
        assert!(
            (result - (-90.0)).abs() < EPSILON,
            "Expected -90.0, got {}",
            result
        );
    }

    #[test]
    fn test_fix_angle_negative_wrap() {
        // Initial angle is more than 180 degrees below final, needs to wrap up
        let result = fix_angle(-170.0, 170.0);
        // -170 should wrap to 190 to be within 180 of 170
        assert!(
            (result - 190.0).abs() < EPSILON,
            "Expected 190.0, got {}",
            result
        );

        let result = fix_angle(-90.0, 180.0);
        // -90 should wrap to 270 to be within 180 of 180
        assert!(
            (result - 270.0).abs() < EPSILON,
            "Expected 270.0, got {}",
            result
        );
    }

    #[test]
    fn test_fix_angle_boundary() {
        // Exactly 180 degrees apart
        let result = fix_angle(0.0, 180.0);
        assert!((result - 0.0).abs() < EPSILON, "Expected 0.0, got {}", result);

        let result = fix_angle(180.0, 0.0);
        assert!(
            (result - 180.0).abs() < EPSILON,
            "Expected 180.0, got {}",
            result
        );
    }

    // =========================================================================
    // lerp_vec3 tests
    // =========================================================================

    #[test]
    fn test_lerp_vec3_endpoints() {
        let start = Vec3::new(0.0, 0.0, 0.0);
        let end = Vec3::new(10.0, 20.0, 30.0);

        let result = lerp_vec3(0.0, start, end);
        assert!(
            (result - start).length() < EPSILON,
            "At t=0, expected start position"
        );

        let result = lerp_vec3(1.0, start, end);
        assert!(
            (result - end).length() < EPSILON,
            "At t=1, expected end position"
        );
    }

    #[test]
    fn test_lerp_vec3_midpoint() {
        let start = Vec3::new(0.0, 0.0, 0.0);
        let end = Vec3::new(10.0, 20.0, 30.0);
        let expected = Vec3::new(5.0, 10.0, 15.0);

        let result = lerp_vec3(0.5, start, end);
        assert!(
            (result - expected).length() < EPSILON,
            "At t=0.5, expected midpoint {:?}, got {:?}",
            expected,
            result
        );
    }

    #[test]
    fn test_lerp_vec3_quarter() {
        let start = Vec3::new(0.0, 0.0, 0.0);
        let end = Vec3::new(100.0, 200.0, 400.0);
        let expected = Vec3::new(25.0, 50.0, 100.0);

        let result = lerp_vec3(0.25, start, end);
        assert!(
            (result - expected).length() < EPSILON,
            "At t=0.25, expected {:?}, got {:?}",
            expected,
            result
        );
    }

    #[test]
    fn test_lerp_vec3_accuracy() {
        // Test with negative values and precision
        let start = Vec3::new(-5.5, 3.3, -1.1);
        let end = Vec3::new(4.5, -6.7, 8.9);

        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let result = lerp_vec3(t, start, end);
            let expected = start + (end - start) * t;
            assert!(
                (result - expected).length() < EPSILON,
                "At t={}, expected {:?}, got {:?}",
                t,
                expected,
                result
            );
        }
    }

    // =========================================================================
    // lerp_angle tests
    // =========================================================================

    #[test]
    fn test_lerp_angle_simple() {
        let result = lerp_angle(0.5, 0.0, 90.0);
        assert!(
            (result - 45.0).abs() < EPSILON,
            "Expected 45.0, got {}",
            result
        );
    }

    #[test]
    fn test_lerp_angle_wrap_around() {
        // Should take the short path from 350 to 10 (going through 0)
        let result = lerp_angle(0.5, 350.0, 10.0);
        // 350 wraps to -10, midpoint between -10 and 10 is 0
        assert!((result - 0.0).abs() < EPSILON, "Expected 0.0, got {}", result);
    }

    // =========================================================================
    // ResidueAnimationState tests
    // =========================================================================

    #[test]
    fn test_residue_interpolation_endpoints() {
        let start_backbone = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let end_backbone = [
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(1.0, 10.0, 0.0),
            Vec3::new(2.0, 10.0, 0.0),
        ];

        // Use angles that don't require wrapping for clear endpoint testing
        let state = ResidueAnimationState::new(
            0,
            start_backbone,
            end_backbone,
            &[60.0, 90.0],
            &[120.0, 180.0],
        );

        // At t=0, should be at start
        let result = state.interpolate(0.0);
        for i in 0..3 {
            assert!(
                (result.backbone[i] - start_backbone[i]).length() < EPSILON,
                "At t=0, backbone[{}] should be at start",
                i
            );
        }
        assert!((result.chis[0] - 60.0).abs() < EPSILON);
        assert!((result.chis[1] - 90.0).abs() < EPSILON);

        // At t=1, should be at end
        let result = state.interpolate(1.0);
        for i in 0..3 {
            assert!(
                (result.backbone[i] - end_backbone[i]).length() < EPSILON,
                "At t=1, backbone[{}] should be at end",
                i
            );
        }
        assert!((result.chis[0] - 120.0).abs() < EPSILON);
        assert!((result.chis[1] - 180.0).abs() < EPSILON);
    }

    #[test]
    fn test_residue_interpolation_midpoint() {
        let start_backbone = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let end_backbone = [
            Vec3::new(10.0, 10.0, 10.0),
            Vec3::new(10.0, 10.0, 10.0),
            Vec3::new(10.0, 10.0, 10.0),
        ];

        let state =
            ResidueAnimationState::new(0, start_backbone, end_backbone, &[0.0], &[90.0]);

        let result = state.interpolate(0.5);
        let expected_pos = Vec3::new(5.0, 5.0, 5.0);

        for i in 0..3 {
            assert!(
                (result.backbone[i] - expected_pos).length() < EPSILON,
                "At t=0.5, backbone[{}] should be at midpoint",
                i
            );
        }
        assert!((result.chis[0] - 45.0).abs() < EPSILON);
    }

    #[test]
    fn test_residue_clamps_t() {
        let start_backbone = [Vec3::ZERO; 3];
        let end_backbone = [Vec3::ONE; 3];

        let state = ResidueAnimationState::new(0, start_backbone, end_backbone, &[], &[]);

        // t < 0 should clamp to 0
        let result = state.interpolate(-0.5);
        assert!(
            (result.backbone[0] - Vec3::ZERO).length() < EPSILON,
            "t < 0 should clamp to start"
        );

        // t > 1 should clamp to 1
        let result = state.interpolate(1.5);
        assert!(
            (result.backbone[0] - Vec3::ONE).length() < EPSILON,
            "t > 1 should clamp to end"
        );
    }

    // =========================================================================
    // needs_animation detection tests
    // =========================================================================

    #[test]
    fn test_needs_animation_no_change() {
        let backbone = [
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        ];

        let mut state = ResidueAnimationState::new(0, backbone, backbone, &[60.0], &[60.0]);
        state.check_needs_animation(0.001);

        assert!(
            !state.needs_animation,
            "Should not need animation when start == end"
        );
    }

    #[test]
    fn test_needs_animation_backbone_change() {
        let start_backbone = [Vec3::ZERO; 3];
        let mut end_backbone = [Vec3::ZERO; 3];
        end_backbone[1] = Vec3::new(0.1, 0.0, 0.0); // Small change in CA

        let mut state =
            ResidueAnimationState::new(0, start_backbone, end_backbone, &[], &[]);
        state.check_needs_animation(0.001);

        assert!(
            state.needs_animation,
            "Should need animation when backbone changes"
        );
    }

    #[test]
    fn test_needs_animation_chi_change() {
        let backbone = [Vec3::ZERO; 3];

        let mut state =
            ResidueAnimationState::new(0, backbone, backbone, &[60.0], &[65.0]);
        state.check_needs_animation(0.001);

        assert!(
            state.needs_animation,
            "Should need animation when chi angles change"
        );
    }

    #[test]
    fn test_needs_animation_within_epsilon() {
        let backbone = [Vec3::ZERO; 3];

        let mut state =
            ResidueAnimationState::new(0, backbone, backbone, &[60.0], &[60.0001]);
        state.check_needs_animation(0.001);

        assert!(
            !state.needs_animation,
            "Should not need animation when change is within epsilon"
        );
    }

    #[test]
    fn test_needs_animation_wrapped_angle() {
        let backbone = [Vec3::ZERO; 3];

        // 350 and -10 are the same angle, should not need animation
        let mut state =
            ResidueAnimationState::new(0, backbone, backbone, &[350.0], &[-10.0]);
        state.check_needs_animation(0.001);

        assert!(
            !state.needs_animation,
            "Should not need animation when wrapped angles are equivalent"
        );
    }
}
