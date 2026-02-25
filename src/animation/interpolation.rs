//! Centralized interpolation utilities for animation.

/// Per-frame interpolation context computed once from behavior + raw progress,
/// then shared across all interpolation to prevent backbone/sidechain desync.
#[derive(Debug, Clone, Copy)]
pub struct InterpolationContext {
    /// Raw progress (0.0 to 1.0), unmodified from animation timer.
    pub raw_t: f32,
    /// Eased progress for smooth animations. This is the primary value
    /// that all interpolation should use unless behavior-specific.
    pub eased_t: f32,
    /// For multi-phase behaviors: progress within current phase (0.0 to 1.0).
    pub phase_t: Option<f32>,
    /// For multi-phase behaviors: eased progress within current phase.
    pub phase_eased_t: Option<f32>,
}

impl InterpolationContext {
    /// Context with just raw and eased values (no phase info).
    pub fn simple(raw_t: f32, eased_t: f32) -> Self {
        Self {
            raw_t,
            eased_t,
            phase_t: None,
            phase_eased_t: None,
        }
    }

    /// Context with phase information (for CollapseExpand, etc).
    pub fn with_phase(
        raw_t: f32,
        eased_t: f32,
        phase_t: f32,
        phase_eased_t: f32,
    ) -> Self {
        Self {
            raw_t,
            eased_t,
            phase_t: Some(phase_t),
            phase_eased_t: Some(phase_eased_t),
        }
    }

    /// Animation complete (t=1.0). Used when no animation is running.
    pub fn identity() -> Self {
        Self {
            raw_t: 1.0,
            eased_t: 1.0,
            phase_t: None,
            phase_eased_t: None,
        }
    }

    /// Linear context (no easing).
    pub fn linear(raw_t: f32) -> Self {
        Self::simple(raw_t, raw_t)
    }

    /// Unified progress value for interpolation (`eased_t` for most behaviors).
    #[inline]
    pub fn unified_t(&self) -> f32 {
        self.eased_t
    }
}

impl Default for InterpolationContext {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_context() {
        let ctx = InterpolationContext::identity();
        assert_eq!(ctx.raw_t, 1.0);
        assert_eq!(ctx.eased_t, 1.0);
        assert_eq!(ctx.unified_t(), 1.0);
    }

    #[test]
    fn test_simple_context() {
        let ctx = InterpolationContext::simple(0.5, 0.7);
        assert_eq!(ctx.raw_t, 0.5);
        assert_eq!(ctx.eased_t, 0.7);
        assert_eq!(ctx.unified_t(), 0.7);
    }

    #[test]
    fn test_with_phase() {
        let ctx = InterpolationContext::with_phase(0.3, 0.4, 0.6, 0.8);
        assert_eq!(ctx.raw_t, 0.3);
        assert_eq!(ctx.eased_t, 0.4);
        assert_eq!(ctx.phase_t, Some(0.6));
        assert_eq!(ctx.phase_eased_t, Some(0.8));
    }
}
