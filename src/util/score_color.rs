//! Per-residue energy score → RGB color mapping.
//!
//! Two modes:
//! - **Absolute** (`score`): Uses fixed REU thresholds (-4 to +4).
//! - **Relative** (`score_relative`): Normalizes within the current structure
//!   using 5th/95th percentile bounds.
//!
//! Both modes use a [`ColorRamp`](score_color::ColorRamp) to map the normalized
//! value to a color. Default ramp: green (good) → yellow (neutral) → red (bad).

/// Absolute energy thresholds in REU.
const GOOD_THRESHOLD: f64 = -4.0;
const BAD_THRESHOLD: f64 = 4.0;

/// A color ramp defined by N evenly-spaced color stops.
/// `t = 0` maps to the first color (good), `t = 1` maps to the last (bad).
pub struct ColorRamp {
    stops: Vec<[f32; 3]>,
}

impl ColorRamp {
    /// Interpolate the ramp at position `t` in [0, 1].
    pub fn sample(&self, t: f32) -> [f32; 3] {
        let t = t.clamp(0.0, 1.0);
        let n = self.stops.len() - 1;
        let scaled = t * n as f32;
        let idx = (scaled as usize).min(n - 1);
        let frac = scaled - idx as f32;

        let a = &self.stops[idx];
        let b = &self.stops[idx + 1];
        [
            a[0] + (b[0] - a[0]) * frac,
            a[1] + (b[1] - a[1]) * frac,
            a[2] + (b[2] - a[2]) * frac,
        ]
    }
}

impl Default for ColorRamp {
    /// Green → Yellow → Red
    fn default() -> Self {
        Self {
            stops: vec![
                [0.1, 0.8, 0.2],  // green (good)
                [1.0, 0.9, 0.1],  // yellow (neutral)
                [0.9, 0.15, 0.1], // red (bad)
            ],
        }
    }
}

/// Absolute mode: map a per-residue energy (REU) to [0, 1] using fixed
/// thresholds.
fn score_to_t_absolute(score: f64) -> f32 {
    if score <= GOOD_THRESHOLD {
        0.0
    } else if score >= BAD_THRESHOLD {
        1.0
    } else if score <= 0.0 {
        (0.5 * (1.0 - score / GOOD_THRESHOLD)) as f32
    } else {
        (0.5 + 0.5 * score / BAD_THRESHOLD) as f32
    }
}

/// Absolute per-residue score colors using the default color ramp.
pub fn per_residue_score_colors(scores: &[f64]) -> Vec<[f32; 3]> {
    per_residue_score_colors_with_ramp(scores, &ColorRamp::default())
}

/// Absolute per-residue score colors using a custom color ramp.
pub fn per_residue_score_colors_with_ramp(
    scores: &[f64],
    ramp: &ColorRamp,
) -> Vec<[f32; 3]> {
    scores
        .iter()
        .map(|&s| ramp.sample(score_to_t_absolute(s)))
        .collect()
}

/// Relative per-residue score colors using the default color ramp.
/// Normalizes to the 5th/95th percentile range within the given scores.
pub fn per_residue_score_colors_relative(scores: &[f64]) -> Vec<[f32; 3]> {
    per_residue_score_colors_relative_with_ramp(scores, &ColorRamp::default())
}

/// Relative per-residue score colors using a custom color ramp.
pub fn per_residue_score_colors_relative_with_ramp(
    scores: &[f64],
    ramp: &ColorRamp,
) -> Vec<[f32; 3]> {
    if scores.is_empty() {
        return Vec::new();
    }

    let mut sorted: Vec<f64> = scores.to_vec();
    sorted
        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lo_idx = (sorted.len() as f64 * 0.05) as usize;
    let hi_idx = ((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1);
    let min_score = sorted[lo_idx];
    let max_score = sorted[hi_idx];
    let range = max_score - min_score;

    scores
        .iter()
        .map(|&score| {
            let t = if range.abs() < 1e-6 {
                0.5
            } else {
                ((score - min_score) / range).clamp(0.0, 1.0) as f32
            };
            ramp.sample(t)
        })
        .collect()
}
