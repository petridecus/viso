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

/// Discrete chain color palette — perceptually distinct, PBR-friendly
/// colors that look good under directional lighting.
///
/// For ≤`CHAIN_PALETTE.len()` chains the palette is indexed directly.
/// Beyond that, falls back to an evenly-spaced HSL sweep.
const CHAIN_PALETTE: &[[f32; 3]] = &[
    [0.05, 0.45, 1.00], // electric blue
    [1.00, 0.35, 0.00], // pure orange
    [0.00, 0.85, 0.20], // neon green
    [1.00, 0.10, 0.25], // fire red
    [0.55, 0.20, 1.00], // violet
    [1.00, 0.85, 0.00], // bright yellow
    [0.00, 0.85, 0.85], // electric cyan
    [1.00, 0.05, 0.60], // hot pink
    [0.40, 0.90, 0.00], // chartreuse
    [0.10, 0.30, 1.00], // deep blue
    [1.00, 0.50, 0.20], // tangerine
    [0.80, 0.00, 1.00], // purple
];

/// Compute a chain color for index `idx` out of `total` chains.
///
/// Cycles through the hand-picked palette.
pub(crate) fn chain_color_indexed(idx: usize, _total: usize) -> [f32; 3] {
    CHAIN_PALETTE[idx % CHAIN_PALETTE.len()]
}

/// Legacy API: positional [0,1] → color (still used by `initial_chain_colors`).
pub(crate) fn chain_color(t: f32) -> [f32; 3] {
    let hue = t * 360.0;
    hsl_to_rgb(hue, 0.6, 0.55)
}

/// Convert HSL (h in degrees, s and l in [0,1]) to linear RGB.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h2 = h / 60.0;
    let x = c * (1.0 - (h2 % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match h2 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    [r1 + m, g1 + m, b1 + m]
}

/// Compute per-residue colors based on backbone color mode.
///
/// Supports chain coloring (rainbow), secondary structure coloring, and
/// score-based coloring (absolute or relative).
///
/// `entity_chain_counts` maps each entity to the number of backbone chains it
/// contributed to the flat `backbone_chains` vec. When present, chain coloring
/// assigns the same color to all backbone segments from the same entity.
pub(crate) fn compute_per_residue_colors(
    backbone_chains: &[Vec<glam::Vec3>],
    ss_types: &[molex::secondary_structure::SSType],
    per_residue_scores: &[Option<&[f64]>],
    color_mode: &super::BackboneColorMode,
    entity_chain_counts: Option<&[usize]>,
) -> Vec<[f32; 3]> {
    use molex::secondary_structure::SSType;

    let residue_count = ss_types.len().max(1);
    match color_mode {
        super::BackboneColorMode::Score
        | super::BackboneColorMode::ScoreRelative => {
            let mut all_scores: Vec<f64> = Vec::new();
            for &s in per_residue_scores.iter().flatten() {
                all_scores.extend_from_slice(s);
            }
            if all_scores.is_empty() {
                return vec![[0.5, 0.5, 0.5]; residue_count];
            }
            match color_mode {
                super::BackboneColorMode::Score => {
                    per_residue_score_colors(&all_scores)
                }
                _ => per_residue_score_colors_relative(&all_scores),
            }
        }
        super::BackboneColorMode::SecondaryStructure => {
            if ss_types.is_empty() {
                vec![[0.5, 0.5, 0.5]; residue_count]
            } else {
                ss_types.iter().map(SSType::color).collect()
            }
        }
        super::BackboneColorMode::Chain => per_chain_colors(
            backbone_chains,
            entity_chain_counts,
            residue_count,
        ),
    }
}

/// Assign chain colors, grouping backbone segments by entity when available.
fn per_chain_colors(
    backbone_chains: &[Vec<glam::Vec3>],
    entity_chain_counts: Option<&[usize]>,
    residue_count: usize,
) -> Vec<[f32; 3]> {
    if backbone_chains.is_empty() {
        return vec![[0.5, 0.5, 0.5]; residue_count];
    }

    // When entity grouping is available, all backbone segments from
    // the same entity share one color.
    if let Some(counts) = entity_chain_counts {
        let mut chain_to_entity = Vec::with_capacity(backbone_chains.len());
        for (entity_idx, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                chain_to_entity.push(entity_idx);
            }
        }
        let n_entities = counts.len();
        let mut colors = Vec::with_capacity(residue_count);
        for (chain_idx, chain) in backbone_chains.iter().enumerate() {
            let eidx = chain_to_entity.get(chain_idx).copied().unwrap_or(0);
            let color = chain_color_indexed(eidx, n_entities);
            let n_residues = chain.len() / 3;
            colors.extend(std::iter::repeat_n(color, n_residues));
        }
        return colors;
    }

    // Fallback: one color per backbone chain
    let num_chains = backbone_chains.len();
    let mut colors = Vec::with_capacity(residue_count);
    for (chain_idx, chain) in backbone_chains.iter().enumerate() {
        let color = chain_color_indexed(chain_idx, num_chains);
        let n_residues = chain.len() / 3;
        colors.extend(std::iter::repeat_n(color, n_residues));
    }
    colors
}
