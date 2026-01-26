//! Secondary structure detection using Cα geometry
//!
//! Detects alpha helices and beta sheets based on Cα-Cα distances
//! and assigns per-residue classifications.

use glam::Vec3;

/// Secondary structure type for a residue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SSType {
    Helix,
    Sheet,
    Coil,
}

impl SSType {
    /// Get the color for this SS type (RGB, 0-1 range)
    pub fn color(&self) -> [f32; 3] {
        match self {
            SSType::Helix => [0.9, 0.3, 0.5],  // Magenta/pink for helices
            SSType::Sheet => [0.95, 0.85, 0.3], // Yellow for sheets
            SSType::Coil => [0.6, 0.85, 0.6],   // Pale green for coils (original color)
        }
    }
}

/// Detect secondary structure from Cα positions
///
/// Uses distance-based heuristics:
/// - Helix: Cα(i)-Cα(i+3) ≈ 5.0-5.5Å, Cα(i)-Cα(i+4) ≈ 5.5-6.5Å
/// - Sheet: Extended conformation with Cα(i)-Cα(i+2) ≈ 6.0-7.5Å
///
/// Returns a Vec of SSType, one per residue (same length as ca_positions)
pub fn detect_secondary_structure(ca_positions: &[Vec3]) -> Vec<SSType> {
    let n = ca_positions.len();
    if n < 4 {
        return vec![SSType::Coil; n];
    }

    // First pass: compute raw classifications based on local geometry
    let mut raw_ss: Vec<SSType> = vec![SSType::Coil; n];

    for i in 0..n {
        // Check for helix pattern (need i+3 and i+4)
        if i + 4 < n {
            let d_i3 = (ca_positions[i] - ca_positions[i + 3]).length();
            let d_i4 = (ca_positions[i] - ca_positions[i + 4]).length();

            // Helix: Cα(i)-Cα(i+3) ~ 5.0-5.5Å, Cα(i)-Cα(i+4) ~ 5.5-6.5Å
            let is_helix = d_i3 >= 4.5 && d_i3 <= 6.0 && d_i4 >= 5.0 && d_i4 <= 7.0;

            if is_helix {
                // Mark this residue and the next few as potential helix
                raw_ss[i] = SSType::Helix;
            }
        }

        // Check for sheet pattern (extended conformation)
        if i + 2 < n && raw_ss[i] != SSType::Helix {
            let d_i1 = (ca_positions[i] - ca_positions[i + 1]).length();
            let d_i2 = (ca_positions[i] - ca_positions[i + 2]).length();

            // Sheet: Extended, so Cα(i)-Cα(i+1) ~ 3.8Å and Cα(i)-Cα(i+2) ~ 6.5-7.5Å
            let is_extended = d_i1 >= 3.5 && d_i1 <= 4.1 && d_i2 >= 6.0 && d_i2 <= 8.0;

            if is_extended {
                raw_ss[i] = SSType::Sheet;
            }
        }
    }

    // Second pass: smooth assignments (require minimum run length)
    let min_helix_length = 4;
    let min_sheet_length = 3;

    let mut smoothed = vec![SSType::Coil; n];

    // Find and validate helix runs
    let mut i = 0;
    while i < n {
        if raw_ss[i] == SSType::Helix {
            // Count consecutive helix residues
            let start = i;
            while i < n && raw_ss[i] == SSType::Helix {
                i += 1;
            }
            let run_length = i - start;

            // Keep helix if long enough, extend to cover full turn
            if run_length >= min_helix_length {
                // Extend helix assignment to cover the full helical segment
                let end = (start + run_length + 3).min(n);
                for j in start..end {
                    smoothed[j] = SSType::Helix;
                }
            }
        } else {
            i += 1;
        }
    }

    // Find and validate sheet runs (only where not already helix)
    i = 0;
    while i < n {
        if raw_ss[i] == SSType::Sheet && smoothed[i] != SSType::Helix {
            let start = i;
            while i < n && raw_ss[i] == SSType::Sheet && smoothed[i] != SSType::Helix {
                i += 1;
            }
            let run_length = i - start;

            if run_length >= min_sheet_length {
                for j in start..i {
                    smoothed[j] = SSType::Sheet;
                }
            }
        } else {
            i += 1;
        }
    }

    smoothed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_chain() {
        let result = detect_secondary_structure(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_short_chain() {
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)];
        let result = detect_secondary_structure(&positions);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&s| s == SSType::Coil));
    }

    #[test]
    fn test_ss_colors() {
        assert_eq!(SSType::Helix.color(), [0.9, 0.3, 0.5]);
        assert_eq!(SSType::Sheet.color(), [0.95, 0.85, 0.3]);
        assert_eq!(SSType::Coil.color(), [0.6, 0.85, 0.6]);
    }
}
