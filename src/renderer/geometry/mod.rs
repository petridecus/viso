//! Molecular geometry renderers.
//!
//! Each renderer produces GPU-ready vertex/instance data for a specific
//! molecular representation: unified backbone (tubes + ribbons), sidechain
//! capsules, ball-and-stick ligands, nucleic acid rings/stems, constraint
//! bands, and interactive pulls.

/// Unified backbone renderer (protein + nucleic acid).
pub mod backbone;
/// Ball-and-stick renderer for ligands, ions, and waters.
pub mod ball_and_stick;
/// Constraint band renderer (pulls, H-bonds, disulfides).
pub mod band;
/// Nucleic acid ring + stem renderer.
pub mod nucleic_acid;
/// Interactive pull arrow renderer.
pub mod pull;
/// Capsule sidechain renderer.
pub mod sidechain;
