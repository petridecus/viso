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
/// Structural bond renderer (H-bonds, disulfide bonds).
pub mod bond;
/// Isosurface mesh renderer (electron density maps).
pub(crate) mod isosurface;
/// Nucleic acid ring + stem renderer.
pub mod nucleic_acid;
/// Interactive pull arrow renderer.
pub mod pull;
/// Sheet-surface sidechain position adjustment.
pub(crate) mod sheet_adjust;
/// Capsule sidechain renderer.
pub mod sidechain;

pub use backbone::{BackboneRenderer, ChainPair, PreparedBackboneData};
pub use ball_and_stick::{BallAndStickRenderer, PreparedBallAndStickData};
pub use band::BandRenderer;
pub use bond::BondRenderer;
pub use nucleic_acid::NucleicAcidRenderer;
pub use pull::PullRenderer;
pub use sidechain::{SidechainRenderer, SidechainView};
