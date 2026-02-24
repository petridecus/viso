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
/// Capsule sidechain renderer.
pub mod capsule_sidechain;
/// Bind groups shared across all molecular draw calls.
pub mod draw_context;
/// Shared indexed-mesh draw-pass abstraction.
pub(crate) mod mesh_pass;
/// Nucleic acid ring + stem renderer.
pub mod nucleic_acid;
/// Reusable impostor-pass primitives and instance types.
pub mod primitives;
/// Interactive pull arrow renderer.
pub mod pull;

use draw_context::DrawBindGroups;

/// Trait shared by all molecular renderers.
///
/// Every molecular renderer (backbone, sidechains, bands, pulls,
/// ball-and-stick, nucleic acids) uses the same draw signature. The trait
/// serves as documentation and enables future refactoring (e.g. iterating
/// a renderer list). No dynamic dispatch is used today.
pub trait MolecularRenderer {
    /// Draw this renderer's geometry into the given render pass.
    fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &DrawBindGroups<'a>,
    );
}
