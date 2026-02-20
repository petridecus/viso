//! Molecular geometry renderers.
//!
//! Each renderer produces GPU-ready vertex/instance data for a specific
//! molecular representation: backbone tubes, secondary-structure ribbons,
//! sidechain capsules, ball-and-stick ligands, nucleic acid backbones,
//! constraint bands, and interactive pulls.

/// Ball-and-stick renderer for ligands, ions, and waters.
pub mod ball_and_stick;
/// Constraint band renderer (pulls, H-bonds, disulfides).
pub mod band;
/// Shared capsule instance layout for impostor rendering.
pub mod capsule_instance;
/// Capsule sidechain renderer.
pub mod capsule_sidechain;
/// Bind groups shared across all molecular draw calls.
pub mod draw_context;
/// Nucleic acid backbone ribbon renderer.
pub mod nucleic_acid;
/// Interactive pull arrow renderer.
pub mod pull;
/// Secondary-structure ribbon renderer.
pub mod ribbon;
/// Backbone tube renderer.
pub mod tube;

use draw_context::DrawBindGroups;

/// Trait shared by all molecular renderers.
///
/// Every molecular renderer (tubes, ribbons, sidechains, bands, pulls,
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
