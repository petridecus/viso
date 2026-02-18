pub mod ball_and_stick;
pub mod band;
pub mod capsule_instance;
pub mod capsule_sidechain;
pub mod draw_context;
pub mod nucleic_acid;
pub mod pull;
pub mod ribbon;
pub mod tube;

use draw_context::DrawBindGroups;

/// Trait shared by all molecular renderers.
///
/// Every molecular renderer (tubes, ribbons, sidechains, bands, pulls,
/// ball-and-stick, nucleic acids) uses the same draw signature. The trait
/// serves as documentation and enables future refactoring (e.g. iterating
/// a renderer list). No dynamic dispatch is used today.
pub trait MolecularRenderer {
    fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &DrawBindGroups<'a>,
    );
}
