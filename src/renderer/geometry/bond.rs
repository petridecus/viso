//! Structural bond renderer (H-bonds, disulfide bonds).
//!
//! Renders structural bonds as thin capsules between atom pairs.
//! Completely independent of the constraint band system — bonds are
//! static structural annotations, not scoring-engine constraints.

use glam::Vec3;

use crate::error::VisoError;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::options::BondStyle;
use crate::renderer::impostor::{CapsuleInstance, ImpostorPass, ShaderDef};

/// A single structural bond to be rendered.
#[derive(Debug, Clone)]
pub(crate) struct StructuralBond {
    /// World-space position of the first atom.
    pub(crate) pos_a: Vec3,
    /// World-space position of the second atom.
    pub(crate) pos_b: Vec3,
    /// RGB color.
    pub(crate) color: [f32; 3],
    /// Capsule radius in Angstroms.
    pub(crate) radius: f32,
    /// Flat residue index (for picking).
    pub(crate) residue_idx: u32,
    /// Visual style.
    pub(crate) style: BondStyle,
    /// Emissive glow factor (0.0 = no glow, 1.0 = fully self-lit).
    pub(crate) emissive: f32,
    /// Opacity (0.0 = fully transparent, 1.0 = fully opaque).
    pub(crate) opacity: f32,
}

/// Renders structural bonds as thin capsules.
pub(crate) struct BondRenderer {
    pass: ImpostorPass<CapsuleInstance>,
}

impl BondRenderer {
    /// Create a new bond renderer with an empty instance buffer.
    pub(crate) fn new(
        context: &RenderContext,
        layouts: &crate::renderer::PipelineLayouts,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "Structural Bond",
                shader: Shader::Capsule,
            },
            layouts,
            6,
            shader_composer,
        )?;
        Ok(Self { pass })
    }

    /// Upload bond instances to the GPU.
    ///
    /// Returns `true` if the GPU buffer was reallocated (picking bind
    /// groups may need rebuilding).
    pub(crate) fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bonds: &[StructuralBond],
    ) -> bool {
        let instances = Self::generate_instances(bonds);
        self.pass.write_instances(device, queue, &instances)
    }

    /// Generate capsule instances from bond data.
    fn generate_instances(bonds: &[StructuralBond]) -> Vec<CapsuleInstance> {
        let mut instances = Vec::with_capacity(bonds.len());

        for bond in bonds {
            // Skip degenerate bonds.
            if (bond.pos_b - bond.pos_a).length_squared() < 0.000_001 {
                continue;
            }

            match bond.style {
                BondStyle::Solid => {
                    instances.push(Self::capsule(bond, bond.pos_a, bond.pos_b));
                }
                BondStyle::Dashed => {
                    Self::emit_dashed(bond, &mut instances);
                }
                BondStyle::Stippled => {
                    Self::emit_stippled(bond, &mut instances);
                }
            }
        }

        instances
    }

    /// Single solid capsule between two points.
    fn capsule(bond: &StructuralBond, a: Vec3, b: Vec3) -> CapsuleInstance {
        CapsuleInstance {
            endpoint_a: [a.x, a.y, a.z, bond.radius],
            endpoint_b: [b.x, b.y, b.z, bond.residue_idx as f32],
            color_a: [
                bond.color[0],
                bond.color[1],
                bond.color[2],
                bond.emissive,
            ],
            color_b: [
                bond.color[0],
                bond.color[1],
                bond.color[2],
                bond.opacity,
            ],
        }
    }

    /// Emit dashed segments along the bond axis.
    fn emit_dashed(bond: &StructuralBond, out: &mut Vec<CapsuleInstance>) {
        const DASH_FRACTION: f32 = 0.12;
        const GAP_FRACTION: f32 = 0.08;
        let step = DASH_FRACTION + GAP_FRACTION;
        let n_segments = (1.0 / step).ceil() as u32;
        let dir = bond.pos_b - bond.pos_a;

        for i in 0..n_segments {
            let t = i as f32 * step;
            if t >= 1.0 {
                break;
            }
            let t_end = (t + DASH_FRACTION).min(1.0);
            let a = bond.pos_a + dir * t;
            let b = bond.pos_a + dir * t_end;
            out.push(Self::capsule(bond, a, b));
        }
    }

    /// Emit stippled (short dense) segments along the bond axis.
    fn emit_stippled(bond: &StructuralBond, out: &mut Vec<CapsuleInstance>) {
        const DOT_FRACTION: f32 = 0.06;
        const GAP_FRACTION: f32 = 0.04;
        let step = DOT_FRACTION + GAP_FRACTION;
        let n_segments = (1.0 / step).ceil() as u32;
        let dir = bond.pos_b - bond.pos_a;

        for i in 0..n_segments {
            let t = i as f32 * step;
            if t >= 1.0 {
                break;
            }
            let t_end = (t + DOT_FRACTION).min(1.0);
            let a = bond.pos_a + dir * t;
            let b = bond.pos_a + dir * t_end;
            out.push(Self::capsule(bond, a, b));
        }
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub(crate) fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![self.pass.buffer_info("Structural Bonds")]
    }

    /// Draw bond capsules into the given render pass.
    pub(crate) fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &crate::renderer::draw_context::DrawBindGroups<'a>,
    ) {
        self.pass.draw(render_pass, bind_groups);
    }
}
