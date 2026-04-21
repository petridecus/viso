//! Structural bond renderer (H-bonds, disulfide bonds).
//!
//! Renders structural bonds as thin capsules between atom pairs.
//! Completely independent of the constraint band system — bonds are
//! static structural annotations, not scoring-engine constraints.

use glam::Vec3;
use molex::AtomId;

use crate::engine::viso_state::{EntityPositions, SceneRenderState};
use crate::error::VisoError;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::options::{BondOptions, BondStyle, ColorOptions};
use crate::renderer::impostor::{CapsuleInstance, ImpostorPass, ShaderDef};

/// A single structural bond to be rendered.
#[derive(Debug, Clone)]
pub struct StructuralBond {
    /// World-space position of the first atom.
    pub pos_a: Vec3,
    /// World-space position of the second atom.
    pub pos_b: Vec3,
    /// RGB color.
    pub color: [f32; 3],
    /// Capsule radius in Angstroms.
    pub radius: f32,
    /// Flat residue index (for picking).
    pub residue_idx: u32,
    /// Visual style.
    pub style: BondStyle,
    /// Emissive glow factor (0.0 = no glow, 1.0 = fully self-lit).
    pub emissive: f32,
    /// Opacity (0.0 = fully transparent, 1.0 = fully opaque).
    pub opacity: f32,
}

/// Resolve cross-entity structural bonds from derived scene state into
/// renderable [`StructuralBond`]s.
///
/// Both endpoint lists are `(AtomId, AtomId)` pairs rederived from the
/// `Assembly` at sync time. Positions are resolved through
/// [`EntityPositions`] so the render path reads only derived state and
/// animator output — never `&Assembly` or `&MoleculeEntity`.
pub(crate) fn resolve_structural_bonds(
    scene_state: &SceneRenderState,
    positions: &EntityPositions,
    bond_opts: &BondOptions,
    colors: &ColorOptions,
) -> Vec<StructuralBond> {
    let mut bonds = Vec::new();

    if bond_opts.hydrogen_bonds.visible {
        bonds.extend(resolve_hbonds(
            &scene_state.hbond_endpoints,
            positions,
            bond_opts,
            colors,
        ));
    }

    if bond_opts.disulfide_bonds.visible {
        bonds.extend(resolve_disulfides(
            &scene_state.disulfide_endpoints,
            positions,
            bond_opts,
            colors,
        ));
    }

    bonds
}

/// Resolve an [`AtomId`] to a world-space position via
/// [`EntityPositions`].
fn atom_position(atom: AtomId, positions: &EntityPositions) -> Option<Vec3> {
    positions
        .get(atom.entity)
        .and_then(|slice| slice.get(atom.index as usize).copied())
}

/// Emit an H-bond capsule per `(donor_N, acceptor_C)` endpoint pair.
fn resolve_hbonds(
    endpoints: &[(AtomId, AtomId)],
    positions: &EntityPositions,
    bond_opts: &BondOptions,
    colors: &ColorOptions,
) -> Vec<StructuralBond> {
    let base_color = colors.band_hbond;
    let w = 0.5;
    let color = [
        base_color[0] + (1.0 - base_color[0]) * w,
        base_color[1] + (1.0 - base_color[1]) * w,
        base_color[2] + (1.0 - base_color[2]) * w,
    ];

    endpoints
        .iter()
        .filter_map(|&(donor, acceptor)| {
            let pos_a = atom_position(donor, positions)?;
            let pos_b = atom_position(acceptor, positions)?;
            Some(StructuralBond {
                pos_a,
                pos_b,
                color,
                radius: bond_opts.hydrogen_bonds.radius,
                residue_idx: donor.index,
                style: bond_opts.hydrogen_bonds.style,
                emissive: 0.6,
                opacity: 0.5,
            })
        })
        .collect()
}

/// Emit a disulfide capsule per `(SG, SG)` endpoint pair.
fn resolve_disulfides(
    endpoints: &[(AtomId, AtomId)],
    positions: &EntityPositions,
    bond_opts: &BondOptions,
    colors: &ColorOptions,
) -> Vec<StructuralBond> {
    let base_color = colors.band_disulfide;
    let w = 0.5;
    let color = [
        base_color[0] + (1.0 - base_color[0]) * w,
        base_color[1] + (1.0 - base_color[1]) * w,
        base_color[2] + (1.0 - base_color[2]) * w,
    ];

    endpoints
        .iter()
        .filter_map(|&(a, b)| {
            let pos_a = atom_position(a, positions)?;
            let pos_b = atom_position(b, positions)?;
            Some(StructuralBond {
                pos_a,
                pos_b,
                color,
                radius: bond_opts.disulfide_bonds.radius,
                residue_idx: 0,
                style: bond_opts.disulfide_bonds.style,
                emissive: 0.6,
                opacity: 0.5,
            })
        })
        .collect()
}

/// Renders structural bonds as thin capsules.
pub struct BondRenderer {
    pass: ImpostorPass<CapsuleInstance>,
}

impl BondRenderer {
    /// Create a new bond renderer with an empty instance buffer.
    pub fn new(
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
    pub fn update(
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
    pub fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![self.pass.buffer_info("Structural Bonds")]
    }

    /// Draw bond capsules into the given render pass.
    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &crate::renderer::draw_context::DrawBindGroups<'a>,
    ) {
        self.pass.draw(render_pass, bind_groups);
    }
}
