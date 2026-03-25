//! Structural bond renderer (H-bonds, disulfide bonds).
//!
//! Renders structural bonds as thin capsules between atom pairs.
//! Completely independent of the constraint band system — bonds are
//! static structural annotations, not scoring-engine constraints.

use glam::Vec3;

use crate::error::VisoError;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::options::{BondOptions, BondStyle, ColorOptions};
use crate::renderer::geometry::backbone::spline::project_backbone_atoms;
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

/// Resolve detected bonds (from molex) into renderable [`StructuralBond`]s.
///
/// H-bond endpoints are projected onto the backbone Catmull-Rom spline
/// via [`project_backbone_atoms`] so they sit on the rendered ribbon.
/// Disulfide endpoints use raw SG atom positions (sidechain atoms are
/// not spline-smoothed).
pub(crate) fn resolve_structural_bonds(
    hbonds: &[molex::HBond],
    disulfides: &[molex::DisulfideBond],
    backbone_chains: &[Vec<Vec3>],
    atoms: &[molex::Atom],
    bond_opts: &BondOptions,
    colors: &ColorOptions,
) -> Vec<StructuralBond> {
    let mut bonds = Vec::new();

    if bond_opts.hydrogen_bonds.visible {
        bonds.extend(resolve_hbonds(
            hbonds,
            backbone_chains,
            bond_opts,
            colors,
        ));
    }

    if bond_opts.disulfide_bonds.visible {
        bonds.extend(resolve_disulfides(disulfides, atoms, bond_opts, colors));
    }

    bonds
}

/// Map detected H-bonds to spline-projected backbone positions.
fn resolve_hbonds(
    hbonds: &[molex::HBond],
    backbone_chains: &[Vec<Vec3>],
    bond_opts: &BondOptions,
    colors: &ColorOptions,
) -> Vec<StructuralBond> {
    let (n_positions, c_positions) = project_backbone_atoms(backbone_chains);

    let base_color = colors.band_hbond;
    let w = 0.5;
    let color = [
        base_color[0] + (1.0 - base_color[0]) * w,
        base_color[1] + (1.0 - base_color[1]) * w,
        base_color[2] + (1.0 - base_color[2]) * w,
    ];

    hbonds
        .iter()
        .filter_map(|hb| {
            // Donor N-H → acceptor C=O: render from donor's N to
            // acceptor's C projected onto the backbone spline.
            let pos_a = n_positions.get(hb.donor).copied()?;
            let pos_b = c_positions.get(hb.acceptor).copied()?;
            Some(StructuralBond {
                pos_a,
                pos_b,
                color,
                radius: bond_opts.hydrogen_bonds.radius,
                residue_idx: hb.donor as u32,
                style: bond_opts.hydrogen_bonds.style,
                emissive: 0.6,
                opacity: 0.5,
            })
        })
        .collect()
}

/// Map detected disulfide bonds to SG atom positions.
fn resolve_disulfides(
    disulfides: &[molex::DisulfideBond],
    atoms: &[molex::Atom],
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

    disulfides
        .iter()
        .filter_map(|ds| {
            let pos_a = atoms.get(ds.sg_a).map(|a| a.position)?;
            let pos_b = atoms.get(ds.sg_b).map(|a| a.position)?;
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
