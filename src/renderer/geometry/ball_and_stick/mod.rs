//! Ball-and-stick renderer for small molecules (ligands, ions, waters).
//!
//! Renders atoms as ray-cast sphere impostors and bonds as capsule impostors.
//! Uses distance-based bond inference from
//! `molex::analysis::bonds::covalent`.

mod instances;

use glam::Vec3;
use molex::MoleculeEntity;

use crate::error::VisoError;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::options::{ColorOptions, DisplayOptions, DrawingMode};
use crate::renderer::impostor::{
    CapsuleInstance, ImpostorPass, ShaderDef, SphereInstance,
};

/// Radius for bond capsules (thinner than protein sidechains)
pub(super) const BOND_RADIUS: f32 = 0.15;

/// Fraction of vdw_radius for ball-and-stick atom spheres
pub(super) const BALL_RADIUS_SCALE: f32 = 0.3;

/// Sphere radius for Stick mode joints (matches sidechain capsule caps).
pub(super) const STICK_SPHERE_RADIUS: f32 = 0.15;

/// Bond radius for Stick mode — matches sidechain capsule radius so
/// stick geometry has the same visual weight as sidechain bonds.
pub(super) const STICK_BOND_RADIUS: f32 = 0.3;

/// Larger spheres for ions
pub(super) const ION_RADIUS_SCALE: f32 = 0.5;

/// Small sphere for water oxygen
pub(super) const WATER_RADIUS: f32 = 0.3;

/// Perpendicular offset for double bond parallel capsules
pub(super) const DOUBLE_BOND_OFFSET: f32 = 0.2;

/// Warm beige/tan carbon tint for lipid molecules.
pub(super) const LIPID_CARBON_TINT: [f32; 3] = [0.76, 0.70, 0.50];

/// Return atom color: if a carbon tint is provided, carbon atoms use it;
/// all other elements (N, O, S, P, etc.) keep standard CPK coloring.
pub(super) fn atom_color(
    elem: molex::Element,
    carbon_tint: Option<[f32; 3]>,
) -> [f32; 3] {
    match (elem, carbon_tint) {
        (molex::Element::C, Some(tint)) => tint,
        _ => elem.cpk_color(),
    }
}

/// Build a map from atom index to sequential residue index for a polymer
/// entity (protein or nucleic acid). Returns `None` for non-polymer entities.
pub(super) fn build_atom_residue_map(
    entity: &MoleculeEntity,
    atom_count: usize,
) -> Option<Vec<usize>> {
    // Collect residue atom ranges from whichever polymer type matches.
    let ranges: Vec<std::ops::Range<usize>> =
        if let Some(protein) = entity.as_protein() {
            protein
                .residues
                .iter()
                .map(|r| r.atom_range.clone())
                .collect()
        } else {
            let na = entity.as_nucleic_acid()?;
            na.residues.iter().map(|r| r.atom_range.clone()).collect()
        };
    let mut map = vec![0usize; atom_count];
    for (seq_idx, range) in ranges.into_iter().enumerate() {
        for atom_idx in range {
            if atom_idx < map.len() {
                map[atom_idx] = seq_idx;
            }
        }
    }
    Some(map)
}

/// Pre-computed instance data for GPU upload.
pub struct PreparedBallAndStickData<'a> {
    /// Raw bytes for sphere instance data.
    pub sphere_bytes: &'a [u8],
    /// Number of sphere instances.
    pub sphere_count: u32,
    /// Raw bytes for bond capsule instance data.
    pub capsule_bytes: &'a [u8],
    /// Number of bond capsule instances.
    pub capsule_count: u32,
}

/// Output buffers for instance generation.
#[derive(Default)]
pub(super) struct InstanceCollector {
    pub(super) spheres: Vec<SphereInstance>,
    pub(super) bonds: Vec<CapsuleInstance>,
}

impl InstanceCollector {
    /// Push a visual bond capsule.
    ///
    /// `endpoints` is `[pos_a, pos_b]`; `colors` is `[color_a, color_b]`.
    pub(super) fn push_bond(
        &mut self,
        endpoints: [[f32; 3]; 2],
        radius: f32,
        colors: [[f32; 3]; 2],
        pick_id: u32,
    ) {
        let [pos_a, pos_b] = endpoints;
        self.bonds.push(CapsuleInstance {
            endpoint_a: [pos_a[0], pos_a[1], pos_a[2], radius],
            endpoint_b: [pos_b[0], pos_b[1], pos_b[2], pick_id as f32],
            color_a: [colors[0][0], colors[0][1], colors[0][2], 0.0],
            color_b: [colors[1][0], colors[1][1], colors[1][2], 0.0],
        });
    }
}

/// Find any vector perpendicular to the given vector.
pub(super) fn find_perpendicular(v: Vec3) -> Vec3 {
    if v.length_squared() < 1e-8 {
        return Vec3::X;
    }
    let candidate = if v.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    v.cross(candidate).normalize()
}

/// Renders small molecules as ray-cast sphere + capsule impostors.
pub struct BallAndStickRenderer {
    sphere_pass: ImpostorPass<SphereInstance>,
    bond_pass: ImpostorPass<CapsuleInstance>,
}

impl BallAndStickRenderer {
    /// Create a new ball-and-stick renderer with empty buffers.
    pub fn new(
        context: &RenderContext,
        layouts: &crate::renderer::PipelineLayouts,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let sphere_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "BnS Sphere",
                shader: Shader::Sphere,
            },
            layouts,
            6,
            shader_composer,
        )?;

        let bond_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "BnS Bond",
                shader: Shader::Capsule,
            },
            layouts,
            6,
            shader_composer,
        )?;

        Ok(Self {
            sphere_pass,
            bond_pass,
        })
    }

    /// Generate all instances from entity data (pure CPU, no GPU access
    /// needed).
    ///
    /// `pick_id_offset` is added to all per-atom pick IDs so that IDs are
    /// globally unique when multiple entities are concatenated. Pass `0` for
    /// the live-update path (the scene processor concatenation applies offsets
    /// at the byte level instead).
    ///
    /// `mode` controls Stick vs BallAndStick rendering for
    /// protein/NA entities that were routed through this pipeline via a
    /// drawing mode override. For non-protein entities this is ignored.
    ///
    /// `per_residue_colors`, when provided, overrides CPK element
    /// coloring for polymer entities (Stick/BnS) — each atom is colored
    /// by its residue's backbone color (chain, SS, score, etc.).
    ///
    /// Returns (sphere_instances, bond_instances).
    pub(crate) fn generate_all_instances(
        entities: &[MoleculeEntity],
        display: &DisplayOptions,
        colors: Option<&ColorOptions>,
        pick_id_offset: u32,
        mode: DrawingMode,
        per_residue_colors: Option<&[[f32; 3]]>,
    ) -> (Vec<SphereInstance>, Vec<CapsuleInstance>) {
        let mut out = InstanceCollector::default();
        let mut atom_offset = pick_id_offset;

        for entity in entities {
            let entity_atom_count = entity.atom_count() as u32;
            Self::generate_entity_instances(
                entity,
                display,
                colors,
                atom_offset,
                mode,
                per_residue_colors,
                &mut out,
            );
            atom_offset += entity_atom_count;
        }

        (out.spheres, out.bonds)
    }

    /// Generate instances for a single entity, dispatching by molecule type.
    ///
    /// Proteins and nucleic acids only appear here when the user has
    /// overridden their drawing mode to Stick or `BallAndStick`.
    fn generate_entity_instances(
        entity: &MoleculeEntity,
        display: &DisplayOptions,
        colors: Option<&ColorOptions>,
        atom_offset: u32,
        mode: DrawingMode,
        per_residue_colors: Option<&[[f32; 3]]>,
        out: &mut InstanceCollector,
    ) {
        use molex::MoleculeType;

        let atoms = entity.atom_set();
        match entity.molecule_type() {
            // Protein/NA in Stick or BnS mode only — Cartoon-mode
            // polymers are in non_protein_entities for the NA renderer
            // but must NOT produce BnS instances.
            MoleculeType::Protein | MoleculeType::DNA | MoleculeType::RNA
                if mode != DrawingMode::Cartoon =>
            {
                instances::generate_polymer_bns_instances(
                    entity,
                    atom_offset,
                    mode,
                    per_residue_colors,
                    out,
                );
            }
            MoleculeType::Ligand | MoleculeType::Cofactor => {
                let tint = (entity.molecule_type() == MoleculeType::Cofactor)
                    .then(|| instances::resolve_cofactor_tint(entity, colors));
                instances::generate_ligand_instances(
                    atoms,
                    tint,
                    atom_offset,
                    out,
                );
            }
            MoleculeType::Lipid => {
                let lipid_tint =
                    colors.map_or(LIPID_CARBON_TINT, |c| c.lipid_carbon_tint);
                if display.lipid_ball_and_stick() {
                    instances::generate_ligand_instances(
                        atoms,
                        Some(lipid_tint),
                        atom_offset,
                        out,
                    );
                } else {
                    instances::generate_coarse_lipid_instances(
                        atoms,
                        lipid_tint,
                        atom_offset,
                        out,
                    );
                }
            }
            MoleculeType::Ion if display.show_ions => {
                instances::generate_ion_instances(atoms, atom_offset, out);
            }
            MoleculeType::Water if display.show_waters => {
                instances::generate_water_instances(atoms, atom_offset, out);
            }
            MoleculeType::Solvent if display.show_solvent => {
                let sc = colors.map_or([0.6, 0.6, 0.6], |c| c.solvent_color);
                instances::generate_solvent_instances(
                    atoms,
                    sc,
                    atom_offset,
                    out,
                );
            }
            _ => {}
        }
    }

    /// Regenerate all instances from entity data.
    pub fn update_from_entities(
        &mut self,
        context: &RenderContext,
        entities: &[MoleculeEntity],
        display: &DisplayOptions,
        colors: Option<&ColorOptions>,
    ) {
        let (sphere_instances, bond_instances) = Self::generate_all_instances(
            entities,
            display,
            colors,
            0,
            DrawingMode::BallAndStick,
            None,
        );
        let _ = self.sphere_pass.write_instances(
            &context.device,
            &context.queue,
            &sphere_instances,
        );
        let _ = self.bond_pass.write_instances(
            &context.device,
            &context.queue,
            &bond_instances,
        );
    }

    /// Apply pre-computed instance data (GPU upload only, no CPU generation).
    pub fn apply_prepared(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &PreparedBallAndStickData,
    ) {
        let _ = self.sphere_pass.write_bytes(
            device,
            queue,
            data.sphere_bytes,
            data.sphere_count,
        );
        let _ = self.bond_pass.write_bytes(
            device,
            queue,
            data.capsule_bytes,
            data.capsule_count,
        );
    }

    /// Draw both spheres and bonds in a single render pass.
    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &crate::renderer::draw_context::DrawBindGroups<'a>,
    ) {
        self.sphere_pass.draw(render_pass, bind_groups);
        self.bond_pass.draw(render_pass, bind_groups);
    }

    /// Get the sphere instance buffer (visual pass).
    pub fn sphere_buffer(&self) -> &wgpu::Buffer {
        self.sphere_pass.buffer()
    }

    /// Get the bond capsule instance buffer (visual pass).
    pub fn bond_buffer(&self) -> &wgpu::Buffer {
        self.bond_pass.buffer()
    }

    /// Get the sphere instance count.
    pub fn sphere_count(&self) -> u32 {
        self.sphere_pass.instance_count
    }

    /// Get the bond capsule instance count.
    pub fn bond_count(&self) -> u32 {
        self.bond_pass.instance_count
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![
            self.sphere_pass.buffer_info("BnS Spheres"),
            self.bond_pass.buffer_info("BnS Bonds"),
        ]
    }
}
