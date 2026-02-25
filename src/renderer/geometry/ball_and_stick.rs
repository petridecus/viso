//! Ball-and-stick renderer for small molecules (ligands, ions, waters).
//!
//! Renders atoms as ray-cast sphere impostors and bonds as capsule impostors.
//! Uses distance-based bond inference from
//! `foldit_conv::ops::bond_inference`.

use foldit_conv::ops::bond_inference::{
    infer_bonds, BondOrder, InferredBond, DEFAULT_TOLERANCE,
};
use foldit_conv::types::coords::Element;
use foldit_conv::types::entity::{MoleculeEntity, MoleculeType};
use glam::Vec3;

use crate::error::VisoError;
use crate::gpu::render_context::RenderContext;
use crate::gpu::shader_composer::ShaderComposer;
use crate::options::{ColorOptions, DisplayOptions};
use crate::renderer::impostor::capsule::CapsuleInstance;
use crate::renderer::impostor::sphere::SphereInstance;
use crate::renderer::impostor::{ImpostorPass, ShaderDef};
use crate::renderer::picking::utils::{
    picking_bond, picking_sphere, SMALL_MOLECULE_PICK_OFFSET,
};

/// Radius for bond capsules (thinner than protein sidechains)
const BOND_RADIUS: f32 = 0.15;

/// Fraction of vdw_radius for ball-and-stick atom spheres
const BALL_RADIUS_SCALE: f32 = 0.3;

/// Larger spheres for ions
const ION_RADIUS_SCALE: f32 = 0.5;

/// Small sphere for water oxygen
const WATER_RADIUS: f32 = 0.3;

/// Perpendicular offset for double bond parallel capsules
const DOUBLE_BOND_OFFSET: f32 = 0.2;

/// Warm beige/tan carbon tint for lipid molecules.
const LIPID_CARBON_TINT: [f32; 3] = [0.76, 0.70, 0.50];

/// Collect atom positions as `Vec3` from a `Coords` block.
fn atom_positions(coords: &foldit_conv::types::coords::Coords) -> Vec<Vec3> {
    coords
        .atoms
        .iter()
        .map(|a| Vec3::new(a.x, a.y, a.z))
        .collect()
}

/// Return atom color: if a carbon tint is provided, carbon atoms use it;
/// all other elements (N, O, S, P, etc.) keep standard CPK coloring.
fn atom_color(elem: Element, carbon_tint: Option<[f32; 3]>) -> [f32; 3] {
    match (elem, carbon_tint) {
        (Element::C, Some(tint)) => tint,
        _ => elem.cpk_color(),
    }
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
    /// Raw bytes for picking capsule instance data.
    pub picking_bytes: &'a [u8],
    /// Number of picking capsule instances.
    pub picking_count: u32,
}

/// Output buffers for instance generation.
#[derive(Default)]
struct InstanceCollector {
    spheres: Vec<SphereInstance>,
    bonds: Vec<CapsuleInstance>,
    picking: Vec<CapsuleInstance>,
}

impl InstanceCollector {
    /// Push a visual bond capsule and its paired picking capsule.
    ///
    /// `endpoints` is `[pos_a, pos_b]`; `colors` is `[color_a, color_b]`.
    fn push_bond(
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
        self.picking
            .push(picking_bond(pos_a, pos_b, radius, pick_id));
    }

    /// Push a degenerate picking capsule (sphere) for atom picking.
    fn push_picking_sphere(
        &mut self,
        pos: [f32; 3],
        radius: f32,
        pick_id: u32,
    ) {
        self.picking.push(picking_sphere(pos, radius, pick_id));
    }
}

/// Renders small molecules as ray-cast sphere + capsule impostors.
#[allow(clippy::struct_field_names)]
pub struct BallAndStickRenderer {
    sphere_pass: ImpostorPass<SphereInstance>,
    bond_pass: ImpostorPass<CapsuleInstance>,
    picking_pass: ImpostorPass<CapsuleInstance>,
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
                path: "raster/impostor/sphere.wgsl",
            },
            layouts,
            6,
            shader_composer,
        )?;

        let bond_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "BnS Bond",
                path: "raster/impostor/capsule.wgsl",
            },
            layouts,
            6,
            shader_composer,
        )?;

        let picking_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "BnS Picking",
                path: "raster/impostor/capsule.wgsl",
            },
            layouts,
            6,
            shader_composer,
        )?;

        Ok(Self {
            sphere_pass,
            bond_pass,
            picking_pass,
        })
    }

    /// Generate all instances from entity data (pure CPU, no GPU access
    /// needed).
    ///
    /// Returns (sphere_instances, bond_instances, picking_instances).
    pub(crate) fn generate_all_instances(
        entities: &[MoleculeEntity],
        display: &DisplayOptions,
        colors: Option<&ColorOptions>,
    ) -> (
        Vec<SphereInstance>,
        Vec<CapsuleInstance>,
        Vec<CapsuleInstance>,
    ) {
        let mut out = InstanceCollector::default();

        for entity in entities {
            match entity.molecule_type {
                MoleculeType::Ligand => {
                    Self::generate_ligand_instances(
                        &entity.coords,
                        None,
                        &mut out,
                    );
                }
                MoleculeType::Cofactor => {
                    let tint =
                        Self::resolve_cofactor_tint(&entity.coords, colors);
                    Self::generate_ligand_instances(
                        &entity.coords,
                        Some(tint),
                        &mut out,
                    );
                }
                MoleculeType::Lipid => {
                    let lipid_tint = colors
                        .map_or(LIPID_CARBON_TINT, |c| c.lipid_carbon_tint);
                    if display.lipid_ball_and_stick() {
                        Self::generate_ligand_instances(
                            &entity.coords,
                            Some(lipid_tint),
                            &mut out,
                        );
                    } else {
                        Self::generate_coarse_lipid_instances(
                            &entity.coords,
                            lipid_tint,
                            &mut out,
                        );
                    }
                }
                MoleculeType::Ion if display.show_ions => {
                    Self::generate_ion_instances(&entity.coords, &mut out);
                }
                MoleculeType::Water if display.show_waters => {
                    Self::generate_water_instances(&entity.coords, &mut out);
                }
                MoleculeType::Solvent if display.show_solvent => {
                    let sc =
                        colors.map_or([0.6, 0.6, 0.6], |c| c.solvent_color);
                    Self::generate_solvent_instances(
                        &entity.coords,
                        sc,
                        &mut out,
                    );
                }
                _ => {}
            }
        }

        (out.spheres, out.bonds, out.picking)
    }

    /// Regenerate all instances from entity data.
    pub fn update_from_entities(
        &mut self,
        context: &RenderContext,
        entities: &[MoleculeEntity],
        display: &DisplayOptions,
        colors: Option<&ColorOptions>,
    ) {
        let (sphere_instances, bond_instances, picking_instances) =
            Self::generate_all_instances(entities, display, colors);
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
        let _ = self.picking_pass.write_instances(
            &context.device,
            &context.queue,
            &picking_instances,
        );
    }

    /// Generate ball-and-stick instances for a small molecule entity.
    ///
    /// When `carbon_tint` is `Some(color)`, carbon atoms and their bond
    /// endpoints use the tint color instead of CPK gray; heteroatoms (N, O,
    /// S, P, …) keep standard CPK coloring. Pass `None` for plain ligand
    /// rendering.
    fn generate_ligand_instances(
        coords: &foldit_conv::types::coords::Coords,
        carbon_tint: Option<[f32; 3]>,
        out: &mut InstanceCollector,
    ) {
        let positions = atom_positions(coords);

        // Generate atom spheres
        for (i, pos) in positions.iter().enumerate() {
            let elem =
                coords.elements.get(i).copied().unwrap_or(Element::Unknown);
            let color = atom_color(elem, carbon_tint);
            let radius = elem.vdw_radius() * BALL_RADIUS_SCALE;
            let pick_id = SMALL_MOLECULE_PICK_OFFSET + i as u32;
            let p = [pos.x, pos.y, pos.z];

            out.spheres.push(SphereInstance {
                center: [pos.x, pos.y, pos.z, radius],
                color: [color[0], color[1], color[2], pick_id as f32],
            });
            out.push_picking_sphere(p, radius, pick_id);
        }

        // Infer bonds per-residue (avoids O(n²) on large entities)
        let inferred_bonds = Self::infer_bonds_per_residue(coords);
        for bond in &inferred_bonds {
            let pos_a = positions[bond.atom_a];
            let pos_b = positions[bond.atom_b];
            let color_a = atom_color(
                coords
                    .elements
                    .get(bond.atom_a)
                    .copied()
                    .unwrap_or(Element::Unknown),
                carbon_tint,
            );
            let color_b = atom_color(
                coords
                    .elements
                    .get(bond.atom_b)
                    .copied()
                    .unwrap_or(Element::Unknown),
                carbon_tint,
            );
            let pick_id = SMALL_MOLECULE_PICK_OFFSET + bond.atom_a as u32;
            let a = [pos_a.x, pos_a.y, pos_a.z];
            let b = [pos_b.x, pos_b.y, pos_b.z];

            if bond.order == BondOrder::Double {
                let axis = (pos_b - pos_a).normalize_or_zero();
                let perp = find_perpendicular(axis);
                let offset = perp * DOUBLE_BOND_OFFSET;
                let thin_radius = BOND_RADIUS * 0.7;

                let a1 = pos_a + offset;
                let b1 = pos_b + offset;
                out.push_bond(
                    [[a1.x, a1.y, a1.z], [b1.x, b1.y, b1.z]],
                    thin_radius,
                    [color_a, color_b],
                    pick_id,
                );

                let a2 = pos_a - offset;
                let b2 = pos_b - offset;
                out.push_bond(
                    [[a2.x, a2.y, a2.z], [b2.x, b2.y, b2.z]],
                    thin_radius,
                    [color_a, color_b],
                    pick_id,
                );
            } else {
                out.push_bond([a, b], BOND_RADIUS, [color_a, color_b], pick_id);
            }
        }
    }

    fn generate_coarse_lipid_instances(
        coords: &foldit_conv::types::coords::Coords,
        lipid_carbon_tint: [f32; 3],
        out: &mut InstanceCollector,
    ) {
        let positions = atom_positions(coords);

        // Generate atom spheres (skip H entirely, CG radii for others)
        for (i, pos) in positions.iter().enumerate() {
            let elem =
                coords.elements.get(i).copied().unwrap_or(Element::Unknown);
            let pick_id = SMALL_MOLECULE_PICK_OFFSET + i as u32;

            match elem {
                Element::P => {
                    // Phosphorus: medium sphere, CPK orange
                    let color = elem.cpk_color();
                    out.spheres.push(SphereInstance {
                        center: [pos.x, pos.y, pos.z, 1.0],
                        color: [color[0], color[1], color[2], pick_id as f32],
                    });
                    out.push_picking_sphere(
                        [pos.x, pos.y, pos.z],
                        1.0,
                        pick_id,
                    );
                }
                Element::O | Element::N => {
                    // Head-group: small spheres, element-colored
                    let color = elem.cpk_color();
                    out.spheres.push(SphereInstance {
                        center: [pos.x, pos.y, pos.z, 0.35],
                        color: [color[0], color[1], color[2], pick_id as f32],
                    });
                }
                // H, C, and everything else: no sphere in CG mode
                _ => {}
            }
        }

        let bonds = Self::infer_bonds_per_residue(coords);
        Self::emit_coarse_bonds(
            &bonds,
            &positions,
            &coords.elements,
            Some(lipid_carbon_tint),
            out,
        );
    }

    /// Emit coarse-grained bond capsules from pre-inferred bonds, skipping
    /// hydrogen atoms and using thinner radii for C-C tail bonds.
    fn emit_coarse_bonds(
        bonds: &[InferredBond],
        positions: &[Vec3],
        elements: &[Element],
        carbon_tint: Option<[f32; 3]>,
        out: &mut InstanceCollector,
    ) {
        for bond in bonds {
            let elem_a = elements
                .get(bond.atom_a)
                .copied()
                .unwrap_or(Element::Unknown);
            let elem_b = elements
                .get(bond.atom_b)
                .copied()
                .unwrap_or(Element::Unknown);
            if elem_a == Element::H || elem_b == Element::H {
                continue;
            }

            let pos_a = positions[bond.atom_a];
            let pos_b = positions[bond.atom_b];
            let pick_id = SMALL_MOLECULE_PICK_OFFSET + bond.atom_a as u32;

            // C-C tail bonds thin, head-group bonds thicker
            let both_carbon = elem_a == Element::C && elem_b == Element::C;
            let radius = if both_carbon { 0.06 } else { 0.10 };
            let color_a = atom_color(elem_a, carbon_tint);
            let color_b = atom_color(elem_b, carbon_tint);

            out.bonds.push(CapsuleInstance {
                endpoint_a: [pos_a.x, pos_a.y, pos_a.z, radius],
                endpoint_b: [pos_b.x, pos_b.y, pos_b.z, pick_id as f32],
                color_a: [color_a[0], color_a[1], color_a[2], 0.0],
                color_b: [color_b[0], color_b[1], color_b[2], 0.0],
            });
        }
    }

    fn generate_ion_instances(
        coords: &foldit_conv::types::coords::Coords,
        out: &mut InstanceCollector,
    ) {
        for (i, atom) in coords.atoms.iter().enumerate() {
            let elem =
                coords.elements.get(i).copied().unwrap_or(Element::Unknown);
            let color = elem.cpk_color();
            let radius = elem.vdw_radius() * ION_RADIUS_SCALE;
            let pick_id = SMALL_MOLECULE_PICK_OFFSET + i as u32;
            let p = [atom.x, atom.y, atom.z];

            out.spheres.push(SphereInstance {
                center: [atom.x, atom.y, atom.z, radius],
                color: [color[0], color[1], color[2], pick_id as f32],
            });
            out.push_picking_sphere(p, radius, pick_id);
        }
    }

    fn generate_water_instances(
        coords: &foldit_conv::types::coords::Coords,
        out: &mut InstanceCollector,
    ) {
        let positions = atom_positions(coords);

        // Water oxygen: light blue sphere
        let water_color: [f32; 3] = [0.5, 0.7, 1.0];

        for (i, pos) in positions.iter().enumerate() {
            let elem =
                coords.elements.get(i).copied().unwrap_or(Element::Unknown);

            // Only render oxygens as visible spheres, skip hydrogens in water
            if elem == Element::O || elem == Element::Unknown {
                let pick_id = SMALL_MOLECULE_PICK_OFFSET + i as u32;
                let p = [pos.x, pos.y, pos.z];

                out.spheres.push(SphereInstance {
                    center: [pos.x, pos.y, pos.z, WATER_RADIUS],
                    color: [
                        water_color[0],
                        water_color[1],
                        water_color[2],
                        pick_id as f32,
                    ],
                });
                out.push_picking_sphere(p, WATER_RADIUS, pick_id);
            }
        }

        // Optionally infer O-H bonds if hydrogens are present
        if coords.num_atoms > 1 {
            let inferred_bonds = infer_bonds(coords, DEFAULT_TOLERANCE);
            for bond in &inferred_bonds {
                let pos_a = positions[bond.atom_a];
                let pos_b = positions[bond.atom_b];
                let pick_id = SMALL_MOLECULE_PICK_OFFSET + bond.atom_a as u32;
                let radius = BOND_RADIUS * 0.7;

                out.push_bond(
                    [[pos_a.x, pos_a.y, pos_a.z], [pos_b.x, pos_b.y, pos_b.z]],
                    radius,
                    [water_color, [0.9, 0.9, 0.9]], // White for hydrogen end
                    pick_id,
                );
            }
        }
    }

    /// Resolve the carbon tint for a cofactor entity, consulting user color
    /// options when available and falling back to built-in per-residue tints.
    fn resolve_cofactor_tint(
        coords: &foldit_conv::types::coords::Coords,
        colors: Option<&ColorOptions>,
    ) -> [f32; 3] {
        coords.res_names.first().map_or_else(
            || [0.5, 0.5, 0.5],
            |rn| {
                colors.map_or_else(
                    || Self::cofactor_carbon_tint(*rn),
                    |c| {
                        c.cofactor_tint(
                            std::str::from_utf8(rn).unwrap_or("").trim(),
                        )
                    },
                )
            },
        )
    }

    /// Carbon tint color for cofactor rendering, keyed by 3-letter residue
    /// name. Applied to carbon atoms and their bond endpoints; heteroatoms
    /// keep CPK.
    fn cofactor_carbon_tint(res_name: [u8; 3]) -> [f32; 3] {
        let name = std::str::from_utf8(&res_name).unwrap_or("").trim();
        match name {
            "CLA" => [0.2, 0.7, 0.3],         // green
            "CHL" => [0.2, 0.6, 0.35],        // green (chlorophyll B)
            "BCR" | "BCB" => [0.9, 0.5, 0.1], // orange
            "HEM" | "HEC" | "HEA" | "HEB" => [0.7, 0.15, 0.15], // dark red
            "PHO" => [0.5, 0.7, 0.3],         // yellow-green
            "PL9" | "PLQ" => [0.6, 0.5, 0.2], // amber
            _ => [0.5, 0.5, 0.5],             // neutral fallback
        }
    }

    /// Generate instances for solvent entities.
    /// Tiny gray spheres (no bonds), skip H atoms.
    fn generate_solvent_instances(
        coords: &foldit_conv::types::coords::Coords,
        solvent_color: [f32; 3],
        out: &mut InstanceCollector,
    ) {
        const SOLVENT_RADIUS: f32 = 0.15;

        for (i, atom) in coords.atoms.iter().enumerate() {
            let elem =
                coords.elements.get(i).copied().unwrap_or(Element::Unknown);
            if elem == Element::H {
                continue;
            }
            let pick_id = SMALL_MOLECULE_PICK_OFFSET + i as u32;
            let p = [atom.x, atom.y, atom.z];

            out.spheres.push(SphereInstance {
                center: [atom.x, atom.y, atom.z, SOLVENT_RADIUS],
                color: [
                    solvent_color[0],
                    solvent_color[1],
                    solvent_color[2],
                    pick_id as f32,
                ],
            });
            out.push_picking_sphere(p, SOLVENT_RADIUS, pick_id);
        }
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
        let _ = self.picking_pass.write_bytes(
            device,
            queue,
            data.picking_bytes,
            data.picking_count,
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

    /// Get the picking buffer for the picking pass.
    pub fn picking_buffer(&self) -> &wgpu::Buffer {
        self.picking_pass.buffer()
    }

    /// Get the picking instance count.
    pub fn picking_count(&self) -> u32 {
        self.picking_pass.instance_count
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![
            self.sphere_pass.buffer_info("BnS Spheres"),
            self.bond_pass.buffer_info("BnS Bonds"),
            self.picking_pass.buffer_info("BnS Picking"),
        ]
    }

    /// Get all non-protein atom positions for camera fitting.
    pub fn collect_positions(
        entities: &[MoleculeEntity],
        display: &DisplayOptions,
    ) -> Vec<Vec3> {
        let mut positions = Vec::new();
        for entity in entities {
            let visible = match entity.molecule_type {
                MoleculeType::Ligand
                | MoleculeType::Cofactor
                | MoleculeType::Lipid => true,
                MoleculeType::Ion => display.show_ions,
                MoleculeType::Water => display.show_waters,
                MoleculeType::Solvent => display.show_solvent,
                _ => false,
            };
            if visible {
                positions.extend(atom_positions(&entity.coords));
            }
        }
        positions
    }

    /// Infer bonds per-residue to avoid O(n²) on large multi-molecule entities.
    /// For single-residue entities (typical ligands), this is identical to
    /// `infer_bonds`. For multi-residue entities (e.g. all lipids lumped
    /// together), this is O(sum of k²) where k = atoms per residue (~130),
    /// vs O(n²) where n = total (~95K).
    fn infer_bonds_per_residue(
        coords: &foldit_conv::types::coords::Coords,
    ) -> Vec<InferredBond> {
        use std::collections::BTreeMap;

        let n = coords.num_atoms;
        if n < 2 {
            return Vec::new();
        }

        // If only one residue, delegate to the standard function
        let first_res = coords.res_nums[0];
        if coords.res_nums.iter().all(|&r| r == first_res) {
            return infer_bonds(coords, DEFAULT_TOLERANCE);
        }

        let positions = atom_positions(coords);

        // Group atom indices by residue number
        let mut residue_atoms: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
        for i in 0..n {
            residue_atoms.entry(coords.res_nums[i]).or_default().push(i);
        }

        let mut bonds = Vec::new();
        for atom_indices in residue_atoms.values() {
            infer_bonds_pairwise(
                atom_indices,
                &positions,
                &coords.elements,
                &mut bonds,
            );
        }
        bonds
    }
}

/// Infer bonds by pairwise distance check over a subset of atom indices.
/// Skips hydrogen atoms entirely.
fn infer_bonds_pairwise(
    atom_indices: &[usize],
    positions: &[Vec3],
    elements: &[Element],
    bonds: &mut Vec<InferredBond>,
) {
    for (ai, &i) in atom_indices.iter().enumerate() {
        let elem_i = elements.get(i).copied().unwrap_or(Element::Unknown);
        if elem_i == Element::H {
            continue;
        }
        let cov_i = elem_i.covalent_radius();

        for &j in &atom_indices[ai + 1..] {
            let elem_j = elements.get(j).copied().unwrap_or(Element::Unknown);
            if elem_j == Element::H {
                continue;
            }
            let cov_j = elem_j.covalent_radius();

            let dist = positions[i].distance(positions[j]);
            let sum_cov = cov_i + cov_j;
            let threshold = sum_cov + DEFAULT_TOLERANCE;

            if dist <= threshold && dist > 0.4 {
                let order = if dist < sum_cov * 0.9 {
                    BondOrder::Double
                } else {
                    BondOrder::Single
                };
                bonds.push(InferredBond {
                    atom_a: i,
                    atom_b: j,
                    order,
                });
            }
        }
    }
}

/// Find any vector perpendicular to the given vector.
fn find_perpendicular(v: Vec3) -> Vec3 {
    if v.length_squared() < 1e-8 {
        return Vec3::X;
    }
    let candidate = if v.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    v.cross(candidate).normalize()
}
