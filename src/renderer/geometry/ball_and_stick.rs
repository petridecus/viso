//! Ball-and-stick renderer for small molecules (ligands, ions, waters).
//!
//! Renders atoms as ray-cast sphere impostors and bonds as capsule impostors.
//! Uses distance-based bond inference from
//! `foldit_conv::coords::bond_inference`.

use foldit_conv::coords::{
    infer_bonds, types::Element, BondOrder, InferredBond, MoleculeEntity,
    MoleculeType, DEFAULT_TOLERANCE,
};
use glam::Vec3;

use crate::{
    gpu::{render_context::RenderContext, shader_composer::ShaderComposer},
    options::{ColorOptions, DisplayOptions},
    renderer::impostor::{
        capsule::CapsuleInstance, sphere::SphereInstance, ImpostorPass,
        ShaderDef,
    },
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
fn atom_positions(coords: &foldit_conv::coords::Coords) -> Vec<Vec3> {
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

/// Renders small molecules as ray-cast sphere + capsule impostors.
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
    ) -> Self {
        let sphere_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "BnS Sphere",
                path: "raster/impostor/sphere.wgsl",
            },
            layouts,
            6,
            shader_composer,
        );

        let bond_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "BnS Bond",
                path: "raster/impostor/capsule.wgsl",
            },
            layouts,
            6,
            shader_composer,
        );

        let picking_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "BnS Picking",
                path: "raster/impostor/capsule.wgsl",
            },
            layouts,
            6,
            shader_composer,
        );

        Self {
            sphere_pass,
            bond_pass,
            picking_pass,
        }
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
                    let tint = entity.coords.res_names.first().map_or(
                        [0.5, 0.5, 0.5],
                        |rn| {
                            if let Some(c) = colors {
                                c.cofactor_tint(
                                    std::str::from_utf8(rn)
                                        .unwrap_or("")
                                        .trim(),
                                )
                            } else {
                                Self::cofactor_carbon_tint(*rn)
                            }
                        },
                    );
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        entities: &[MoleculeEntity],
        display: &DisplayOptions,
        colors: Option<&ColorOptions>,
    ) {
        let (sphere_instances, bond_instances, picking_instances) =
            Self::generate_all_instances(entities, display, colors);
        let _ =
            self.sphere_pass
                .write_instances(device, queue, &sphere_instances);
        let _ = self
            .bond_pass
            .write_instances(device, queue, &bond_instances);
        let _ = self.picking_pass.write_instances(
            device,
            queue,
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
        coords: &foldit_conv::coords::Coords,
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
            // Use a large residue_idx offset so picking doesn't conflict with
            // protein
            let pick_id = 100_000 + i as u32;

            out.spheres.push(SphereInstance {
                center: [pos.x, pos.y, pos.z, radius],
                color: [color[0], color[1], color[2], pick_id as f32],
            });

            // Degenerate capsule for picking (endpoint_a == endpoint_b)
            out.picking.push(CapsuleInstance {
                endpoint_a: [pos.x, pos.y, pos.z, radius],
                endpoint_b: [pos.x, pos.y, pos.z, pick_id as f32],
                color_a: [0.0; 4],
                color_b: [0.0; 4],
            });
        }

        // Infer bonds per-residue (avoids O(n²) on large multi-molecule
        // entities)
        let inferred_bonds = Self::infer_bonds_per_residue(coords);
        for bond in &inferred_bonds {
            let pos_a = positions[bond.atom_a];
            let pos_b = positions[bond.atom_b];
            let elem_a = coords
                .elements
                .get(bond.atom_a)
                .copied()
                .unwrap_or(Element::Unknown);
            let elem_b = coords
                .elements
                .get(bond.atom_b)
                .copied()
                .unwrap_or(Element::Unknown);
            let color_a = atom_color(elem_a, carbon_tint);
            let color_b = atom_color(elem_b, carbon_tint);
            // Use pick_id from atom_a for the bond
            let pick_id = 100_000 + bond.atom_a as u32;

            if bond.order == BondOrder::Double {
                // Two parallel capsules offset perpendicular to bond axis
                let axis = (pos_b - pos_a).normalize_or_zero();
                let perp = find_perpendicular(axis);
                let offset = perp * DOUBLE_BOND_OFFSET;
                let thin_radius = BOND_RADIUS * 0.7;

                // Bond 1 (offset +)
                let a1 = pos_a + offset;
                let b1 = pos_b + offset;
                out.bonds.push(CapsuleInstance {
                    endpoint_a: [a1.x, a1.y, a1.z, thin_radius],
                    endpoint_b: [b1.x, b1.y, b1.z, pick_id as f32],
                    color_a: [color_a[0], color_a[1], color_a[2], 0.0],
                    color_b: [color_b[0], color_b[1], color_b[2], 0.0],
                });
                out.picking.push(CapsuleInstance {
                    endpoint_a: [a1.x, a1.y, a1.z, thin_radius],
                    endpoint_b: [b1.x, b1.y, b1.z, pick_id as f32],
                    color_a: [0.0; 4],
                    color_b: [0.0; 4],
                });

                // Bond 2 (offset -)
                let a2 = pos_a - offset;
                let b2 = pos_b - offset;
                out.bonds.push(CapsuleInstance {
                    endpoint_a: [a2.x, a2.y, a2.z, thin_radius],
                    endpoint_b: [b2.x, b2.y, b2.z, pick_id as f32],
                    color_a: [color_a[0], color_a[1], color_a[2], 0.0],
                    color_b: [color_b[0], color_b[1], color_b[2], 0.0],
                });
                out.picking.push(CapsuleInstance {
                    endpoint_a: [a2.x, a2.y, a2.z, thin_radius],
                    endpoint_b: [b2.x, b2.y, b2.z, pick_id as f32],
                    color_a: [0.0; 4],
                    color_b: [0.0; 4],
                });
            } else {
                // Single bond (or triple/aromatic rendered as single for
                // now)
                out.bonds.push(CapsuleInstance {
                    endpoint_a: [pos_a.x, pos_a.y, pos_a.z, BOND_RADIUS],
                    endpoint_b: [pos_b.x, pos_b.y, pos_b.z, pick_id as f32],
                    color_a: [color_a[0], color_a[1], color_a[2], 0.0],
                    color_b: [color_b[0], color_b[1], color_b[2], 0.0],
                });
                out.picking.push(CapsuleInstance {
                    endpoint_a: [pos_a.x, pos_a.y, pos_a.z, BOND_RADIUS],
                    endpoint_b: [pos_b.x, pos_b.y, pos_b.z, pick_id as f32],
                    color_a: [0.0; 4],
                    color_b: [0.0; 4],
                });
            }
        }
    }

    fn generate_coarse_lipid_instances(
        coords: &foldit_conv::coords::Coords,
        lipid_carbon_tint: [f32; 3],
        out: &mut InstanceCollector,
    ) {
        let positions = atom_positions(coords);

        // Generate atom spheres (skip H entirely, CG radii for others)
        for (i, pos) in positions.iter().enumerate() {
            let elem =
                coords.elements.get(i).copied().unwrap_or(Element::Unknown);
            let pick_id = 100_000 + i as u32;

            match elem {
                Element::P => {
                    // Phosphorus: medium sphere, CPK orange
                    let color = elem.cpk_color();
                    out.spheres.push(SphereInstance {
                        center: [pos.x, pos.y, pos.z, 1.0],
                        color: [color[0], color[1], color[2], pick_id as f32],
                    });
                    // Degenerate capsule for picking (P atoms only)
                    out.picking.push(CapsuleInstance {
                        endpoint_a: [pos.x, pos.y, pos.z, 1.0],
                        endpoint_b: [pos.x, pos.y, pos.z, pick_id as f32],
                        color_a: [0.0; 4],
                        color_b: [0.0; 4],
                    });
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

        // Infer bonds PER RESIDUE to avoid O(n²) on the entire lipid entity
        // (all lipids are grouped into one entity — distance checks across
        // ~95K atoms would take minutes; per-residue is ~130 atoms each).
        let lipid_tint = Some(lipid_carbon_tint);

        // Group atom indices by residue number
        let mut residue_atoms: std::collections::BTreeMap<i32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for i in 0..coords.num_atoms {
            residue_atoms.entry(coords.res_nums[i]).or_default().push(i);
        }

        for atom_indices in residue_atoms.values() {
            // Pairwise bond inference within this residue only
            for (ai, &i) in atom_indices.iter().enumerate() {
                let elem_i =
                    coords.elements.get(i).copied().unwrap_or(Element::Unknown);
                if elem_i == Element::H {
                    continue;
                }
                let cov_i = elem_i.covalent_radius();

                for &j in &atom_indices[ai + 1..] {
                    let elem_j = coords
                        .elements
                        .get(j)
                        .copied()
                        .unwrap_or(Element::Unknown);
                    if elem_j == Element::H {
                        continue;
                    }
                    let cov_j = elem_j.covalent_radius();

                    let dist = positions[i].distance(positions[j]);
                    let threshold = cov_i + cov_j + DEFAULT_TOLERANCE;

                    if dist <= threshold && dist > 0.4 {
                        let pos_a = positions[i];
                        let pos_b = positions[j];
                        let pick_id = 100_000 + i as u32;

                        // C-C tail bonds thin, head-group bonds thicker
                        let both_carbon =
                            elem_i == Element::C && elem_j == Element::C;
                        let radius = if both_carbon { 0.06 } else { 0.10 };
                        // Carbon gets lipid tint, heteroatoms keep CPK
                        let color_a = atom_color(elem_i, lipid_tint);
                        let color_b = atom_color(elem_j, lipid_tint);

                        out.bonds.push(CapsuleInstance {
                            endpoint_a: [pos_a.x, pos_a.y, pos_a.z, radius],
                            endpoint_b: [
                                pos_b.x,
                                pos_b.y,
                                pos_b.z,
                                pick_id as f32,
                            ],
                            color_a: [color_a[0], color_a[1], color_a[2], 0.0],
                            color_b: [color_b[0], color_b[1], color_b[2], 0.0],
                        });
                    }
                }
            }
        }
    }

    fn generate_ion_instances(
        coords: &foldit_conv::coords::Coords,
        out: &mut InstanceCollector,
    ) {
        for (i, atom) in coords.atoms.iter().enumerate() {
            let elem =
                coords.elements.get(i).copied().unwrap_or(Element::Unknown);
            let color = elem.cpk_color();
            let radius = elem.vdw_radius() * ION_RADIUS_SCALE;
            let pick_id = 100_000 + i as u32;

            out.spheres.push(SphereInstance {
                center: [atom.x, atom.y, atom.z, radius],
                color: [color[0], color[1], color[2], pick_id as f32],
            });

            // Degenerate capsule for picking
            out.picking.push(CapsuleInstance {
                endpoint_a: [atom.x, atom.y, atom.z, radius],
                endpoint_b: [atom.x, atom.y, atom.z, pick_id as f32],
                color_a: [0.0; 4],
                color_b: [0.0; 4],
            });
        }
    }

    fn generate_water_instances(
        coords: &foldit_conv::coords::Coords,
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
                let pick_id = 100_000 + i as u32;

                out.spheres.push(SphereInstance {
                    center: [pos.x, pos.y, pos.z, WATER_RADIUS],
                    color: [
                        water_color[0],
                        water_color[1],
                        water_color[2],
                        pick_id as f32,
                    ],
                });

                // Degenerate capsule for picking
                out.picking.push(CapsuleInstance {
                    endpoint_a: [pos.x, pos.y, pos.z, WATER_RADIUS],
                    endpoint_b: [pos.x, pos.y, pos.z, pick_id as f32],
                    color_a: [0.0; 4],
                    color_b: [0.0; 4],
                });
            }
        }

        // Optionally infer O-H bonds if hydrogens are present
        if coords.num_atoms > 1 {
            let inferred_bonds = infer_bonds(coords, DEFAULT_TOLERANCE);
            for bond in &inferred_bonds {
                let pos_a = positions[bond.atom_a];
                let pos_b = positions[bond.atom_b];
                let pick_id = 100_000 + bond.atom_a as u32;

                out.bonds.push(CapsuleInstance {
                    endpoint_a: [pos_a.x, pos_a.y, pos_a.z, BOND_RADIUS * 0.7],
                    endpoint_b: [pos_b.x, pos_b.y, pos_b.z, pick_id as f32],
                    color_a: [
                        water_color[0],
                        water_color[1],
                        water_color[2],
                        0.0,
                    ],
                    color_b: [0.9, 0.9, 0.9, 0.0], // White for hydrogen end
                });
                out.picking.push(CapsuleInstance {
                    endpoint_a: [pos_a.x, pos_a.y, pos_a.z, BOND_RADIUS * 0.7],
                    endpoint_b: [pos_b.x, pos_b.y, pos_b.z, pick_id as f32],
                    color_a: [0.0; 4],
                    color_b: [0.0; 4],
                });
            }
        }
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
        coords: &foldit_conv::coords::Coords,
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
            let pick_id = 100_000 + i as u32;

            out.spheres.push(SphereInstance {
                center: [atom.x, atom.y, atom.z, SOLVENT_RADIUS],
                color: [
                    solvent_color[0],
                    solvent_color[1],
                    solvent_color[2],
                    pick_id as f32,
                ],
            });

            out.picking.push(CapsuleInstance {
                endpoint_a: [atom.x, atom.y, atom.z, SOLVENT_RADIUS],
                endpoint_b: [atom.x, atom.y, atom.z, pick_id as f32],
                color_a: [0.0; 4],
                color_b: [0.0; 4],
            });
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
        coords: &foldit_conv::coords::Coords,
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
            for (ai, &i) in atom_indices.iter().enumerate() {
                let elem_i =
                    coords.elements.get(i).copied().unwrap_or(Element::Unknown);
                if elem_i == Element::H {
                    continue;
                }
                let cov_i = elem_i.covalent_radius();

                for &j in &atom_indices[ai + 1..] {
                    let elem_j = coords
                        .elements
                        .get(j)
                        .copied()
                        .unwrap_or(Element::Unknown);
                    if elem_i == Element::H && elem_j == Element::H {
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
        bonds
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
