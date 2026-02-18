//! Ball-and-stick renderer for small molecules (ligands, ions, waters).
//!
//! Renders atoms as ray-cast sphere impostors and bonds as capsule impostors.
//! Uses distance-based bond inference from
//! `foldit_conv::coords::bond_inference`.

use bytemuck::Zeroable;
use foldit_conv::coords::{
    infer_bonds, types::Element, BondOrder, InferredBond, MoleculeEntity,
    MoleculeType, DEFAULT_TOLERANCE,
};
use glam::Vec3;

use super::capsule_instance::CapsuleInstance;
use crate::{
    gpu::{
        dynamic_buffer::TypedBuffer, render_context::RenderContext,
        shader_composer::ShaderComposer,
    },
    renderer::pipeline_util,
    util::options::{ColorOptions, DisplayOptions},
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

/// Return atom color: if a carbon tint is provided, carbon atoms use it;
/// all other elements (N, O, S, P, etc.) keep standard CPK coloring.
fn atom_color(elem: Element, carbon_tint: Option<[f32; 3]>) -> [f32; 3] {
    match (elem, carbon_tint) {
        (Element::C, Some(tint)) => tint,
        _ => elem.cpk_color(),
    }
}

/// Display mode for lipid entities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LipidDisplayMode {
    /// Coarse-grained: P spheres, head-group O/N, thin tail bonds, skip H
    #[default]
    CoarseGrained,
    /// Full ball-and-stick (same as ligands)
    BallAndStick,
}

/// Per-instance data for sphere impostor.
/// Must match the WGSL SphereInstance struct layout.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SphereInstance {
    /// xyz = position, w = radius
    pub(crate) center: [f32; 4],
    /// xyz = RGB color, w = entity_id (packed as float)
    pub(crate) color: [f32; 4],
}

/// Pre-computed instance data for GPU upload.
pub struct PreparedBallAndStickData<'a> {
    pub sphere_bytes: &'a [u8],
    pub sphere_count: u32,
    pub capsule_bytes: &'a [u8],
    pub capsule_count: u32,
    pub picking_bytes: &'a [u8],
    pub picking_count: u32,
}

pub struct BallAndStickRenderer {
    // Atom spheres
    sphere_pipeline: wgpu::RenderPipeline,
    sphere_buffer: TypedBuffer<SphereInstance>,
    sphere_bind_group_layout: wgpu::BindGroupLayout,
    sphere_bind_group: wgpu::BindGroup,
    sphere_count: u32,

    // Bonds (reuses capsule_impostor.wgsl)
    bond_pipeline: wgpu::RenderPipeline,
    bond_buffer: TypedBuffer<CapsuleInstance>,
    bond_bind_group_layout: wgpu::BindGroupLayout,
    bond_bind_group: wgpu::BindGroup,
    bond_count: u32,

    // Picking (degenerate capsules for spheres + normal capsules for bonds)
    picking_buffer: TypedBuffer<CapsuleInstance>,
    picking_bind_group_layout: wgpu::BindGroupLayout,
    picking_bind_group: wgpu::BindGroup,
    picking_count: u32,
}

impl BallAndStickRenderer {
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
    ) -> Self {
        // Create sphere pipeline
        let sphere_buffer = TypedBuffer::new_with_data(
            &context.device,
            "Ball-and-Stick Sphere Buffer",
            &[SphereInstance::zeroed()],
            wgpu::BufferUsages::STORAGE,
        );

        let sphere_bind_group_layout =
            Self::create_storage_layout(&context.device, "Sphere");
        let sphere_bind_group = Self::create_storage_bind_group(
            &context.device,
            &sphere_bind_group_layout,
            &sphere_buffer,
            "Sphere",
        );
        let sphere_pipeline = Self::create_sphere_pipeline(
            context,
            &sphere_bind_group_layout,
            camera_layout,
            lighting_layout,
            selection_layout,
            shader_composer,
        );

        // Create bond pipeline (reuses capsule_impostor.wgsl)
        let bond_buffer = TypedBuffer::new_with_data(
            &context.device,
            "Ball-and-Stick Bond Buffer",
            &[CapsuleInstance::zeroed()],
            wgpu::BufferUsages::STORAGE,
        );

        let bond_bind_group_layout =
            Self::create_storage_layout(&context.device, "Bond");
        let bond_bind_group = Self::create_storage_bind_group(
            &context.device,
            &bond_bind_group_layout,
            &bond_buffer,
            "Bond",
        );
        let bond_pipeline = Self::create_bond_pipeline(
            context,
            &bond_bind_group_layout,
            camera_layout,
            lighting_layout,
            selection_layout,
            shader_composer,
        );

        // Create picking buffer (degenerate capsules)
        let picking_buffer = TypedBuffer::new_with_data(
            &context.device,
            "Ball-and-Stick Picking Buffer",
            &[CapsuleInstance::zeroed()],
            wgpu::BufferUsages::STORAGE,
        );
        let picking_bind_group_layout =
            Self::create_storage_layout(&context.device, "BnS Picking");
        let picking_bind_group = Self::create_storage_bind_group(
            &context.device,
            &picking_bind_group_layout,
            &picking_buffer,
            "BnS Picking",
        );

        Self {
            sphere_pipeline,
            sphere_buffer,
            sphere_bind_group_layout,
            sphere_bind_group,
            sphere_count: 0,

            bond_pipeline,
            bond_buffer,
            bond_bind_group_layout,
            bond_bind_group,
            bond_count: 0,

            picking_buffer,
            picking_bind_group_layout,
            picking_bind_group,
            picking_count: 0,
        }
    }

    fn create_storage_layout(
        device: &wgpu::Device,
        label: &str,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("Ball-and-Stick {} Layout", label)),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    fn create_storage_bind_group<T: bytemuck::Pod>(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffer: &TypedBuffer<T>,
        label: &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.buffer().as_entire_binding(),
            }],
            label: Some(&format!("Ball-and-Stick {} Bind Group", label)),
        })
    }

    fn create_sphere_pipeline(
        context: &RenderContext,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
    ) -> wgpu::RenderPipeline {
        let shader = shader_composer.compose(
            &context.device,
            "Sphere Impostor Shader",
            include_str!("../../../assets/shaders/raster/impostor/sphere.wgsl"),
            "sphere_impostor.wgsl",
        );

        let pipeline_layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Sphere Impostor Pipeline Layout"),
                bind_group_layouts: &[
                    bind_group_layout,
                    camera_layout,
                    lighting_layout,
                    selection_layout,
                ],
                immediate_size: 0,
            },
        );

        context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Sphere Impostor Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &pipeline_util::hdr_fragment_targets(),
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(pipeline_util::depth_stencil_state()),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
    }

    fn create_bond_pipeline(
        context: &RenderContext,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
    ) -> wgpu::RenderPipeline {
        // Reuse capsule_impostor.wgsl for bonds
        let shader = shader_composer.compose(
            &context.device,
            "Ball-and-Stick Bond Shader",
            include_str!(
                "../../../assets/shaders/raster/impostor/capsule.wgsl"
            ),
            "capsule_impostor.wgsl",
        );

        let pipeline_layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Ball-and-Stick Bond Pipeline Layout"),
                bind_group_layouts: &[
                    bind_group_layout,
                    camera_layout,
                    lighting_layout,
                    selection_layout,
                ],
                immediate_size: 0,
            },
        );

        context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Ball-and-Stick Bond Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &pipeline_util::hdr_fragment_targets(),
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(pipeline_util::depth_stencil_state()),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
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
        let mut sphere_instances = Vec::new();
        let mut bond_instances = Vec::new();
        let mut picking_instances = Vec::new();

        for entity in entities {
            match entity.molecule_type {
                MoleculeType::Ligand => {
                    Self::generate_ligand_instances(
                        &entity.coords,
                        entity.entity_id,
                        None,
                        &mut sphere_instances,
                        &mut bond_instances,
                        &mut picking_instances,
                    );
                }
                MoleculeType::Cofactor => {
                    let tint = entity
                        .coords
                        .res_names
                        .first()
                        .map(|rn| {
                            if let Some(c) = colors {
                                c.cofactor_tint(
                                    std::str::from_utf8(rn)
                                        .unwrap_or("")
                                        .trim(),
                                )
                            } else {
                                Self::cofactor_carbon_tint(rn)
                            }
                        })
                        .unwrap_or([0.5, 0.5, 0.5]);
                    Self::generate_ligand_instances(
                        &entity.coords,
                        entity.entity_id,
                        Some(tint),
                        &mut sphere_instances,
                        &mut bond_instances,
                        &mut picking_instances,
                    );
                }
                MoleculeType::Lipid => {
                    let lipid_tint = colors
                        .map_or(LIPID_CARBON_TINT, |c| c.lipid_carbon_tint);
                    if display.lipid_ball_and_stick() {
                        Self::generate_ligand_instances(
                            &entity.coords,
                            entity.entity_id,
                            Some(lipid_tint),
                            &mut sphere_instances,
                            &mut bond_instances,
                            &mut picking_instances,
                        );
                    } else {
                        Self::generate_coarse_lipid_instances(
                            &entity.coords,
                            entity.entity_id,
                            lipid_tint,
                            &mut sphere_instances,
                            &mut bond_instances,
                            &mut picking_instances,
                        );
                    }
                }
                MoleculeType::Ion if display.show_ions => {
                    Self::generate_ion_instances(
                        &entity.coords,
                        entity.entity_id,
                        &mut sphere_instances,
                        &mut picking_instances,
                    );
                }
                MoleculeType::Water if display.show_waters => {
                    Self::generate_water_instances(
                        &entity.coords,
                        entity.entity_id,
                        &mut sphere_instances,
                        &mut bond_instances,
                        &mut picking_instances,
                    );
                }
                MoleculeType::Solvent if display.show_solvent => {
                    let sc =
                        colors.map_or([0.6, 0.6, 0.6], |c| c.solvent_color);
                    Self::generate_solvent_instances(
                        &entity.coords,
                        entity.entity_id,
                        sc,
                        &mut sphere_instances,
                        &mut picking_instances,
                    );
                }
                _ => {}
            }
        }

        (sphere_instances, bond_instances, picking_instances)
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
        self.apply_instances(
            device,
            queue,
            &sphere_instances,
            &bond_instances,
            &picking_instances,
        );
    }

    /// Upload pre-computed instances to GPU buffers, recreating bind groups if
    /// reallocated.
    fn apply_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sphere_instances: &[SphereInstance],
        bond_instances: &[CapsuleInstance],
        picking_instances: &[CapsuleInstance],
    ) {
        // Update sphere buffer
        let sphere_reallocated = if sphere_instances.is_empty() {
            self.sphere_buffer
                .write(device, queue, &[SphereInstance::zeroed()])
        } else {
            self.sphere_buffer.write(device, queue, sphere_instances)
        };
        if sphere_reallocated {
            self.sphere_bind_group = Self::create_storage_bind_group(
                device,
                &self.sphere_bind_group_layout,
                &self.sphere_buffer,
                "Sphere",
            );
        }
        self.sphere_count = sphere_instances.len() as u32;

        // Update bond buffer
        let bond_reallocated = if bond_instances.is_empty() {
            self.bond_buffer
                .write(device, queue, &[CapsuleInstance::zeroed()])
        } else {
            self.bond_buffer.write(device, queue, bond_instances)
        };
        if bond_reallocated {
            self.bond_bind_group = Self::create_storage_bind_group(
                device,
                &self.bond_bind_group_layout,
                &self.bond_buffer,
                "Bond",
            );
        }
        self.bond_count = bond_instances.len() as u32;

        // Update picking buffer
        let picking_reallocated = if picking_instances.is_empty() {
            self.picking_buffer.write(
                device,
                queue,
                &[CapsuleInstance::zeroed()],
            )
        } else {
            self.picking_buffer.write(device, queue, picking_instances)
        };
        if picking_reallocated {
            self.picking_bind_group = Self::create_storage_bind_group(
                device,
                &self.picking_bind_group_layout,
                &self.picking_buffer,
                "BnS Picking",
            );
        }
        self.picking_count = picking_instances.len() as u32;
    }

    /// Generate ball-and-stick instances for a small molecule entity.
    ///
    /// When `carbon_tint` is `Some(color)`, carbon atoms and their bond
    /// endpoints use the tint color instead of CPK gray; heteroatoms (N, O,
    /// S, P, …) keep standard CPK coloring. Pass `None` for plain ligand
    /// rendering.
    fn generate_ligand_instances(
        coords: &foldit_conv::coords::Coords,
        _entity_id: u32,
        carbon_tint: Option<[f32; 3]>,
        spheres: &mut Vec<SphereInstance>,
        bonds: &mut Vec<CapsuleInstance>,
        picking: &mut Vec<CapsuleInstance>,
    ) {
        let positions: Vec<Vec3> = coords
            .atoms
            .iter()
            .map(|a| Vec3::new(a.x, a.y, a.z))
            .collect();

        // Generate atom spheres
        for (i, pos) in positions.iter().enumerate() {
            let elem =
                coords.elements.get(i).copied().unwrap_or(Element::Unknown);
            let color = atom_color(elem, carbon_tint);
            let radius = elem.vdw_radius() * BALL_RADIUS_SCALE;
            // Use a large residue_idx offset so picking doesn't conflict with
            // protein
            let pick_id = 100000 + i as u32;

            spheres.push(SphereInstance {
                center: [pos.x, pos.y, pos.z, radius],
                color: [color[0], color[1], color[2], pick_id as f32],
            });

            // Degenerate capsule for picking (endpoint_a == endpoint_b)
            picking.push(CapsuleInstance {
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
            let pick_id = 100000 + bond.atom_a as u32;

            match bond.order {
                BondOrder::Double => {
                    // Two parallel capsules offset perpendicular to bond axis
                    let axis = (pos_b - pos_a).normalize_or_zero();
                    let perp = find_perpendicular(axis);
                    let offset = perp * DOUBLE_BOND_OFFSET;
                    let thin_radius = BOND_RADIUS * 0.7;

                    // Bond 1 (offset +)
                    let a1 = pos_a + offset;
                    let b1 = pos_b + offset;
                    bonds.push(CapsuleInstance {
                        endpoint_a: [a1.x, a1.y, a1.z, thin_radius],
                        endpoint_b: [b1.x, b1.y, b1.z, pick_id as f32],
                        color_a: [color_a[0], color_a[1], color_a[2], 0.0],
                        color_b: [color_b[0], color_b[1], color_b[2], 0.0],
                    });
                    picking.push(CapsuleInstance {
                        endpoint_a: [a1.x, a1.y, a1.z, thin_radius],
                        endpoint_b: [b1.x, b1.y, b1.z, pick_id as f32],
                        color_a: [0.0; 4],
                        color_b: [0.0; 4],
                    });

                    // Bond 2 (offset -)
                    let a2 = pos_a - offset;
                    let b2 = pos_b - offset;
                    bonds.push(CapsuleInstance {
                        endpoint_a: [a2.x, a2.y, a2.z, thin_radius],
                        endpoint_b: [b2.x, b2.y, b2.z, pick_id as f32],
                        color_a: [color_a[0], color_a[1], color_a[2], 0.0],
                        color_b: [color_b[0], color_b[1], color_b[2], 0.0],
                    });
                    picking.push(CapsuleInstance {
                        endpoint_a: [a2.x, a2.y, a2.z, thin_radius],
                        endpoint_b: [b2.x, b2.y, b2.z, pick_id as f32],
                        color_a: [0.0; 4],
                        color_b: [0.0; 4],
                    });
                }
                _ => {
                    // Single bond (or triple/aromatic rendered as single for
                    // now)
                    bonds.push(CapsuleInstance {
                        endpoint_a: [pos_a.x, pos_a.y, pos_a.z, BOND_RADIUS],
                        endpoint_b: [pos_b.x, pos_b.y, pos_b.z, pick_id as f32],
                        color_a: [color_a[0], color_a[1], color_a[2], 0.0],
                        color_b: [color_b[0], color_b[1], color_b[2], 0.0],
                    });
                    picking.push(CapsuleInstance {
                        endpoint_a: [pos_a.x, pos_a.y, pos_a.z, BOND_RADIUS],
                        endpoint_b: [pos_b.x, pos_b.y, pos_b.z, pick_id as f32],
                        color_a: [0.0; 4],
                        color_b: [0.0; 4],
                    });
                }
            }
        }
    }

    fn generate_coarse_lipid_instances(
        coords: &foldit_conv::coords::Coords,
        _entity_id: u32,
        lipid_carbon_tint: [f32; 3],
        spheres: &mut Vec<SphereInstance>,
        bonds: &mut Vec<CapsuleInstance>,
        picking: &mut Vec<CapsuleInstance>,
    ) {
        let positions: Vec<Vec3> = coords
            .atoms
            .iter()
            .map(|a| Vec3::new(a.x, a.y, a.z))
            .collect();

        // Generate atom spheres (skip H entirely, CG radii for others)
        for (i, pos) in positions.iter().enumerate() {
            let elem =
                coords.elements.get(i).copied().unwrap_or(Element::Unknown);
            let pick_id = 100000 + i as u32;

            match elem {
                Element::H => continue, // skip hydrogen entirely
                Element::P => {
                    // Phosphorus: medium sphere, CPK orange
                    let color = elem.cpk_color();
                    spheres.push(SphereInstance {
                        center: [pos.x, pos.y, pos.z, 1.0],
                        color: [color[0], color[1], color[2], pick_id as f32],
                    });
                    // Degenerate capsule for picking (P atoms only)
                    picking.push(CapsuleInstance {
                        endpoint_a: [pos.x, pos.y, pos.z, 1.0],
                        endpoint_b: [pos.x, pos.y, pos.z, pick_id as f32],
                        color_a: [0.0; 4],
                        color_b: [0.0; 4],
                    });
                }
                Element::O | Element::N => {
                    // Head-group: small spheres, element-colored
                    let color = elem.cpk_color();
                    spheres.push(SphereInstance {
                        center: [pos.x, pos.y, pos.z, 0.35],
                        color: [color[0], color[1], color[2], pick_id as f32],
                    });
                }
                // C and everything else: no sphere in CG mode
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
                        let pick_id = 100000 + i as u32;

                        // C-C tail bonds thin, head-group bonds thicker
                        let both_carbon =
                            elem_i == Element::C && elem_j == Element::C;
                        let radius = if both_carbon { 0.06 } else { 0.10 };
                        // Carbon gets lipid tint, heteroatoms keep CPK
                        let color_a = atom_color(elem_i, lipid_tint);
                        let color_b = atom_color(elem_j, lipid_tint);

                        bonds.push(CapsuleInstance {
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
        _entity_id: u32,
        spheres: &mut Vec<SphereInstance>,
        picking: &mut Vec<CapsuleInstance>,
    ) {
        for (i, atom) in coords.atoms.iter().enumerate() {
            let elem =
                coords.elements.get(i).copied().unwrap_or(Element::Unknown);
            let color = elem.cpk_color();
            let radius = elem.vdw_radius() * ION_RADIUS_SCALE;
            let pick_id = 100000 + i as u32;

            spheres.push(SphereInstance {
                center: [atom.x, atom.y, atom.z, radius],
                color: [color[0], color[1], color[2], pick_id as f32],
            });

            // Degenerate capsule for picking
            picking.push(CapsuleInstance {
                endpoint_a: [atom.x, atom.y, atom.z, radius],
                endpoint_b: [atom.x, atom.y, atom.z, pick_id as f32],
                color_a: [0.0; 4],
                color_b: [0.0; 4],
            });
        }
    }

    fn generate_water_instances(
        coords: &foldit_conv::coords::Coords,
        _entity_id: u32,
        spheres: &mut Vec<SphereInstance>,
        bonds: &mut Vec<CapsuleInstance>,
        picking: &mut Vec<CapsuleInstance>,
    ) {
        let positions: Vec<Vec3> = coords
            .atoms
            .iter()
            .map(|a| Vec3::new(a.x, a.y, a.z))
            .collect();

        // Water oxygen: light blue sphere
        let water_color: [f32; 3] = [0.5, 0.7, 1.0];

        for (i, pos) in positions.iter().enumerate() {
            let elem =
                coords.elements.get(i).copied().unwrap_or(Element::Unknown);

            // Only render oxygens as visible spheres, skip hydrogens in water
            if elem == Element::O || elem == Element::Unknown {
                let pick_id = 100000 + i as u32;

                spheres.push(SphereInstance {
                    center: [pos.x, pos.y, pos.z, WATER_RADIUS],
                    color: [
                        water_color[0],
                        water_color[1],
                        water_color[2],
                        pick_id as f32,
                    ],
                });

                // Degenerate capsule for picking
                picking.push(CapsuleInstance {
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
                let pick_id = 100000 + bond.atom_a as u32;

                bonds.push(CapsuleInstance {
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
                picking.push(CapsuleInstance {
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
    fn cofactor_carbon_tint(res_name: &[u8; 3]) -> [f32; 3] {
        let name = std::str::from_utf8(res_name).unwrap_or("").trim();
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
        _entity_id: u32,
        solvent_color: [f32; 3],
        spheres: &mut Vec<SphereInstance>,
        picking: &mut Vec<CapsuleInstance>,
    ) {
        const SOLVENT_RADIUS: f32 = 0.15;

        for (i, atom) in coords.atoms.iter().enumerate() {
            let elem =
                coords.elements.get(i).copied().unwrap_or(Element::Unknown);
            if elem == Element::H {
                continue;
            }
            let pick_id = 100000 + i as u32;

            spheres.push(SphereInstance {
                center: [atom.x, atom.y, atom.z, SOLVENT_RADIUS],
                color: [
                    solvent_color[0],
                    solvent_color[1],
                    solvent_color[2],
                    pick_id as f32,
                ],
            });

            picking.push(CapsuleInstance {
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
        let PreparedBallAndStickData {
            sphere_bytes,
            sphere_count,
            capsule_bytes,
            capsule_count,
            picking_bytes,
            picking_count,
        } = *data;
        // Zeroed fallbacks for empty buffers (wgpu requires non-zero bind group
        // buffers)
        let sphere_zero = SphereInstance::zeroed();
        let capsule_zero = CapsuleInstance::zeroed();

        // Update sphere buffer
        let sphere_data = if sphere_bytes.is_empty() {
            bytemuck::bytes_of(&sphere_zero)
        } else {
            sphere_bytes
        };
        let sphere_reallocated =
            self.sphere_buffer.write_bytes(device, queue, sphere_data);
        if sphere_reallocated {
            self.sphere_bind_group = Self::create_storage_bind_group(
                device,
                &self.sphere_bind_group_layout,
                &self.sphere_buffer,
                "Sphere",
            );
        }
        self.sphere_count = sphere_count;

        // Update bond buffer
        let bond_data = if capsule_bytes.is_empty() {
            bytemuck::bytes_of(&capsule_zero)
        } else {
            capsule_bytes
        };
        let bond_reallocated =
            self.bond_buffer.write_bytes(device, queue, bond_data);
        if bond_reallocated {
            self.bond_bind_group = Self::create_storage_bind_group(
                device,
                &self.bond_bind_group_layout,
                &self.bond_buffer,
                "Bond",
            );
        }
        self.bond_count = capsule_count;

        // Update picking buffer
        let picking_data = if picking_bytes.is_empty() {
            bytemuck::bytes_of(&capsule_zero)
        } else {
            picking_bytes
        };
        let picking_reallocated =
            self.picking_buffer.write_bytes(device, queue, picking_data);
        if picking_reallocated {
            self.picking_bind_group = Self::create_storage_bind_group(
                device,
                &self.picking_bind_group_layout,
                &self.picking_buffer,
                "BnS Picking",
            );
        }
        self.picking_count = picking_count;
    }

    /// Draw both spheres and bonds in a single render pass.
    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &super::draw_context::DrawBindGroups<'a>,
    ) {
        // Draw atom spheres
        if self.sphere_count > 0 {
            render_pass.set_pipeline(&self.sphere_pipeline);
            render_pass.set_bind_group(0, &self.sphere_bind_group, &[]);
            render_pass.set_bind_group(1, bind_groups.camera, &[]);
            render_pass.set_bind_group(2, bind_groups.lighting, &[]);
            render_pass.set_bind_group(3, bind_groups.selection, &[]);
            render_pass.draw(0..6, 0..self.sphere_count);
        }

        // Draw bonds
        if self.bond_count > 0 {
            render_pass.set_pipeline(&self.bond_pipeline);
            render_pass.set_bind_group(0, &self.bond_bind_group, &[]);
            render_pass.set_bind_group(1, bind_groups.camera, &[]);
            render_pass.set_bind_group(2, bind_groups.lighting, &[]);
            render_pass.set_bind_group(3, bind_groups.selection, &[]);
            render_pass.draw(0..6, 0..self.bond_count);
        }
    }

    /// Get the picking buffer for the picking pass.
    pub fn picking_buffer(&self) -> &wgpu::Buffer {
        self.picking_buffer.buffer()
    }

    /// Get the picking instance count.
    pub fn picking_count(&self) -> u32 {
        self.picking_count
    }

    /// Get all non-protein atom positions for camera fitting.
    pub fn collect_positions(
        entities: &[MoleculeEntity],
        display: &DisplayOptions,
    ) -> Vec<Vec3> {
        let mut positions = Vec::new();
        for entity in entities {
            match entity.molecule_type {
                // Ligands, cofactors, lipids: always visible
                MoleculeType::Ligand
                | MoleculeType::Cofactor
                | MoleculeType::Lipid => {
                    for atom in &entity.coords.atoms {
                        positions.push(Vec3::new(atom.x, atom.y, atom.z));
                    }
                }
                MoleculeType::Ion if display.show_ions => {
                    for atom in &entity.coords.atoms {
                        positions.push(Vec3::new(atom.x, atom.y, atom.z));
                    }
                }
                MoleculeType::Water if display.show_waters => {
                    for atom in &entity.coords.atoms {
                        positions.push(Vec3::new(atom.x, atom.y, atom.z));
                    }
                }
                MoleculeType::Solvent if display.show_solvent => {
                    for atom in &entity.coords.atoms {
                        positions.push(Vec3::new(atom.x, atom.y, atom.z));
                    }
                }
                _ => {}
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

        let positions: Vec<Vec3> = coords
            .atoms
            .iter()
            .map(|a| Vec3::new(a.x, a.y, a.z))
            .collect();

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
impl super::MolecularRenderer for BallAndStickRenderer {
    fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &super::draw_context::DrawBindGroups<'a>,
    ) {
        self.draw(render_pass, bind_groups);
    }
}

fn find_perpendicular(v: Vec3) -> Vec3 {
    if v.length_squared() < 1e-8 {
        return Vec3::X;
    }
    let candidate = if v.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    v.cross(candidate).normalize()
}
