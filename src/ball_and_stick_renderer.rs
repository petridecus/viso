//! Ball-and-stick renderer for small molecules (ligands, ions, waters).
//!
//! Renders atoms as ray-cast sphere impostors and bonds as capsule impostors.
//! Uses distance-based bond inference from `foldit_conv::coords::bond_inference`.

use bytemuck::Zeroable;
use foldit_conv::coords::{
    infer_bonds, BondOrder, MoleculeEntity, MoleculeType, DEFAULT_TOLERANCE,
};
use foldit_conv::coords::types::Element;
use glam::Vec3;

use crate::dynamic_buffer::TypedBuffer;
use crate::render_context::RenderContext;

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

/// Per-instance data for sphere impostor.
/// Must match the WGSL SphereInstance struct layout.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SphereInstance {
    /// xyz = position, w = radius
    center: [f32; 4],
    /// xyz = RGB color, w = entity_id (packed as float)
    color: [f32; 4],
}

/// Per-instance data for capsule impostor (bonds).
/// Must match the WGSL CapsuleInstance struct layout.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CapsuleInstance {
    /// Endpoint A position (xyz), radius (w)
    endpoint_a: [f32; 4],
    /// Endpoint B position (xyz), residue_idx (w) - packed as float
    endpoint_b: [f32; 4],
    /// Color at endpoint A (RGB), w unused
    color_a: [f32; 4],
    /// Color at endpoint B (RGB), w unused
    color_b: [f32; 4],
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
    ) -> Self {
        // Create sphere pipeline
        let sphere_buffer = TypedBuffer::new_with_data(
            &context.device,
            "Ball-and-Stick Sphere Buffer",
            &[SphereInstance::zeroed()],
            wgpu::BufferUsages::STORAGE,
        );

        let sphere_bind_group_layout = Self::create_storage_layout(&context.device, "Sphere");
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
        );

        // Create bond pipeline (reuses capsule_impostor.wgsl)
        let bond_buffer = TypedBuffer::new_with_data(
            &context.device,
            "Ball-and-Stick Bond Buffer",
            &[CapsuleInstance::zeroed()],
            wgpu::BufferUsages::STORAGE,
        );

        let bond_bind_group_layout = Self::create_storage_layout(&context.device, "Bond");
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
        );

        // Create picking buffer (degenerate capsules)
        let picking_buffer = TypedBuffer::new_with_data(
            &context.device,
            "Ball-and-Stick Picking Buffer",
            &[CapsuleInstance::zeroed()],
            wgpu::BufferUsages::STORAGE,
        );
        let picking_bind_group_layout = Self::create_storage_layout(&context.device, "BnS Picking");
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

    fn create_storage_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("Ball-and-Stick {} Layout", label)),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
    ) -> wgpu::RenderPipeline {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("../assets/shaders/sphere_impostor.wgsl"));

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Sphere Impostor Pipeline Layout"),
                    bind_group_layouts: &[
                        bind_group_layout,
                        camera_layout,
                        lighting_layout,
                        selection_layout,
                    ],
                    immediate_size: 0,
                });

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
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: context.config.format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
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
    ) -> wgpu::RenderPipeline {
        // Reuse capsule_impostor.wgsl for bonds
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("../assets/shaders/capsule_impostor.wgsl"));

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Ball-and-Stick Bond Pipeline Layout"),
                    bind_group_layouts: &[
                        bind_group_layout,
                        camera_layout,
                        lighting_layout,
                        selection_layout,
                    ],
                    immediate_size: 0,
                });

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
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: context.config.format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
    }

    /// Regenerate all instances from entity data.
    pub fn update_from_entities(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        entities: &[MoleculeEntity],
        show_waters: bool,
    ) {
        let mut sphere_instances = Vec::new();
        let mut bond_instances = Vec::new();
        let mut picking_instances = Vec::new();

        for entity in entities {
            match entity.molecule_type {
                MoleculeType::Ligand => {
                    self.generate_ligand_instances(
                        &entity.coords,
                        entity.entity_id,
                        &mut sphere_instances,
                        &mut bond_instances,
                        &mut picking_instances,
                    );
                }
                MoleculeType::Ion => {
                    self.generate_ion_instances(
                        &entity.coords,
                        entity.entity_id,
                        &mut sphere_instances,
                        &mut picking_instances,
                    );
                }
                MoleculeType::Water if show_waters => {
                    self.generate_water_instances(
                        &entity.coords,
                        entity.entity_id,
                        &mut sphere_instances,
                        &mut bond_instances,
                        &mut picking_instances,
                    );
                }
                // Skip Protein, DNA, RNA (handled by other renderers), and Water if hidden
                _ => {}
            }
        }

        // Update sphere buffer
        let sphere_reallocated = if sphere_instances.is_empty() {
            self.sphere_buffer
                .write(device, queue, &[SphereInstance::zeroed()])
        } else {
            self.sphere_buffer.write(device, queue, &sphere_instances)
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
            self.bond_buffer.write(device, queue, &bond_instances)
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
            self.picking_buffer
                .write(device, queue, &[CapsuleInstance::zeroed()])
        } else {
            self.picking_buffer
                .write(device, queue, &picking_instances)
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

    fn generate_ligand_instances(
        &self,
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

        // Generate atom spheres
        for (i, pos) in positions.iter().enumerate() {
            let elem = coords
                .elements
                .get(i)
                .copied()
                .unwrap_or(Element::Unknown);
            let color = elem.cpk_color();
            let radius = elem.vdw_radius() * BALL_RADIUS_SCALE;
            // Use a large residue_idx offset so picking doesn't conflict with protein
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

        // Infer and generate bonds
        let inferred_bonds = infer_bonds(coords, DEFAULT_TOLERANCE);
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
            let color_a = elem_a.cpk_color();
            let color_b = elem_b.cpk_color();
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
                    // Single bond (or triple/aromatic rendered as single for now)
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

    fn generate_ion_instances(
        &self,
        coords: &foldit_conv::coords::Coords,
        _entity_id: u32,
        spheres: &mut Vec<SphereInstance>,
        picking: &mut Vec<CapsuleInstance>,
    ) {
        for (i, atom) in coords.atoms.iter().enumerate() {
            let elem = coords
                .elements
                .get(i)
                .copied()
                .unwrap_or(Element::Unknown);
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
        &self,
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
            let elem = coords
                .elements
                .get(i)
                .copied()
                .unwrap_or(Element::Unknown);

            // Only render oxygens as visible spheres, skip hydrogens in water
            if elem == Element::O || elem == Element::Unknown {
                let pick_id = 100000 + i as u32;

                spheres.push(SphereInstance {
                    center: [pos.x, pos.y, pos.z, WATER_RADIUS],
                    color: [water_color[0], water_color[1], water_color[2], pick_id as f32],
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
                    color_a: [water_color[0], water_color[1], water_color[2], 0.0],
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

    /// Draw both spheres and bonds in a single render pass.
    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        lighting_bind_group: &'a wgpu::BindGroup,
        selection_bind_group: &'a wgpu::BindGroup,
    ) {
        // Draw atom spheres
        if self.sphere_count > 0 {
            render_pass.set_pipeline(&self.sphere_pipeline);
            render_pass.set_bind_group(0, &self.sphere_bind_group, &[]);
            render_pass.set_bind_group(1, camera_bind_group, &[]);
            render_pass.set_bind_group(2, lighting_bind_group, &[]);
            render_pass.set_bind_group(3, selection_bind_group, &[]);
            render_pass.draw(0..6, 0..self.sphere_count);
        }

        // Draw bonds
        if self.bond_count > 0 {
            render_pass.set_pipeline(&self.bond_pipeline);
            render_pass.set_bind_group(0, &self.bond_bind_group, &[]);
            render_pass.set_bind_group(1, camera_bind_group, &[]);
            render_pass.set_bind_group(2, lighting_bind_group, &[]);
            render_pass.set_bind_group(3, selection_bind_group, &[]);
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
    pub fn collect_positions(entities: &[MoleculeEntity], show_waters: bool) -> Vec<Vec3> {
        let mut positions = Vec::new();
        for entity in entities {
            match entity.molecule_type {
                MoleculeType::Ligand | MoleculeType::Ion => {
                    for atom in &entity.coords.atoms {
                        positions.push(Vec3::new(atom.x, atom.y, atom.z));
                    }
                }
                MoleculeType::Water if show_waters => {
                    for atom in &entity.coords.atoms {
                        positions.push(Vec3::new(atom.x, atom.y, atom.z));
                    }
                }
                _ => {}
            }
        }
        positions
    }
}

/// Find any vector perpendicular to the given vector.
fn find_perpendicular(v: Vec3) -> Vec3 {
    if v.length_squared() < 1e-8 {
        return Vec3::X;
    }
    let candidate = if v.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    v.cross(candidate).normalize()
}
