use crate::render_context::RenderContext;
use wgpu::util::DeviceExt;

/// Lighting configuration shared across all shaders
/// NOTE: Must match WGSL struct layout exactly (112 bytes)
///
/// WGSL layout (auto-padded):
///   light1_dir: vec3<f32>     (offset 0,  align 16)
///   _pad1: f32                (offset 12)
///   light2_dir: vec3<f32>     (offset 16, align 16)
///   _pad2: f32                (offset 28)
///   light1_intensity: f32     (offset 32)
///   light2_intensity: f32     (offset 36)
///   ambient: f32              (offset 40)
///   specular_intensity: f32   (offset 44)
///   shininess: f32            (offset 48)
///   rim_power: f32            (offset 52)
///   rim_intensity: f32        (offset 56)
///   rim_directionality: f32   (offset 60)
///   rim_color: vec3<f32>      (offset 64, align 16)
///   ibl_strength: f32         (offset 76)
///   rim_dir: vec3<f32>        (offset 80, align 16)
///   _pad3: f32                (offset 92)
///   Total: 96 bytes
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightingUniform {
    /// Primary light direction (normalized)
    pub light1_dir: [f32; 3],
    pub _pad1: f32,
    /// Secondary light direction (normalized)
    pub light2_dir: [f32; 3],
    pub _pad2: f32,
    /// Primary light intensity
    pub light1_intensity: f32,
    /// Secondary light intensity
    pub light2_intensity: f32,
    /// Ambient light intensity (used as fallback; IBL replaces this when ibl_strength > 0)
    pub ambient: f32,
    /// Specular intensity
    pub specular_intensity: f32,
    /// Specular shininess exponent
    pub shininess: f32,
    /// Rim edge falloff power (higher = tighter edge glow)
    pub rim_power: f32,
    /// Rim edge brightness
    pub rim_intensity: f32,
    /// Rim directionality: 0.0 = pure view-dependent, 1.0 = pure directional back-light
    pub rim_directionality: f32,
    /// Rim light tint color
    pub rim_color: [f32; 3],
    /// IBL diffuse strength (0.0 = use flat ambient, 1.0 = full IBL)
    pub ibl_strength: f32,
    /// Rim back-light direction (normalized, points toward the rim light source)
    pub rim_dir: [f32; 3],
    pub _pad3: f32,
}

impl Default for LightingUniform {
    fn default() -> Self {
        Self {
            // Primary light: upper-left for strong directional contrast
            light1_dir: normalize([-0.3, 0.9, -0.3]),
            _pad1: 0.0,
            // Secondary light: upper-right-front for fill
            light2_dir: normalize([0.3, 0.6, -0.4]),
            _pad2: 0.0,
            // Stronger primary light for better depth perception
            light1_intensity: 0.7,
            light2_intensity: 0.3,
            // Slightly reduced ambient to allow darker shadows
            ambient: 0.12,
            // Broader specular for softer highlights across all geometry
            specular_intensity: 0.35,
            // Lower shininess = wider highlights that read on ribbons too
            shininess: 38.0,
            // Rim edge falloff (was fresnel_power)
            rim_power: 3.0,
            // Rim edge brightness (was fresnel_intensity)
            rim_intensity: 0.3,
            // Mostly view-dependent with slight directional bias
            rim_directionality: 0.3,
            // Warm rim tint for cinematic look
            rim_color: [1.0, 0.85, 0.7],
            // IBL enabled by default
            ibl_strength: 1.0,
            // Rim back-light: below-behind relative to camera
            rim_dir: normalize([0.0, -0.7, 0.5]),
            _pad3: 0.0,
        }
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

pub struct Lighting {
    pub uniform: LightingUniform,
    pub buffer: wgpu::Buffer,
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl Lighting {
    pub fn new(context: &RenderContext) -> Self {
        let uniform = LightingUniform::default();

        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Lighting Buffer"),
                contents: bytemuck::cast_slice(&[uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Generate procedural irradiance cubemap
        let (_irradiance_texture, irradiance_view) =
            generate_irradiance_cubemap(&context.device, &context.queue);

        let irradiance_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Irradiance Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lighting Bind Group Layout"),
                entries: &[
                    // Binding 0: Lighting uniform buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: Irradiance cubemap texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Binding 2: Irradiance sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&irradiance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&irradiance_sampler),
                    },
                ],
                label: Some("Lighting Bind Group"),
            });

        Self {
            uniform,
            buffer,
            layout,
            bind_group,
        }
    }

    pub fn update_gpu(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[self.uniform]));
    }

    /// Update light directions to follow camera (headlamp mode)
    /// Call this each frame after camera updates
    pub fn update_headlamp(
        &mut self,
        camera_right: glam::Vec3,
        camera_up: glam::Vec3,
        camera_forward: glam::Vec3,
    ) {
        // Primary light: upper-left relative to camera view
        // In camera space: (-0.3, 0.9, -0.3) = left, up, toward viewer
        // Negative z ensures surfaces facing camera receive light
        let light1_camera = glam::Vec3::new(-0.3, 0.9, -0.3).normalize();

        // Transform from camera space to world space
        let light1_world = camera_right * light1_camera.x
            + camera_up * light1_camera.y
            + camera_forward * light1_camera.z;

        self.uniform.light1_dir = light1_world.normalize().to_array();

        // Secondary fill light: upper-right relative to camera
        let light2_camera = glam::Vec3::new(0.3, 0.6, -0.4).normalize();
        let light2_world = camera_right * light2_camera.x
            + camera_up * light2_camera.y
            + camera_forward * light2_camera.z;

        self.uniform.light2_dir = light2_world.normalize().to_array();

        // Rim back-light: below-behind relative to camera
        let rim_camera = glam::Vec3::new(0.0, -0.7, 0.5).normalize();
        let rim_world = camera_right * rim_camera.x
            + camera_up * rim_camera.y
            + camera_forward * rim_camera.z;

        self.uniform.rim_dir = rim_world.normalize().to_array();
    }
}

/// Generate a procedural gradient irradiance cubemap for studio-style ambient lighting.
///
/// Produces a 32x32-per-face cubemap with:
/// - Warm white from above (simulating overhead light bounce)
/// - Cool blue from below (simulating floor/shadow bounce)
/// - Neutral gray from sides
///
/// The irradiance map is sampled by surface normal in the shader to provide
/// directionally-varying ambient light, replacing the flat ambient term.
fn generate_irradiance_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = 32u32;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Irradiance Cubemap"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Cubemap face order: +X, -X, +Y, -Y, +Z, -Z
    // For each texel, compute the world-space direction and evaluate the irradiance gradient
    for face in 0..6u32 {
        let mut data = vec![0u8; (size * size * 4) as usize]; // 4 x u8 per texel

        for y in 0..size {
            for x in 0..size {
                // Map texel to [-1, 1] range
                let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;

                // Convert face + UV to world direction
                let dir = cubemap_face_direction(face, u, v);
                let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
                let dir = [dir[0] / len, dir[1] / len, dir[2] / len];

                // Evaluate irradiance based on direction
                let irradiance = evaluate_irradiance(dir);

                // Write as Rgba8Unorm (values are in [0, 1])
                let offset = ((y * size + x) * 4) as usize;
                data[offset] = (irradiance[0].clamp(0.0, 1.0) * 255.0) as u8;
                data[offset + 1] = (irradiance[1].clamp(0.0, 1.0) * 255.0) as u8;
                data[offset + 2] = (irradiance[2].clamp(0.0, 1.0) * 255.0) as u8;
                data[offset + 3] = 255;
            }
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("Irradiance Cubemap View"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    (texture, view)
}

/// Convert cubemap face index + UV coordinates to a world-space direction.
/// Face order: +X, -X, +Y, -Y, +Z, -Z
fn cubemap_face_direction(face: u32, u: f32, v: f32) -> [f32; 3] {
    match face {
        0 => [1.0, -v, -u],  // +X
        1 => [-1.0, -v, u],  // -X
        2 => [u, 1.0, v],    // +Y
        3 => [u, -1.0, -v],  // -Y
        4 => [u, -v, 1.0],   // +Z
        5 => [-u, -v, -1.0], // -Z
        _ => [0.0, 0.0, 1.0],
    }
}

/// Evaluate studio-style irradiance for a given world-space direction.
///
/// The gradient provides:
/// - Warm white from above: (0.18, 0.16, 0.13)
/// - Cool blue from below:  (0.05, 0.06, 0.10)
/// - Neutral gray from sides: ~(0.10, 0.10, 0.10)
///
/// Values are chosen to roughly match the energy of the old flat ambient (0.12)
/// while adding directional variation for depth cues.
fn evaluate_irradiance(dir: [f32; 3]) -> [f32; 3] {
    let y = dir[1]; // Up component

    // Smooth vertical gradient: map y from [-1, 1] to [0, 1]
    let t = y * 0.5 + 0.5;
    // Smooth-step for softer transitions
    let t = t * t * (3.0 - 2.0 * t);

    // Top: warm white, Bottom: cool blue
    let top = [0.18f32, 0.16, 0.13];
    let bottom = [0.05f32, 0.06, 0.10];

    [
        bottom[0] + (top[0] - bottom[0]) * t,
        bottom[1] + (top[1] - bottom[1]) * t,
        bottom[2] + (top[2] - bottom[2]) * t,
    ]
}
