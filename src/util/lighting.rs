use crate::gpu::render_context::RenderContext;
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
///   roughness: f32            (offset 96)
///   metalness: f32            (offset 100)
///   _pad4: f32                (offset 104)
///   _pad5: f32                (offset 108)
///   Total: 112 bytes
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
    /// Surface roughness (0.05 = mirror-like, 1.0 = completely matte)
    pub roughness: f32,
    /// Surface metalness (0.0 = dielectric, 1.0 = metal)
    pub metalness: f32,
    pub _pad4: f32,
    pub _pad5: f32,
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
            // PBR material defaults
            roughness: 0.5,
            metalness: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
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

        // Generate prefiltered environment cubemap for specular IBL
        let (_prefiltered_texture, prefiltered_view) =
            generate_prefiltered_cubemap(&context.device, &context.queue);

        // Generate BRDF integration LUT for split-sum approximation
        let (_brdf_lut_texture, brdf_lut_view) = generate_brdf_lut(&context.device, &context.queue);

        let irradiance_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Irradiance Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
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
                    // Binding 1: Irradiance cubemap texture (diffuse IBL)
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
                    // Binding 2: Irradiance sampler (shared for all IBL textures)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Binding 3: Prefiltered environment cubemap (specular IBL)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Binding 4: BRDF integration LUT
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
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
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&prefiltered_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&brdf_lut_view),
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
        let rim_world =
            camera_right * rim_camera.x + camera_up * rim_camera.y + camera_forward * rim_camera.z;

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

/// Generate a prefiltered environment cubemap for specular IBL.
///
/// Each mip level represents a different roughness: mip 0 = sharp reflections,
/// higher mips = progressively blurred (rougher) reflections.
/// 64x64 base, 6 mip levels.
fn generate_prefiltered_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    let base_size = 64u32;
    let mip_count = 6u32;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Prefiltered Cubemap"),
        size: wgpu::Extent3d {
            width: base_size,
            height: base_size,
            depth_or_array_layers: 6,
        },
        mip_level_count: mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for mip in 0..mip_count {
        let mip_size = base_size >> mip;
        let roughness = mip as f32 / (mip_count - 1) as f32;

        for face in 0..6u32 {
            let mut data = vec![0u8; (mip_size * mip_size * 4) as usize];

            for y in 0..mip_size {
                for x in 0..mip_size {
                    let u = (x as f32 + 0.5) / mip_size as f32 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / mip_size as f32 * 2.0 - 1.0;

                    let dir = cubemap_face_direction(face, u, v);
                    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
                    let dir = [dir[0] / len, dir[1] / len, dir[2] / len];

                    // Evaluate blurred irradiance based on roughness
                    // At roughness=0, sharp environment; at roughness=1, very blurred
                    let color = evaluate_prefiltered(dir, roughness);

                    let offset = ((y * mip_size + x) * 4) as usize;
                    data[offset] = (color[0].clamp(0.0, 1.0) * 255.0) as u8;
                    data[offset + 1] = (color[1].clamp(0.0, 1.0) * 255.0) as u8;
                    data[offset + 2] = (color[2].clamp(0.0, 1.0) * 255.0) as u8;
                    data[offset + 3] = 255;
                }
            }

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip,
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
                    bytes_per_row: Some(mip_size * 4),
                    rows_per_image: Some(mip_size),
                },
                wgpu::Extent3d {
                    width: mip_size,
                    height: mip_size,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("Prefiltered Cubemap View"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    (texture, view)
}

/// Evaluate prefiltered environment color for a given direction and roughness.
///
/// At low roughness, returns a sharper version of the environment.
/// At high roughness, returns a heavily blurred/averaged version.
/// Uses the same studio gradient as the irradiance map but with a specular highlight boost.
fn evaluate_prefiltered(dir: [f32; 3], roughness: f32) -> [f32; 3] {
    // Base irradiance gradient
    let base = evaluate_irradiance(dir);

    // Add a specular highlight at low roughness: a bright spot in the upper region
    // This simulates a key light reflection in the environment
    let highlight_dir = [0.0f32, 0.7, -0.7]; // Approximate key light direction
    let dot = dir[0] * highlight_dir[0] + dir[1] * highlight_dir[1] + dir[2] * highlight_dir[2];
    let dot = dot.max(0.0);

    // Sharper highlight at low roughness, wider at high
    let exponent = 1.0 / (roughness * roughness + 0.01);
    let highlight = dot.powf(exponent.min(256.0));
    let highlight_strength = 0.4 * (1.0 - roughness);

    [
        base[0] + highlight * highlight_strength,
        base[1] + highlight * highlight_strength * 0.95,
        base[2] + highlight * highlight_strength * 0.85,
    ]
}

/// Generate a BRDF integration lookup table for the split-sum IBL approximation.
///
/// 256x256 Rg16Float texture. X axis = NdotV, Y axis = roughness.
/// R = scale, G = bias for the Fresnel term: `specular = F0 * scale + bias`
fn generate_brdf_lut(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = 256u32;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("BRDF LUT"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Generate BRDF LUT on CPU using importance sampling
    let sample_count = 256u32;
    let mut data = vec![0u8; (size * size * 4) as usize]; // 4 bytes per texel (2x f16)

    for y in 0..size {
        for x in 0..size {
            let ndot_v = (x as f32 + 0.5) / size as f32;
            let ndot_v = ndot_v.max(0.001); // Avoid division by zero
            let roughness = (y as f32 + 0.5) / size as f32;
            let roughness = roughness.max(0.01);

            let (scale, bias) = integrate_brdf(ndot_v, roughness, sample_count);

            let offset = ((y * size + x) * 4) as usize;
            let scale_f16 = half::f16::from_f32(scale);
            let bias_f16 = half::f16::from_f32(bias);
            data[offset..offset + 2].copy_from_slice(&scale_f16.to_le_bytes());
            data[offset + 2..offset + 4].copy_from_slice(&bias_f16.to_le_bytes());
        }
    }

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
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

    let view = texture.create_view(&Default::default());
    (texture, view)
}

/// Integrate the BRDF for given NdotV and roughness using importance sampling.
/// Returns (scale, bias) for the split-sum approximation.
fn integrate_brdf(ndot_v: f32, roughness: f32, sample_count: u32) -> (f32, f32) {
    use std::f32::consts::PI;

    let v = [
        (1.0 - ndot_v * ndot_v).sqrt(), // sin
        0.0,
        ndot_v, // cos
    ];

    let mut a = 0.0f32;
    let mut b = 0.0f32;

    let alpha = roughness * roughness;

    for i in 0..sample_count {
        // Hammersley sequence for quasi-random sampling
        let xi = hammersley(i, sample_count);

        // Importance sample the GGX distribution
        let phi = 2.0 * PI * xi[0];
        let cos_theta = ((1.0 - xi[1]) / (1.0 + (alpha * alpha - 1.0) * xi[1])).sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(0.0);

        // Halfway vector in tangent space
        let h = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];

        // Light vector = reflect view around halfway
        let v_dot_h = v[0] * h[0] + v[1] * h[1] + v[2] * h[2];
        let l = [
            2.0 * v_dot_h * h[0] - v[0],
            2.0 * v_dot_h * h[1] - v[1],
            2.0 * v_dot_h * h[2] - v[2],
        ];

        let n_dot_l = l[2].max(0.0);
        let n_dot_h = cos_theta.max(0.0);
        let v_dot_h = v_dot_h.clamp(0.0, 1.0);

        if n_dot_l > 0.0 {
            let g = geometry_smith_ibl(ndot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / (n_dot_h * ndot_v).max(0.0001);
            let fc = (1.0 - v_dot_h).powi(5);

            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }

    (a / sample_count as f32, b / sample_count as f32)
}

/// Smith's geometry function for IBL (uses k = alpha^2 / 2)
fn geometry_smith_ibl(ndot_v: f32, ndot_l: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let k = a / 2.0;
    let ggx_v = ndot_v / (ndot_v * (1.0 - k) + k);
    let ggx_l = ndot_l / (ndot_l * (1.0 - k) + k);
    ggx_v * ggx_l
}

/// Hammersley quasi-random sequence (2D)
fn hammersley(i: u32, n: u32) -> [f32; 2] {
    [i as f32 / n as f32, radical_inverse_vdc(i)]
}

/// Van der Corput radical inverse (base 2)
fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    bits as f32 * 2.328_306_4e-10 // 1.0 / 0x100000000
}
