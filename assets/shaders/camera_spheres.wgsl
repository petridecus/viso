struct CameraUniform {
	view_proj: mat4x4<f32>,
	position: vec3<f32>,
	aspect: f32,
	fovy: f32,
	selected_atom_index: i32,
};

struct VertexOutput {
	@builtin(position) pos: vec4<f32>,
	@location(0) uv: vec2<f32>,
	@location(1) instance_index: u32,
};

struct FragmentOutput {
	@location(0) color: vec4<f32>,
};

const HIGHLIGHT_MIX: f32 = 0.3;

@group(0) @binding(0) var<storage, read> dxs: array<f32>;
@group(0) @binding(1) var<storage, read> dys: array<f32>;
@group(0) @binding(2) var<storage, read> dzs: array<f32>;

@group(1) @binding(0) var<uniform> camera: CameraUniform;

@vertex
fn vs_main(
	@builtin(vertex_index) vidx: u32,
	@builtin(instance_index) iidx: u32
) -> VertexOutput {
	let quad = array<vec2<f32>, 6>(
		vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
		vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
	);

	let center = vec4<f32>(dxs[iidx], dys[iidx], dzs[iidx], 1.0);
	let clip_center = camera.view_proj * center;
	
	// Clip spheres behind camera (w <= 0 in clip space)
	if (clip_center.w <= 0.0) {
		return VertexOutput(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec2<f32>(0.0), 0u);
	}
	
	let world_radius = 2.0;
	let aspect = camera.aspect;
	let uv = quad[vidx];
	
	// Convert to NDC (Normalized Device Coordinates)
	let ndc_center = clip_center / clip_center.w;
	
	// Perspective scaling: convert world radius to NDC based on distance and fovy
	let fovy_rad = camera.fovy * (3.141592653589793 / 180.0);
	let tan_half_fovy = tan(fovy_rad * 0.5);
	let depth = max(clip_center.w, 1e-6);
	let scale = 1.0 / (depth * tan_half_fovy);
	let ndc_radius = world_radius * scale;
	
	// Add offset in NDC space (screen-aligned billboarding)
	let ndc_offset = vec4<f32>(
		uv.x * ndc_radius / aspect,
		uv.y * ndc_radius,
		0.0,
		0.0
	);
	
	// Add offset and convert back to clip space
	let ndc_pos = ndc_center + ndc_offset;
	let clip_pos = ndc_pos * clip_center.w;
	
	return VertexOutput(clip_pos, uv, iidx);
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
	let d2 = dot(in.uv, in.uv);
	let dist = sqrt(d2);

	let alpha = 1.0 - smoothstep(1.0 - fwidth(dist), 1.0, dist);
	if (alpha <= 0.0) { discard; }

	let z = sqrt(max(0.0, 1.0 - d2));
	let normal = vec3<f32>(in.uv.x, in.uv.y, z);

	let light_dir = normalize(vec3<f32>(0.5, 0.5, 1.0));
	let diff = max(dot(normal, light_dir), 0.0);
	let ambient = 0.1;
	let base_color = vec3<f32>(0.2, 0.5, 0.9);
	let final_rgb_non_highlight = base_color * (diff + ambient);
	let is_selected = i32(in.instance_index) == camera.selected_atom_index;
	let final_rgb = select(final_rgb_non_highlight, mix(final_rgb_non_highlight, vec3<f32>(1.0), HIGHLIGHT_MIX), is_selected);

	return FragmentOutput(vec4<f32>(final_rgb, alpha));
}
