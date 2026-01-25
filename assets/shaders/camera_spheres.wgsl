struct CameraUniform {
	view_proj: mat4x4<f32>,
	position: vec3<f32>,
	aspect: f32,
	forward: vec3<f32>,
	fovy: f32,
	selected_atom_index: i32,
};

struct LightingUniform {
	light1_dir: vec3<f32>,
	_pad1: f32,
	light2_dir: vec3<f32>,
	_pad2: f32,
	light1_intensity: f32,
	light2_intensity: f32,
	ambient: f32,
	specular_intensity: f32,
	shininess: f32,
	_pad3: vec3<f32>,
};

struct VertexOutput {
	@builtin(position) pos: vec4<f32>,
	@location(0) uv: vec2<f32>,
	@location(1) instance_index: u32,
	@location(2) is_hydrophobic: f32,
	@location(3) world_center: vec3<f32>,
};

const HIGHLIGHT_MIX: f32 = 0.3;
const SPHERE_RADIUS: f32 = 0.3;

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> camera: CameraUniform;
@group(2) @binding(0) var<uniform> lighting: LightingUniform;

@vertex
fn vs_main(
	@builtin(vertex_index) vidx: u32,
	@builtin(instance_index) iidx: u32
) -> VertexOutput {
	let quad = array<vec2<f32>, 6>(
		vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
		vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
	);

	let atom_data = positions[iidx];
	let center = atom_data.xyz;
	let is_hydrophobic = atom_data.w;
	let clip_center = camera.view_proj * vec4<f32>(center, 1.0);

	if (clip_center.w <= 0.0) {
		return VertexOutput(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec2<f32>(0.0), 0u, 0.0, vec3<f32>(0.0));
	}

	let uv = quad[vidx];
	let ndc_center = clip_center / clip_center.w;

	let fovy_rad = camera.fovy * (3.141592653589793 / 180.0);
	let tan_half_fovy = tan(fovy_rad * 0.5);
	let depth = max(clip_center.w, 1e-6);
	let scale = 1.0 / (depth * tan_half_fovy);
	let ndc_radius = SPHERE_RADIUS * scale;

	let ndc_offset = vec4<f32>(
		uv.x * ndc_radius / camera.aspect,
		uv.y * ndc_radius,
		0.0,
		0.0
	);

	let ndc_pos = ndc_center + ndc_offset;
	let clip_pos = ndc_pos * clip_center.w;

	return VertexOutput(clip_pos, uv, iidx, is_hydrophobic, center);
}

struct FragOut {
	@builtin(frag_depth) depth: f32,
	@location(0) color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
	let d2 = dot(in.uv, in.uv);
	if (d2 > 1.0) { discard; }

	// Screen-space hemisphere normal (this approach was working for lighting)
	let local_z = sqrt(1.0 - d2);
	let local_normal = vec3<f32>(in.uv.x, in.uv.y, local_z);

	// Build camera basis - IDENTICAL to cylinder shader
	let cam_forward = camera.forward;
	let world_up = vec3<f32>(0.0, 1.0, 0.0);
	var cam_right = cross(cam_forward, world_up);
	if (length(cam_right) < 0.001) {
		cam_right = vec3<f32>(1.0, 0.0, 0.0);
	} else {
		cam_right = normalize(cam_right);
	}
	let cam_up = normalize(cross(cam_right, cam_forward));

	// Transform screen-space normal to world space
	let normal = normalize(
		cam_right * local_normal.x +
		cam_up * local_normal.y +
		cam_forward * local_normal.z
	);

	// Approximate world position for view_dir and fog
	let world_pos = in.world_center + normal * SPHERE_RADIUS;
	let view_dir = normalize(camera.position - world_pos);

	// Lighting - IDENTICAL to cylinder shader
	let key_light = normalize(cam_forward + cam_right * 0.4 + cam_up * 0.6);
	let fill_light = normalize(cam_forward - cam_right * 0.3 + cam_up * 0.4);

	// Diffuse
	let key_diff = max(dot(normal, key_light), 0.0) * 0.8;
	let fill_diff = max(dot(normal, fill_light), 0.0) * 0.3;

	// Specular (Blinn-Phong)
	let half_vec = normalize(key_light + view_dir);
	let specular = pow(max(dot(normal, half_vec), 0.0), 64.0) * 0.5;

	// Ambient
	let ambient = 0.1;
	let total_light = ambient + key_diff + fill_diff;

	// Color
	let hydrophobic_color = vec3<f32>(0.3, 0.5, 0.9);
	let hydrophilic_color = vec3<f32>(0.95, 0.6, 0.2);
	let base_color = mix(hydrophilic_color, hydrophobic_color, in.is_hydrophobic);

	// Depth fog
	let world_depth = length(camera.position - world_pos);
	let fog_start = 100.0;
	let fog_end = 500.0;
	let fog_factor = clamp((fog_end - world_depth) / (fog_end - fog_start), 0.0, 1.0);

	let lit_color = base_color * total_light + vec3<f32>(specular);
	let final_rgb_non_highlight = lit_color * fog_factor;

	let is_selected = i32(in.instance_index) == camera.selected_atom_index;
	let final_rgb = select(final_rgb_non_highlight, mix(final_rgb_non_highlight, vec3<f32>(1.0), HIGHLIGHT_MIX), is_selected);

	// Push spheres back in depth so cylinders win at joints
	var out: FragOut;
	out.depth = in.pos.z + 0.005;
	out.color = vec4<f32>(final_rgb, 1.0);
	return out;
}
