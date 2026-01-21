struct VertexOutput {
	@builtin(position) pos: vec4<f32>,
	@location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> dxs: array<f32>;
@group(0) @binding(1) var<storage, read> dys: array<f32>;
@group(0) @binding(2) var<storage, read> dzs: array<f32>;

@vertex
fn vs_main(
	@builtin(vertex_index) vidx: u32,
	@builtin(instance_index) iidx: u32
) -> VertexOutput {
	// this creates a set of 'unit triangles' that we later
	// shave a circle out of using a radius trick in fs_main
	let triangles = array<vec2<f32>, 6>(
		vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
		vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
	);

	let dx = dxs[iidx];
	let dy = dys[iidx];
	let dz = dzs[iidx];

	let radius = 0.01;
	let uv = triangles[vidx];

	// perspective scale, higher z = further away
	let ps = 1.0 / dz;

	// aspect ratio
	let ar = 1.6;

	return VertexOutput(
		vec4<f32>(
			dx + (uv.x * radius * ps / ar), 
			dy + (uv.y * radius * ps), 
			dz, 1.0
		), 
		uv
	);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let d2 = dot(in.uv, in.uv);
	let dist = sqrt(d2);

	let alpha = 1.0 - smoothstep(1.0 - fwidth(dist), 1.0, dist);

	if (alpha <= 0.0) { discard; }

	let z = sqrt(1.0 - d2);
	let normal = vec3<f32>(in.uv.x, in.uv.y, z);

	let light_dir = normalize(vec3<f32>(0.5, 0.5, 1.0));
	let diff = max(dot(normal, light_dir), 0.0);

	let color = vec3<f32>(0.2, 0.5, 0.9) * (diff + 0.1);

	return vec4<f32>(color, 1.0);
}
