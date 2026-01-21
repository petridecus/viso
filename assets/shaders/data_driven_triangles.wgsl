struct VertexOutput {
	@builtin(position) pos: vec4<f32>,
	@location(0) color: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> x_offsets: array<f32>;
@group(0) @binding(1) var<storage, read> y_offsets: array<f32>;

@vertex
fn vs_main(
	@builtin(vertex_index) vidx: u32,
	@builtin(instance_index) iidx: u32
) -> VertexOutput {
	let x = f32(1 - i32(vidx)) * 0.1;
	let y = f32(i32(vidx & 1u) * 2 - 1) * 0.1;

	let dx = x_offsets[iidx];
	let dy = y_offsets[iidx];

	return VertexOutput(
		vec4<f32>(dx + x, y + dy, 0.0, 1.0),
		vec4<f32>(0.2, 0.6, 1.0, 1.0)
	);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	return in.color;
}
