struct VertexOutput {
	@builtin(position) pos: vec4<f32>,
	@location(0) color_factor: f32,
};

@vertex
fn vs_main(
	@builtin(vertex_index) vidx: u32,
	@builtin(instance_index) iidx: u32
) -> VertexOutput {
	let x = f32(1 - i32(vidx)) * 0.1;
	let y = f32(i32(vidx & 1u) * 2 - 1) * 0.1;

	let dx = f32(iidx % 10u) * 0.2 - 0.9;
	let dy = f32(iidx / 10u) * 0.2 - 0.9;

	return VertexOutput(vec4<f32>(x + dx, y + dy, 0.0, 1.0), y + 0.5);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let color_top = vec4<f32>(1.0, 0.5, 0.2, 1.0);
	let color_bottom = vec4<f32>(0.1, 0.1, 0.5, 1.0);

	return mix(color_bottom, color_top, in.color_factor);
}
