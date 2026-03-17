//! GPU per-residue color buffer with eased transitions.
//!
//! Stores per-residue colors in a storage buffer on the GPU, allowing color
//! changes (score/SS/relative mode switches, score updates) without mesh
//! rebuilds. Transitions interpolate over ~300ms with ease-out for smooth
//! visual feedback.

use web_time::Instant;

use wgpu::util::DeviceExt;

use crate::gpu::pipeline_helpers::read_only_storage_buffer;
use crate::util::easing::EasingFunction;

/// Duration of color transitions in seconds.
const TRANSITION_DURATION: f32 = 0.3;

/// Default residue color (neutral gray) used for padding and initialization.
const DEFAULT_RESIDUE_COLOR: [f32; 4] = [0.5, 0.5, 0.5, 1.0];

/// GPU storage buffer for per-residue colors with eased transitions.
///
/// Modeled after `SelectionBuffer` — owns a storage buffer, bind group layout,
/// and bind group. All renderers reference the same layout at pipeline creation
/// time and bind the same bind group at draw time.
pub struct ResidueColorBuffer {
    buffer: wgpu::Buffer,
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) bind_group: wgpu::BindGroup,
    capacity: usize,
    /// Current displayed colors on the GPU (rgba, a=1.0).
    current_colors: Vec<[f32; 4]>,
    /// Colors at the start of the current transition.
    start_colors: Vec<[f32; 4]>,
    /// Target colors for the current transition.
    target_colors: Vec<[f32; 4]>,
    /// When the current transition started (None = not transitioning).
    transition_start: Option<Instant>,
    /// Easing function for transitions.
    easing: EasingFunction,
}

impl ResidueColorBuffer {
    /// Initializes all residues to default gray `[0.5, 0.5, 0.5]`.
    pub fn new(device: &wgpu::Device, max_residues: usize) -> Self {
        let capacity = max_residues.max(1);
        let data: Vec<[f32; 4]> = vec![DEFAULT_RESIDUE_COLOR; capacity];

        let layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Residue Color Bind Group Layout"),
                entries: &[read_only_storage_buffer(0)],
            });

        let (buffer, bind_group) =
            create_buffer_and_bind_group(device, &layout, &data);

        Self {
            buffer,
            layout,
            bind_group,
            capacity,
            current_colors: data.clone(),
            start_colors: data.clone(),
            target_colors: data,
            transition_start: None,
            easing: EasingFunction::DEFAULT,
        }
    }

    /// Snap to colors immediately (no transition).
    ///
    /// Used on structure load or when residue count changes.
    pub fn set_colors_immediate(
        &mut self,
        queue: &wgpu::Queue,
        colors: &[[f32; 3]],
    ) {
        let mut padded: Vec<[f32; 4]> =
            colors.iter().map(|c| [c[0], c[1], c[2], 1.0]).collect();
        padded.resize(self.capacity, DEFAULT_RESIDUE_COLOR);

        self.current_colors.clone_from(&padded);
        self.start_colors.clone_from(&padded);
        self.target_colors = padded;
        self.transition_start = None;

        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&self.current_colors),
        );
    }

    /// Set target colors and begin a smooth transition.
    ///
    /// Captures current displayed colors as start, sets target, starts timer.
    /// If called during an active transition, the current mid-lerp colors
    /// become the new start (smooth preemption).
    pub fn set_target_colors(&mut self, colors: &[[f32; 3]]) {
        self.start_colors = self.current_colors.clone();

        let data: Vec<[f32; 4]> =
            colors.iter().map(|c| [c[0], c[1], c[2], 1.0]).collect();
        let mut padded = data;
        padded.resize(self.capacity, DEFAULT_RESIDUE_COLOR);
        self.target_colors = padded;

        self.transition_start = Some(Instant::now());
    }

    /// Update the color buffer for the current frame.
    ///
    /// If transitioning: lerps start→target with easing, writes GPU buffer.
    /// Returns `true` if still transitioning (caller should request redraw).
    pub fn update(&mut self, queue: &wgpu::Queue) -> bool {
        let Some(start) = self.transition_start else {
            return false;
        };

        let elapsed = start.elapsed().as_secs_f32();
        let raw_t = (elapsed / TRANSITION_DURATION).min(1.0);
        let t = self.easing.evaluate(raw_t);

        for i in 0..self.current_colors.len() {
            for c in 0..3 {
                self.current_colors[i][c] = self.start_colors[i][c]
                    + (self.target_colors[i][c] - self.start_colors[i][c]) * t;
            }
            self.current_colors[i][3] = 1.0;
        }

        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&self.current_colors),
        );

        if raw_t >= 1.0 {
            self.transition_start = None;
            return false;
        }

        true
    }

    /// Ensure the buffer has capacity for at least `required` residues.
    ///
    /// Recreates the buffer and bind_group if current capacity is insufficient.
    pub fn ensure_capacity(&mut self, device: &wgpu::Device, required: usize) {
        if required <= self.capacity {
            return;
        }

        let data: Vec<[f32; 4]> = vec![DEFAULT_RESIDUE_COLOR; required];
        let (buffer, bind_group) =
            create_buffer_and_bind_group(device, &self.layout, &data);

        self.buffer = buffer;
        self.bind_group = bind_group;
        self.capacity = required;
        self.current_colors.clone_from(&data);
        self.start_colors.clone_from(&data);
        self.target_colors = data;
        self.transition_start = None;
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        let bytes = self.capacity * 16; // [f32; 4] = 16 bytes per residue
        vec![("Residue Color", bytes, bytes)]
    }
}

/// Create a storage buffer and bind group for residue color data.
fn create_buffer_and_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    data: &[[f32; 4]],
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Residue Color Buffer"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Residue Color Bind Group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    (buffer, bind_group)
}
