//! Per-residue picking via ray-sphere intersection
//!
//! Uses CA atom positions as proxy spheres for each residue.
//! Supports hover detection on mouse move and selection on mouse click.

use crate::camera::core::Camera;
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

/// Picking radius for CA spheres (Angstroms)
const PICK_RADIUS: f32 = 2.5;

/// Selection buffer for GPU - stores selection state as a bit array
pub struct SelectionBuffer {
    buffer: wgpu::Buffer,
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    /// Number of residues (for sizing)
    capacity: usize,
}

impl SelectionBuffer {
    pub fn new(device: &wgpu::Device, max_residues: usize) -> Self {
        // Round up to multiple of 32 bits
        let num_words = (max_residues + 31) / 32;
        let data = vec![0u32; num_words.max(1)];
        
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Selection Buffer"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Selection Bind Group Layout"),
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
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Selection Bind Group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        
        Self {
            buffer,
            layout,
            bind_group,
            capacity: max_residues,
        }
    }
    
    /// Update selection state from a list of selected residue indices
    pub fn update(&self, queue: &wgpu::Queue, selected_residues: &[i32]) {
        let num_words = (self.capacity + 31) / 32;
        let mut data = vec![0u32; num_words.max(1)];
        
        for &idx in selected_residues {
            if idx >= 0 && (idx as usize) < self.capacity {
                let word_idx = idx as usize / 32;
                let bit_idx = idx as usize % 32;
                data[word_idx] |= 1u32 << bit_idx;
            }
        }
        
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&data));
    }
}

/// Manages residue picking state
pub struct Picking {
    /// CA positions for each residue (flattened across all chains)
    ca_positions: Vec<Vec3>,
    
    /// Currently hovered residue index (-1 if none)
    pub hovered_residue: i32,
    
    /// Currently selected residue indices
    pub selected_residues: Vec<i32>,
}

impl Picking {
    pub fn new() -> Self {
        Self {
            ca_positions: Vec::new(),
            hovered_residue: -1,
            selected_residues: Vec::new(),
        }
    }
    
    /// Update picking data from backbone residue chains
    pub fn update_from_residue_chains(&mut self, chains: &[crate::protein_data::BackboneChain]) {
        self.ca_positions.clear();
        for chain in chains {
            for residue in &chain.residues {
                self.ca_positions.push(residue.ca_pos);
            }
        }
        if !self.ca_positions.is_empty() {
            let center: Vec3 = self.ca_positions.iter().copied().sum::<Vec3>() / self.ca_positions.len() as f32;
            eprintln!("Picking: loaded {} CA positions from {} chains, center at {:?}",
                self.ca_positions.len(), chains.len(), center);
        }
    }
    
    /// Update picking data from legacy backbone chains (N, CA, C format)
    pub fn update_from_backbone_chains(&mut self, chains: &[Vec<Vec3>]) {
        self.ca_positions.clear();
        for chain in chains {
            // CA positions are every 3rd atom starting at index 1 (N=0, CA=1, C=2)
            for (i, &pos) in chain.iter().enumerate() {
                if i % 3 == 1 {
                    self.ca_positions.push(pos);
                }
            }
        }
        eprintln!("Picking: loaded {} CA positions from {} backbone chains", self.ca_positions.len(), chains.len());
    }
    
    /// Cast a ray from screen coordinates and return the hit residue index (or -1)
    pub fn pick(
        &self,
        screen_x: f32,
        screen_y: f32,
        screen_width: f32,
        screen_height: f32,
        view_proj: Mat4,
        _camera_pos: Vec3,  // Unused, we compute ray origin from unprojection
    ) -> i32 {
        // Early out if no positions to pick
        if self.ca_positions.is_empty() {
            return -1;
        }

        let (ray_origin, ray_dir) = self.screen_to_ray(
            screen_x, screen_y,
            screen_width, screen_height,
            view_proj,
        );

        let result = self.ray_pick(ray_origin, ray_dir);

        // Debug: log pick attempts
        static PICK_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let count = PICK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count == 0 {
            // Log first pick attempt with diagnostic info
            let center: Vec3 = self.ca_positions.iter().copied().sum::<Vec3>() / self.ca_positions.len() as f32;
            eprintln!("First pick: screen({}, {}) size({}, {})", screen_x, screen_y, screen_width, screen_height);
            eprintln!("  ray origin {:?}, ray dir {:?}", ray_origin, ray_dir);
            eprintln!("  protein center {:?}, {} residues", center, self.ca_positions.len());
        }
        if result >= 0 {
            eprintln!("Pick hit residue {} (of {} total)", result, self.ca_positions.len());
        }

        result
    }

    /// Convert screen coordinates to a world-space ray (origin and direction)
    fn screen_to_ray(
        &self,
        screen_x: f32,
        screen_y: f32,
        screen_width: f32,
        screen_height: f32,
        view_proj: Mat4,
    ) -> (Vec3, Vec3) {
        // Convert to NDC (-1 to 1), y flipped for screen coordinates
        let ndc_x = (screen_x / screen_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_y / screen_height) * 2.0;

        // Invert view-projection matrix
        let inv_view_proj = view_proj.inverse();

        // Unproject near and far points (wgpu uses 0-1 depth range after correction)
        let ndc_near = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let ndc_far = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let world_near = inv_view_proj * ndc_near;
        let world_far = inv_view_proj * ndc_far;

        // Perspective divide
        let ray_origin = world_near.truncate() / world_near.w;
        let world_far = world_far.truncate() / world_far.w;

        let ray_dir = (world_far - ray_origin).normalize();

        (ray_origin, ray_dir)
    }
    
    /// Perform ray-sphere intersection against all CA positions
    fn ray_pick(&self, ray_origin: Vec3, ray_dir: Vec3) -> i32 {
        let mut closest_t = f32::INFINITY;
        let mut closest_idx: i32 = -1;
        
        for (i, &pos) in self.ca_positions.iter().enumerate() {
            if let Some(t) = ray_sphere_intersect(ray_origin, ray_dir, pos, PICK_RADIUS) {
                if t > 0.0 && t < closest_t {
                    closest_t = t;
                    closest_idx = i as i32;
                }
            }
        }
        
        closest_idx
    }
    
    /// Update hover state based on screen position
    pub fn update_hover(
        &mut self,
        screen_x: f32,
        screen_y: f32,
        screen_width: f32,
        screen_height: f32,
        view_proj: Mat4,
        camera_pos: Vec3,
    ) {
        self.hovered_residue = self.pick(
            screen_x, screen_y,
            screen_width, screen_height,
            view_proj,
            camera_pos,
        );
    }
    
    /// Handle click for selection
    /// Returns true if selection changed
    pub fn handle_click(
        &mut self,
        screen_x: f32,
        screen_y: f32,
        screen_width: f32,
        screen_height: f32,
        view_proj: Mat4,
        camera_pos: Vec3,
        shift_held: bool,
    ) -> bool {
        let hit = self.pick(
            screen_x, screen_y,
            screen_width, screen_height,
            view_proj,
            camera_pos,
        );
        
        if hit < 0 {
            // Clicked on empty space - clear selection
            if !self.selected_residues.is_empty() {
                self.selected_residues.clear();
                return true;
            }
            return false;
        }
        
        if shift_held {
            // Shift-click: toggle selection
            if let Some(pos) = self.selected_residues.iter().position(|&r| r == hit) {
                self.selected_residues.remove(pos);
            } else {
                self.selected_residues.push(hit);
            }
        } else {
            // Regular click: replace selection
            self.selected_residues.clear();
            self.selected_residues.push(hit);
        }
        
        true
    }
    
    /// Clear all selection
    pub fn clear_selection(&mut self) {
        self.selected_residues.clear();
        self.hovered_residue = -1;
    }

    /// Check if a residue is selected
    pub fn is_selected(&self, residue_idx: i32) -> bool {
        self.selected_residues.contains(&residue_idx)
    }

    /// Update hover state using camera directly (more reliable than matrix inversion)
    pub fn update_hover_from_camera(
        &mut self,
        screen_x: f32,
        screen_y: f32,
        screen_width: f32,
        screen_height: f32,
        camera: &Camera,
    ) {
        self.hovered_residue = self.pick_from_camera(screen_x, screen_y, screen_width, screen_height, camera);
    }

    /// Handle click using camera directly
    pub fn handle_click_from_camera(
        &mut self,
        screen_x: f32,
        screen_y: f32,
        screen_width: f32,
        screen_height: f32,
        camera: &Camera,
        shift_held: bool,
    ) -> bool {
        let hit = self.pick_from_camera(screen_x, screen_y, screen_width, screen_height, camera);

        if hit < 0 {
            if !self.selected_residues.is_empty() {
                self.selected_residues.clear();
                return true;
            }
            return false;
        }

        if shift_held {
            if let Some(pos) = self.selected_residues.iter().position(|&r| r == hit) {
                self.selected_residues.remove(pos);
            } else {
                self.selected_residues.push(hit);
            }
        } else {
            self.selected_residues.clear();
            self.selected_residues.push(hit);
        }

        true
    }

    /// Pick using camera parameters directly (avoids matrix inversion issues)
    fn pick_from_camera(
        &self,
        screen_x: f32,
        screen_y: f32,
        screen_width: f32,
        screen_height: f32,
        camera: &Camera,
    ) -> i32 {
        if self.ca_positions.is_empty() {
            return -1;
        }

        // Compute ray from camera parameters
        let ray_origin = camera.eye;

        // Camera basis vectors
        let forward = (camera.target - camera.eye).normalize();
        let right = forward.cross(camera.up).normalize();
        let up = right.cross(forward);

        // Convert screen coords to normalized device coords (-1 to 1)
        let ndc_x = (screen_x / screen_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_y / screen_height) * 2.0;

        // Compute ray direction using FOV and aspect
        let half_fov = (camera.fovy / 2.0).to_radians();
        let tan_fov = half_fov.tan();

        // Scale by aspect ratio and FOV
        let ray_x = ndc_x * camera.aspect * tan_fov;
        let ray_y = ndc_y * tan_fov;

        // Ray direction in world space
        let ray_dir = (forward + right * ray_x + up * ray_y).normalize();

        // Debug first pick
        static PICK_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let count = PICK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count == 0 {
            let center: Vec3 = self.ca_positions.iter().copied().sum::<Vec3>() / self.ca_positions.len() as f32;
            eprintln!("First pick: screen({:.0}, {:.0}) size({:.0}, {:.0})", screen_x, screen_y, screen_width, screen_height);
            eprintln!("  camera eye {:?}, target {:?}", camera.eye, camera.target);
            eprintln!("  ray origin {:?}, ray dir {:?}", ray_origin, ray_dir);
            eprintln!("  protein center {:?}, {} residues", center, self.ca_positions.len());
        }

        let result = self.ray_pick(ray_origin, ray_dir);
        if result >= 0 {
            eprintln!("Pick hit residue {} (of {} total)", result, self.ca_positions.len());
        }

        result
    }
    
    /// Get the CA position for a residue index
    pub fn get_ca_position(&self, residue_idx: i32) -> Option<Vec3> {
        if residue_idx >= 0 && (residue_idx as usize) < self.ca_positions.len() {
            Some(self.ca_positions[residue_idx as usize])
        } else {
            None
        }
    }
    
    /// Get the number of residues
    pub fn residue_count(&self) -> usize {
        self.ca_positions.len()
    }
}

/// Ray-sphere intersection test
/// Returns the distance along the ray to the first intersection, or None if no hit
fn ray_sphere_intersect(ray_origin: Vec3, ray_dir: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let oc = ray_origin - center;
    let a = ray_dir.dot(ray_dir);
    let b = 2.0 * oc.dot(ray_dir);
    let c = oc.dot(oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    
    if discriminant < 0.0 {
        return None;
    }
    
    let t = (-b - discriminant.sqrt()) / (2.0 * a);
    if t > 0.0 {
        Some(t)
    } else {
        // Try the far intersection (we're inside the sphere)
        let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
        if t2 > 0.0 {
            Some(t2)
        } else {
            None
        }
    }
}

impl Default for Picking {
    fn default() -> Self {
        Self::new()
    }
}
