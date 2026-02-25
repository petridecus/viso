use glam::{Mat4, Vec3};

/// Perspective camera defined by eye position, target, and projection
/// parameters.
pub struct Camera {
    /// Eye (camera) position in world space.
    pub eye: Vec3,
    /// Look-at target position.
    pub target: Vec3,
    /// Up direction vector.
    pub up: Vec3,
    /// Viewport aspect ratio (width / height).
    pub aspect: f32,
    /// Vertical field of view in degrees.
    pub fovy: f32,
    /// Near clipping plane distance.
    pub znear: f32,
    /// Far clipping plane distance.
    pub zfar: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
/// GPU uniform buffer holding the view-projection matrix and camera metadata.
pub struct CameraUniform {
    /// Combined view-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Camera world-space position.
    pub position: [f32; 3],
    /// Viewport aspect ratio.
    pub aspect: f32,
    /// Camera forward direction for lighting.
    pub forward: [f32; 3],
    /// Vertical field of view in degrees.
    pub fovy: f32,
    /// Currently hovered residue index (-1 if none).
    pub hovered_residue: i32,
    /// Debug visualization mode (0 = off, 1 = show normals).
    pub debug_mode: u32,
    /// Padding for GPU alignment.
    pub(crate) _pad: [f32; 2],
}

impl Camera {
    /// Build the combined view-projection matrix.
    pub fn build_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        // perspective_rh already uses [0,1] depth range (wgpu/Vulkan
        // convention)
        let proj = Mat4::perspective_rh(
            self.fovy.to_radians(),
            self.aspect,
            self.znear,
            self.zfar,
        );
        proj * view
    }

    /// Get just the projection matrix for SSAO
    pub fn build_projection(&self) -> Mat4 {
        // perspective_rh already uses [0,1] depth range (wgpu/Vulkan
        // convention)
        Mat4::perspective_rh(
            self.fovy.to_radians(),
            self.aspect,
            self.znear,
            self.zfar,
        )
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraUniform {
    /// Create a new camera uniform with identity view-projection.
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            position: [0.0; 3],
            aspect: 1.6,
            forward: [0.0, 0.0, -1.0],
            fovy: 45.0,
            hovered_residue: -1,
            debug_mode: 0,
            _pad: [0.0; 2],
        }
    }

    /// Update uniform fields from the given camera's current state.
    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_matrix().to_cols_array_2d();
        self.position = camera.eye.to_array();
        self.aspect = camera.aspect;
        // Compute actual camera forward direction (from eye toward target)
        let forward = (camera.target - camera.eye).normalize();
        self.forward = forward.to_array();
        self.fovy = camera.fovy;
    }
}
