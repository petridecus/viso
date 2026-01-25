use glam::{Mat4, Vec3};

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub position: [f32; 3],
    pub aspect: f32,
    pub forward: [f32; 3],  // Camera forward direction for lighting
    pub fovy: f32,
    pub selected_atom_index: i32,
    pub _pad: [f32; 3],
}

impl Camera {
    pub fn build_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);

        let proj = Mat4::perspective_rh(self.fovy.to_radians(), self.aspect, self.znear, self.zfar);

        // wgpu correction matrix (maps z differently from the default, opengl)
        let correction: Mat4 = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
        ]);

        correction * proj * view
    }

    /// Get just the projection matrix (with wgpu correction) for SSAO
    pub fn build_projection(&self) -> Mat4 {
        let proj = Mat4::perspective_rh(self.fovy.to_radians(), self.aspect, self.znear, self.zfar);

        // wgpu correction matrix
        let correction: Mat4 = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
        ]);

        correction * proj
    }
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            position: [0.0; 3],
            aspect: 1.6,
            forward: [0.0, 0.0, -1.0],
            fovy: 45.0,
            selected_atom_index: -1,
            _pad: [0.0; 3],
        }
    }

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
