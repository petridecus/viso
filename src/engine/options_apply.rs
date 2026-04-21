//! Options application: push runtime config to GPU subsystems.

use super::VisoEngine;

impl VisoEngine {
    /// Push lighting options to the GPU uniform.
    pub(super) fn apply_lighting(&mut self) {
        self.gpu.apply_lighting(&self.options.lighting);
    }

    /// Push post-processing options to the composite pass.
    pub(super) fn apply_post_processing(&mut self) {
        self.gpu.apply_post_processing(&self.options);
    }

    /// Push camera options to the controller.
    pub(super) fn apply_camera(&mut self) {
        self.camera_controller
            .apply_camera_options(&self.options.camera);
    }

    /// Push debug options to the camera uniform.
    pub(super) fn apply_debug(&mut self) {
        self.camera_controller
            .apply_debug_options(&self.options.debug, &self.gpu.context.queue);
    }
}
