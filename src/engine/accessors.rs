//! Read-only query methods and lifecycle helpers for [`ProteinRenderEngine`].

use glam::Vec3;

use super::ProteinRenderEngine;
use crate::{options::Options, scene::Scene};

// ── Lifecycle ──

impl ProteinRenderEngine {
    /// Advance camera animation and apply any pending scene from the
    /// background processor.
    ///
    /// Call once per frame before [`render`](Self::render):
    /// ```ignore
    /// engine.update(dt);
    /// engine.render()?;
    /// ```
    pub fn update(&mut self, dt: f32) {
        let _ = self.camera_controller.update_animation(dt);
        self.apply_pending_scene();
    }

    /// Stop the background scene processor thread.
    pub fn shutdown(&mut self) {
        self.scene_processor.shutdown();
    }
}

// ── Scene access ──

impl ProteinRenderEngine {
    /// Read-only access to the scene graph.
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Mutable access to the scene graph for staged-data mutations.
    ///
    /// Mutations through this handle (add/remove entities, update coords, set
    /// focus, toggle visibility, set SS overrides) are all data operations.
    /// Call [`sync_scene_to_renderers`](Self::sync_scene_to_renderers) after
    /// mutations to push changes to the GPU.
    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }
}

// ── Query (read-only state inspection) ──

impl ProteinRenderEngine {
    /// Currently hovered residue index, or -1 if none.
    pub fn hovered_residue(&self) -> i32 {
        self.picking.hovered_residue
    }

    /// Currently selected residue indices.
    pub fn selected_residues(&self) -> &[i32] {
        &self.picking.selected_residues
    }

    /// Clear the current residue selection.
    pub fn clear_selection(&mut self) {
        self.picking.clear_selection();
    }

    /// Current screen size in physical pixels `(width, height)`.
    pub fn screen_size(&self) -> (u32, u32) {
        (self.context.config.width, self.context.config.height)
    }

    /// Current frames per second.
    pub fn fps(&self) -> f32 {
        self.frame_timing.fps()
    }

    /// Name of the currently active options preset, if any.
    pub fn active_preset(&self) -> Option<&str> {
        self.active_preset.as_deref()
    }

    /// Whether a trajectory is loaded.
    pub fn has_trajectory(&self) -> bool {
        self.trajectory_player.is_some()
    }

    /// Read-only access to the current options.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// GPU buffer sizes across all renderers.
    ///
    /// Each entry is `(label, used_bytes, allocated_bytes)`.
    pub fn gpu_buffer_stats(&self) -> Vec<(&str, usize, usize)> {
        let mut stats = Vec::new();
        stats.extend(self.backbone_renderer.buffer_info());
        stats.extend(self.sidechain_renderer.buffer_info());
        stats.extend(self.ball_and_stick_renderer.buffer_info());
        stats.extend(self.band_renderer.buffer_info());
        stats.extend(self.pull_renderer.buffer_info());
        stats.extend(self.nucleic_acid_renderer.buffer_info());
        stats.extend(self.selection_buffer.buffer_info());
        stats.extend(self.residue_color_buffer.buffer_info());
        stats
    }

    /// Unproject screen coordinates to a world-space point on a plane at
    /// the depth of `reference_point`.
    ///
    /// Uses the current screen size internally — callers do not need to
    /// pass width/height.
    pub fn unproject(
        &self,
        screen_x: f32,
        screen_y: f32,
        reference_point: Vec3,
    ) -> Vec3 {
        let (w, h) = self.screen_size();
        self.camera_controller.screen_to_world_at_depth(
            screen_x,
            screen_y,
            w,
            h,
            reference_point,
        )
    }
}

// ── Configuration ──

impl ProteinRenderEngine {
    /// Set the GPU render scale (supersampling factor).
    pub fn set_render_scale(&mut self, scale: u32) {
        self.context.render_scale = scale;
    }

    /// Set the DPI scale factor for picking coordinate conversion.
    ///
    /// Call this with the window's `scale_factor()` when mouse events come
    /// from a webview (CSS/logical pixels) while the picking texture is
    /// in physical pixels.
    pub fn set_dpi_scale(&mut self, scale: f64) {
        self.dpi_scale = scale as f32;
    }
}

// ── Visualization (bands, pulls) ──

impl ProteinRenderEngine {
    /// Clear all constraint band visualizations.
    pub fn clear_bands(&mut self) {
        self.band_renderer.clear();
    }

    /// Clear the interactive pull visualization.
    pub fn clear_pulls(&mut self) {
        self.pull_renderer.clear();
    }
}
