//! Read-only query methods and lifecycle helpers for [`VisoEngine`].

use foldit_conv::types::entity::MoleculeEntity;
use glam::Vec3;

use super::VisoEngine;
use crate::animation::transition::Transition;
use crate::options::VisoOptions;
use crate::scene::{EntityResidueRange, Focus, Scene};

// ── Camera ──

impl VisoEngine {
    /// Fit camera to the currently focused element.
    pub fn fit_camera_to_focus(&mut self) {
        match *self.scene.focus() {
            Focus::Session => {
                let positions = self.scene.all_positions();
                if !positions.is_empty() {
                    self.camera_controller
                        .fit_to_positions_animated(&positions);
                }
            }
            Focus::Entity(eid) => {
                if let Some(se) = self.scene.entity(eid) {
                    let positions = se.entity.positions();
                    if !positions.is_empty() {
                        self.camera_controller
                            .fit_to_positions_animated(&positions);
                    }
                }
            }
        }
    }
}

// ── Lifecycle ──

impl VisoEngine {
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

impl VisoEngine {
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

// ── Source-of-truth entity access ──

impl VisoEngine {
    /// Get the canonical entity list (source of truth).
    ///
    /// This is the authoritative copy of entity data on the engine.
    /// Scene may temporarily differ during animation.
    #[allow(dead_code)]
    pub fn entities(&self) -> &[MoleculeEntity] {
        &self.entities
    }

    /// Get per-entity residue ranges in the flat concatenated arrays.
    ///
    /// Each entry maps an entity ID to its start index and count in the
    /// global residue array. Populated when the scene processor delivers
    /// a prepared scene.
    #[allow(dead_code)]
    pub fn entity_ranges(&self) -> &[EntityResidueRange] {
        &self.entity_ranges
    }

    /// Set the animation behavior for a specific entity.
    ///
    /// This behavior will be used when the entity is next updated.
    /// Overrides the default smooth transition for the given entity.
    #[allow(dead_code)]
    pub fn set_entity_behavior(
        &mut self,
        entity_id: u32,
        transition: Transition,
    ) {
        let _ = self.entity_behaviors.insert(entity_id, transition.clone());
        self.animator.set_entity_behavior(entity_id, transition);
    }

    /// Clear a per-entity behavior override, reverting to default (smooth).
    #[allow(dead_code)]
    pub fn clear_entity_behavior(&mut self, entity_id: u32) {
        let _ = self.entity_behaviors.remove(&entity_id);
        self.animator.clear_entity_behavior(entity_id);
    }
}

// ── Input state ──

impl VisoEngine {
    /// Update the cursor position used for GPU picking.
    ///
    /// Call this each frame (or on each `CursorMoved` event) so the
    /// picking pass reads from the correct screen coordinate.
    pub fn set_cursor_pos(&mut self, x: f32, y: f32) {
        self.cursor_pos = (x, y);
    }

    /// The pick target currently under the cursor (resolved from the
    /// previous frame's GPU picking pass).
    pub fn hovered_target(&self) -> crate::renderer::picking::PickTarget {
        self.pick.hovered_target
    }
}

// ── Query (read-only state inspection) ──

impl VisoEngine {
    /// Currently hovered residue index, or -1 if none.
    pub fn hovered_residue(&self) -> i32 {
        self.pick.hovered_target.as_residue_i32()
    }

    /// Currently selected residue indices.
    pub fn selected_residues(&self) -> &[i32] {
        self.pick.selected_residues()
    }

    /// Clear the current residue selection.
    pub fn clear_selection(&mut self) {
        let _ = self.pick.clear_selection();
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
    pub fn options(&self) -> &VisoOptions {
        &self.options
    }

    /// GPU buffer sizes across all renderers.
    ///
    /// Each entry is `(label, used_bytes, allocated_bytes)`.
    pub fn gpu_buffer_stats(&self) -> Vec<(&str, usize, usize)> {
        let mut stats = Vec::new();
        stats.extend(self.renderers.buffer_info());
        stats.extend(self.pick.selection.buffer_info());
        stats.extend(self.pick.residue_colors.buffer_info());
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
            glam::Vec2::new(screen_x, screen_y),
            glam::UVec2::new(w, h),
            reference_point,
        )
    }
}

// ── Configuration ──

impl VisoEngine {
    /// Set the GPU render scale (supersampling factor).
    pub fn set_render_scale(&mut self, scale: u32) {
        self.context.render_scale = scale;
    }
}

// ── Visualization (bands, pulls) ──

impl VisoEngine {
    /// Clear all constraint band visualizations.
    pub fn clear_bands(&mut self) {
        self.renderers.band.clear();
    }

    /// Clear the interactive pull visualization.
    pub fn clear_pulls(&mut self) {
        self.renderers.pull.clear();
    }
}
