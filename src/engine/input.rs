//! Input & Selection methods for ProteinRenderEngine

use glam::Vec2;

use super::ProteinRenderEngine;
use crate::input::{ClickResult, InputEvent, KeyAction, MouseButton};
use crate::scene::Focus;

// ── Unified input handler ──

impl ProteinRenderEngine {
    /// Process a platform-agnostic input event.
    ///
    /// This is the primary input entry point. Consumers forward raw window
    /// events as [`InputEvent`] variants; the engine internally dispatches
    /// to camera rotation/pan/zoom, picking hover updates, and click
    /// detection/selection.
    ///
    /// Returns `true` if selection changed (only relevant for
    /// [`InputEvent::MouseButton`] releases).
    ///
    /// # Example
    ///
    /// ```ignore
    /// engine.handle_input(InputEvent::CursorMoved { x, y });
    /// engine.handle_input(InputEvent::Scroll { delta: 1.0 });
    /// ```
    pub fn handle_input(&mut self, event: InputEvent) -> bool {
        match event {
            InputEvent::CursorMoved { x, y } => {
                self.dispatch_cursor_moved(x, y);
                false
            }
            InputEvent::MouseButton { button, pressed } => {
                self.dispatch_mouse_button(button, pressed)
            }
            InputEvent::Scroll { delta } => {
                self.camera_controller.zoom(delta);
                false
            }
            InputEvent::ModifiersChanged { shift } => {
                self.camera_controller.shift_pressed = shift;
                false
            }
        }
    }

    /// Cursor moved — compute delta, forward to camera/picking.
    fn dispatch_cursor_moved(&mut self, x: f32, y: f32) {
        // Update picking hover position and get cursor delta
        let (delta_x, delta_y) = self.input.handle_mouse_position(x, y);

        // Camera rotate/pan if left-mouse is down on background
        if self.camera_controller.mouse_pressed
            && self.input.mouse_down_target.is_none()
        {
            let delta = Vec2::new(delta_x, delta_y);
            if delta.length_squared() > 1.0 {
                self.input.mark_dragging();
            }
            if self.camera_controller.shift_pressed {
                self.camera_controller.pan(delta);
            } else {
                self.camera_controller.rotate(delta);
            }
        }
    }

    /// Mouse button — dispatch press/release for left button.
    fn dispatch_mouse_button(
        &mut self,
        button: MouseButton,
        pressed: bool,
    ) -> bool {
        if button == MouseButton::Left {
            if pressed {
                self.input.handle_mouse_down(self.pick.hovered_target);
                self.camera_controller.mouse_pressed = true;
                false
            } else {
                self.camera_controller.mouse_pressed = false;
                self.handle_mouse_up()
            }
        } else {
            false
        }
    }

    /// Release mouse state without processing click detection.
    ///
    /// Used by consumers that intercept mouse events for pull/band drag
    /// and need to release the mouse without triggering selection changes.
    pub fn release_mouse_state(&mut self) {
        self.camera_controller.mouse_pressed = false;
    }

    /// Handle mouse button release for selection.
    /// Returns true if selection changed.
    fn handle_mouse_up(&mut self) -> bool {
        let shift_held = self.camera_controller.shift_pressed;
        let hovered = self.pick.hovered_target;

        match self.input.process_mouse_up(hovered, shift_held) {
            ClickResult::NoAction => false,
            ClickResult::SingleClick { target, shift_held } => {
                self.pick.picking
                    .handle_click(target.as_residue_i32(), shift_held)
            }
            ClickResult::DoubleClick { target, shift_held } => {
                self.select_ss_segment(target.as_residue_i32(), shift_held)
            }
            ClickResult::TripleClick { target, shift_held } => {
                self.select_chain(target.as_residue_i32(), shift_held)
            }
            ClickResult::ClearSelection => {
                if self.pick.picking.selected_residues.is_empty() {
                    false
                } else {
                    self.pick.picking.selected_residues.clear();
                    true
                }
            }
        }
    }

    /// Select all residues in the same secondary structure segment as the given
    /// residue. If shift_held is true, adds to existing selection; otherwise
    /// replaces selection.
    fn select_ss_segment(
        &mut self,
        residue_idx: i32,
        shift_held: bool,
    ) -> bool {
        if residue_idx < 0
            || (residue_idx as usize) >= self.sc.cached_ss_types.len()
        {
            return false;
        }

        let idx = residue_idx as usize;
        let target_ss = self.sc.cached_ss_types[idx];

        // Find the start of this SS segment (walk backwards)
        let mut start = idx;
        while start > 0 && self.sc.cached_ss_types[start - 1] == target_ss {
            start -= 1;
        }

        // Find the end of this SS segment (walk forwards)
        let mut end = idx;
        while end + 1 < self.sc.cached_ss_types.len()
            && self.sc.cached_ss_types[end + 1] == target_ss
        {
            end += 1;
        }

        // If shift is NOT held, clear existing selection first
        if !shift_held {
            self.pick.picking.selected_residues.clear();
        }

        // Add all residues in this segment to selection (avoid duplicates)
        for i in start..=end {
            let residue = i as i32;
            if !self.pick.picking.selected_residues.contains(&residue) {
                self.pick.picking.selected_residues.push(residue);
            }
        }

        true
    }

    /// Select all residues in the same chain as the given residue.
    /// If shift_held is true, adds to existing selection; otherwise replaces
    /// selection.
    fn select_chain(&mut self, residue_idx: i32, shift_held: bool) -> bool {
        if residue_idx < 0 {
            return false;
        }
        let target = residue_idx as usize;
        let chains = self.renderers.backbone.cached_chains();

        // Find the chain range containing the target residue
        let mut global_start = 0usize;
        let chain_range = chains.iter().find_map(|chain| {
            let chain_residues = chain.len() / 3;
            let global_end = global_start + chain_residues;
            let result = (target >= global_start && target < global_end)
                .then_some(global_start..global_end);
            global_start = global_end;
            result
        });

        let Some(range) = chain_range else {
            return false;
        };

        if !shift_held {
            self.pick.picking.selected_residues.clear();
        }
        for i in range {
            let residue = i as i32;
            if !self.pick.picking.selected_residues.contains(&residue) {
                self.pick.picking.selected_residues.push(residue);
            }
        }
        true
    }
}

// ── KeyAction execution ──

impl KeyAction {
    /// Execute this action on the given engine.
    pub fn execute(self, engine: &mut ProteinRenderEngine) {
        match self {
            Self::RecenterCamera => engine.fit_camera_to_focus(),
            Self::ToggleWaters => engine.toggle_waters(),
            Self::ToggleIons => engine.toggle_ions(),
            Self::ToggleSolvent => engine.toggle_solvent(),
            Self::ToggleLipids => engine.toggle_lipids(),
            Self::ToggleAutoRotate => {
                let _ = engine.camera_controller.toggle_auto_rotate();
            }
            Self::ToggleTrajectory => {
                if engine.trajectory_player.is_some() {
                    engine.toggle_trajectory();
                }
            }
            Self::CycleFocus => {
                let _ = engine.scene.cycle_focus();
                engine.fit_camera_to_focus();
            }
            Self::ResetFocus => {
                engine.scene.set_focus(Focus::Session);
                engine.fit_camera_to_focus();
            }
            Self::Cancel => engine.pick.picking.clear_selection(),
        }
    }
}
