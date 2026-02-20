//! Input & Selection methods for ProteinRenderEngine

use glam::Vec2;
use winit::event::MouseButton;

use super::ProteinRenderEngine;
use crate::{
    input::{ClickResult, KeyAction},
    scene::Focus,
};

impl ProteinRenderEngine {
    /// Forward mouse movement deltas for camera orbit, pan, or drag selection.
    pub fn handle_mouse_move(&mut self, delta_x: f32, delta_y: f32) {
        // Only allow rotation/pan if mouse down was on background (not on a
        // residue)
        if self.camera_controller.mouse_pressed
            && self.input.mouse_down_residue < 0
        {
            let delta = Vec2::new(delta_x, delta_y);
            // Mark that we're dragging (moved after mouse down)
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

    /// Handle mouse button press/release
    /// On press: record what residue (if any) is under cursor
    /// On release: handled by handle_mouse_up
    pub fn handle_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        if button == MouseButton::Left {
            if pressed {
                // Mouse down - record what's under cursor
                self.input.handle_mouse_down(self.picking.hovered_residue);
            }
            self.camera_controller.mouse_pressed = pressed;
        }
    }

    /// Handle mouse button release for selection
    /// Returns true if selection changed
    pub fn handle_mouse_up(&mut self) -> bool {
        let shift_held = self.camera_controller.shift_pressed;
        let hovered = self.picking.hovered_residue;

        match self.input.process_mouse_up(hovered, shift_held) {
            ClickResult::NoAction => false,
            ClickResult::SingleClick { shift_held } => {
                self.picking.handle_click(shift_held)
            }
            ClickResult::DoubleClick {
                residue,
                shift_held,
            } => self.select_ss_segment(residue, shift_held),
            ClickResult::TripleClick {
                residue,
                shift_held,
            } => self.select_chain(residue, shift_held),
            ClickResult::ClearSelection => {
                if !self.picking.selected_residues.is_empty() {
                    self.picking.selected_residues.clear();
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Select all residues in the same secondary structure segment as the given
    /// residue If shift_held is true, adds to existing selection; otherwise
    /// replaces selection
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
            self.picking.selected_residues.clear();
        }

        // Add all residues in this segment to selection (avoid duplicates)
        for i in start..=end {
            let residue = i as i32;
            if !self.picking.selected_residues.contains(&residue) {
                self.picking.selected_residues.push(residue);
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

        // Get backbone chains to determine chain boundaries
        let agg = self.scene.aggregated();
        let chains = &agg.backbone_chains;

        // Walk chains to find which one contains this residue
        let mut global_start = 0usize;
        for chain in chains {
            let chain_residues = chain.len() / 3;
            let global_end = global_start + chain_residues;
            if target >= global_start && target < global_end {
                // Found the chain — select all its residues
                if !shift_held {
                    self.picking.selected_residues.clear();
                }
                for i in global_start..global_end {
                    let residue = i as i32;
                    if !self.picking.selected_residues.contains(&residue) {
                        self.picking.selected_residues.push(residue);
                    }
                }
                return true;
            }
            global_start = global_end;
        }

        false
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
            Self::Cancel => engine.picking.clear_selection(),
        }
    }
}
