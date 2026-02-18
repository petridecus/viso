use crate::camera::controller::CameraController;
use glam::Vec2;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

pub struct InputHandler {
    last_mouse_pos: Vec2,
}

impl Default for InputHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl InputHandler {
    pub fn new() -> Self {
        Self {
            last_mouse_pos: Vec2::ZERO,
        }
    }

    /// Returns true if the event was consumed by the camera
    pub fn handle_event(
        &mut self,
        controller: &mut CameraController,
        event: &WindowEvent,
    ) -> bool {
        match event {
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                controller.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                controller.shift_pressed = modifiers.state().shift_key();
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current_pos =
                    Vec2::new(position.x as f32, position.y as f32);
                let delta = current_pos - self.last_mouse_pos;
                self.last_mouse_pos = current_pos;

                if controller.mouse_pressed {
                    if controller.shift_pressed {
                        controller.pan(delta);
                    } else {
                        controller.rotate(delta);
                    }
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                controller.zoom(scroll);
                true
            }
            _ => false,
        }
    }
}
