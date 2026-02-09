pub mod animation;
pub mod ball_and_stick_renderer;
pub mod band_renderer;
pub mod bond_topology;
pub mod camera;
pub mod capsule_sidechain_renderer;
pub mod composite;
pub mod dynamic_buffer;
pub mod easing;
pub mod engine;
pub mod frame_timing;
pub mod lighting;
pub mod picking;
pub mod pull_renderer;
pub mod render_context;
pub mod ribbon_renderer;
pub mod ssao;
pub mod text_renderer;
pub mod scene;
pub mod tube_renderer;

use engine::ProteinRenderEngine;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

struct RenderApp {
    window: Option<Arc<Window>>,
    engine: Option<ProteinRenderEngine>,
    last_mouse_pos: (f32, f32),
}

impl RenderApp {
    fn new() -> Self {
        Self {
            window: None,
            engine: None,
            last_mouse_pos: (0.0, 0.0),
        }
    }
}

impl ApplicationHandler for RenderApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes())
                    .unwrap(),
            );
            let size = window.inner_size();
            let engine = pollster::block_on(ProteinRenderEngine::new(window.clone(), (size.width, size.height)));

            window.request_redraw();
            self.window = Some(window);
            self.engine = Some(engine);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(newsize) => {
                if let Some(engine) = &mut self.engine {
                    engine.resize(newsize.width, newsize.height);
                }
            }

            WindowEvent::ScaleFactorChanged { .. } => {
                if let (Some(window), Some(engine)) = (&self.window, &mut self.engine) {
                    let newsize = window.inner_size();
                    engine.resize(newsize.width, newsize.height);
                }
            }

            WindowEvent::RedrawRequested => {
                if let (Some(window), Some(engine)) = (&self.window, &mut self.engine) {
                    let _ = engine.render();
                    // Request continuous redraws for smooth FPS updates
                    window.request_redraw();
                }
            }

            WindowEvent::MouseInput { button, state, .. } => {
                if let Some(engine) = &mut self.engine {
                    let pressed = state == ElementState::Pressed;

                    // Handle mouse button state change (tracks mouse_down_residue internally)
                    engine.handle_mouse_button(button, pressed);

                    // On mouse up, handle selection logic
                    if button == winit::event::MouseButton::Left && !pressed {
                        engine.handle_mouse_up();
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let delta_x = position.x as f32 - self.last_mouse_pos.0;
                let delta_y = position.y as f32 - self.last_mouse_pos.1;

                if let Some(engine) = &mut self.engine {
                    // Update camera (drag)
                    engine.handle_mouse_move(delta_x, delta_y);
                    
                    // Update hover (always, for picking)
                    engine.handle_mouse_position(position.x as f32, position.y as f32);
                }

                self.last_mouse_pos = (position.x as f32, position.y as f32);

                // Request redraw on mouse movement
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(engine) = &mut self.engine {
                    match delta {
                        MouseScrollDelta::LineDelta(_, y) => engine.handle_mouse_wheel(y),
                        MouseScrollDelta::PixelDelta(pos) => {
                            engine.handle_mouse_wheel(pos.y as f32 * 0.01)
                        }
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::ModifiersChanged(modifiers) => {
                if let Some(engine) = &mut self.engine {
                    engine.update_modifiers(modifiers.state());
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let Some(engine) = &mut self.engine {
                        use winit::keyboard::{Key, NamedKey};
                        match &event.logical_key {
                            // V toggles view mode (Tube <-> Ribbon)
                            Key::Character(c) if c.as_str() == "v" || c.as_str() == "V" => {
                                engine.toggle_view_mode();
                            }
                            // W toggles water visibility
                            Key::Character(c) if c.as_str() == "w" || c.as_str() == "W" => {
                                engine.toggle_waters();
                            }
                            // Escape clears selection
                            Key::Named(NamedKey::Escape) => {
                                engine.picking.clear_selection();
                            }
                            _ => {}
                        }
                    }
                }
            }

            _ => (),
        }
    }
}

fn main() {
    let mut renderer = RenderApp::new();
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut renderer).expect("Event loop error");
}
