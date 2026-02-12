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
pub mod nucleic_acid_renderer;
pub mod options;
pub mod picking;
pub mod pull_renderer;
pub mod render_context;
pub mod ribbon_renderer;
pub mod ssao;
pub mod text_renderer;
pub mod scene;
pub mod scene_processor;
pub mod trajectory;
pub mod tube_renderer;

use engine::ProteinRenderEngine;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Duration after the last resize event before we consider the startup burst
/// over and allow downsizes again. 500ms is generous — EDID negotiation is
/// typically 10-100ms, and winit's spurious events arrive within ~50ms.
const RESIZE_STABILISATION_MS: u128 = 500;

struct RenderApp {
    window: Option<Arc<Window>>,
    engine: Option<ProteinRenderEngine>,
    last_mouse_pos: (f32, f32),
    /// Largest size (by area) seen during the current resize burst.
    /// Prevents later spurious smaller events from clobbering the correct size.
    max_size_seen: Option<(u32, u32)>,
    /// Timestamp of the last resize event — used to detect when the startup
    /// burst has settled so that intentional user resizes (e.g. window drag,
    /// snap to half-screen) can take effect.
    last_resize_at: Option<Instant>,
}

impl RenderApp {
    fn new() -> Self {
        Self {
            window: None,
            engine: None,
            last_mouse_pos: (0.0, 0.0),
            max_size_seen: None,
            last_resize_at: None,
        }
    }
}

impl ApplicationHandler for RenderApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            // Use an explicit initial size to avoid winit's 800x600 hardcoded default.
            // This prevents the most common spurious resize event on external monitors
            // where EDID negotiation delays cause transient 800x600 states.
            let attrs = Window::default_attributes()
                .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .unwrap(),
            );

            let size = window.inner_size();
            let scale = window.scale_factor();
            let outer = window.outer_size();
            let monitor = window.current_monitor();
            eprintln!("[main::resumed] inner_size={:?} outer_size={:?} scale_factor={} monitor={:?}",
                size, outer, scale, monitor.as_ref().map(|m| (m.size(), m.scale_factor(), m.name())));
            let (width, height) = (size.width, size.height);
            eprintln!("[main::resumed] passing size={}x{} to engine", width, height);

            let engine = pollster::block_on(ProteinRenderEngine::new(window.clone(), (width, height)));

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

            WindowEvent::Resized(event_size) => {
                if let (Some(window), Some(engine)) = (&self.window, &mut self.engine) {
                    let inner = window.inner_size();
                    let (ew, eh) = (event_size.width, event_size.height);
                    let (iw, ih) = (inner.width, inner.height);

                    // Use whichever is larger — event_size leads inner_size when
                    // Windows snaps/tiles the window, but inner_size is more
                    // trustworthy for downsizes.
                    let (w, h) = if (ew as u64) * (eh as u64) >= (iw as u64) * (ih as u64) {
                        (ew, eh)
                    } else {
                        (iw, ih)
                    };

                    let now = Instant::now();

                    // If enough time has passed since the last resize, reset
                    // max_size_seen so intentional user resizes (drag, snap)
                    // are not blocked.
                    if let Some(last) = self.last_resize_at {
                        if now.duration_since(last).as_millis() > RESIZE_STABILISATION_MS {
                            eprintln!("[main::Resized] stabilisation period elapsed — resetting max_size_seen");
                            self.max_size_seen = None;
                        }
                    }
                    self.last_resize_at = Some(now);

                    // Never go smaller than the max we've seen during this burst.
                    // This prevents spurious 1280x720 events from clobbering the
                    // correct 1904x1020 during the startup resize sequence.
                    let candidate_area = (w as u64) * (h as u64);
                    if let Some((mw, mh)) = self.max_size_seen {
                        let max_area = (mw as u64) * (mh as u64);
                        if candidate_area < max_area {
                            eprintln!(
                                "[main::Resized] IGNORING downsize {}x{} (area {}), max seen {}x{} (area {})",
                                w, h, candidate_area, mw, mh, max_area
                            );
                            return;
                        }
                    }

                    self.max_size_seen = Some((w, h));
                    eprintln!("[main::Resized] event={:?} inner={:?} using={}x{} (new max)",
                        event_size, inner, w, h);
                    engine.resize(w, h);
                }
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let (Some(window), Some(engine)) = (&self.window, &mut self.engine) {
                    let size = window.inner_size();
                    eprintln!("[main::ScaleFactorChanged] new_scale={} inner_size={:?}", scale_factor, size);
                    engine.resize(size.width, size.height);
                }
            }

            WindowEvent::RedrawRequested => {
                if let (Some(window), Some(engine)) = (&self.window, &mut self.engine) {
                    match engine.render() {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Outdated) => {
                            // Surface/swapchain out of sync with window — the GPU
                            // backend is telling us the configured size is wrong.
                            // Trust inner_size() here (this is the "ground truth"
                            // correction) and reset max_size_seen so we don't block
                            // the corrected size.
                            let size = window.inner_size();
                            eprintln!(
                                "[main::RedrawRequested] Outdated — reconfiguring to {}x{}, resetting max_size_seen",
                                size.width, size.height
                            );
                            self.max_size_seen = Some((size.width, size.height));
                            engine.resize(size.width, size.height);
                        }
                        Err(e) => {
                            eprintln!("[main::RedrawRequested] render error: {:?}", e);
                        }
                    }
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
                            // W toggles water/ion visibility
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
