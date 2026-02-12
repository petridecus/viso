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

/// Format a diagnostic timestamp: elapsed ms since t0 + sequence number.
fn ts(t0: &Instant, seq: &mut u64) -> String {
    *seq += 1;
    let elapsed = t0.elapsed();
    format!("[{:>6}.{:03}ms seq={}]",
        elapsed.as_millis(),
        elapsed.as_micros() % 1000,
        *seq)
}

struct RenderApp {
    window: Option<Arc<Window>>,
    engine: Option<ProteinRenderEngine>,
    last_mouse_pos: (f32, f32),
    /// Monotonic clock used for all diagnostic timestamps
    t0: Instant,
    /// Sequence counter for events (to spot ordering even with identical timestamps)
    event_seq: u64,
}

impl RenderApp {
    fn new() -> Self {
        Self {
            window: None,
            engine: None,
            last_mouse_pos: (0.0, 0.0),
            t0: Instant::now(),
            event_seq: 0,
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
            let scale = window.scale_factor();
            let outer = window.outer_size();
            let monitor = window.current_monitor();
            let t = ts(&self.t0, &mut self.event_seq);
            eprintln!("{} [RESUMED] inner_size={:?} outer_size={:?} scale_factor={} monitor={:?}",
                t, size, outer, scale, monitor.as_ref().map(|m| (m.size(), m.scale_factor(), m.name())));
            let (width, height) = (size.width, size.height);
            eprintln!("{} [RESUMED] passing size={}x{} (physical px) to engine", t, width, height);

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
                let t = ts(&self.t0, &mut self.event_seq);
                if let (Some(window), Some(engine)) = (&self.window, &mut self.engine) {
                    let inner = window.inner_size();
                    let outer = window.outer_size();
                    let scale = window.scale_factor();
                    let cfg_w = engine.context.config.width;
                    let cfg_h = engine.context.config.height;
                    eprintln!("{} [RESIZED] event_payload={}x{} \
                        inner_size={}x{} outer_size={}x{} scale_factor={:.4} \
                        surface_config={}x{} \
                        event_vs_inner={}",
                        t,
                        event_size.width, event_size.height,
                        inner.width, inner.height,
                        outer.width, outer.height,
                        scale,
                        cfg_w, cfg_h,
                        if event_size == inner { "MATCH" } else { "MISMATCH" },
                    );
                    // Use inner_size (physical pixels) — the authoritative value
                    engine.resize(inner.width, inner.height);
                    // Log surface config AFTER the resize to confirm what was applied
                    let t2 = ts(&self.t0, &mut self.event_seq);
                    eprintln!("{} [RESIZED] post-resize surface_config={}x{}",
                        t2, engine.context.config.width, engine.context.config.height);
                }
            }

            WindowEvent::ScaleFactorChanged { scale_factor, inner_size_writer: _ } => {
                let t = ts(&self.t0, &mut self.event_seq);
                if let (Some(window), Some(engine)) = (&self.window, &mut self.engine) {
                    let inner = window.inner_size();
                    let outer = window.outer_size();
                    let cfg_w = engine.context.config.width;
                    let cfg_h = engine.context.config.height;
                    eprintln!("{} [SCALE_FACTOR_CHANGED] new_scale={:.4} \
                        inner_size={}x{} outer_size={}x{} \
                        surface_config_before={}x{}",
                        t,
                        scale_factor,
                        inner.width, inner.height,
                        outer.width, outer.height,
                        cfg_w, cfg_h,
                    );
                    engine.resize(inner.width, inner.height);
                    let t2 = ts(&self.t0, &mut self.event_seq);
                    eprintln!("{} [SCALE_FACTOR_CHANGED] post-resize surface_config={}x{}",
                        t2, engine.context.config.width, engine.context.config.height);
                }
            }

            WindowEvent::RedrawRequested => {
                // Compute log-gating fields before borrowing engine/window
                let elapsed_ms = self.t0.elapsed().as_millis();
                let should_log = elapsed_ms < 3000 || self.event_seq % 300 == 0;
                if let (Some(window), Some(engine)) = (&self.window, &mut self.engine) {
                    if should_log {
                        let inner = window.inner_size();
                        let scale = window.scale_factor();
                        let cfg_w = engine.context.config.width;
                        let cfg_h = engine.context.config.height;
                        let t = ts(&self.t0, &mut self.event_seq);
                        eprintln!("{} [REDRAW] inner_size={}x{} scale={:.4} surface_config={}x{} match={}",
                            t,
                            inner.width, inner.height,
                            scale,
                            cfg_w, cfg_h,
                            if inner.width == cfg_w && inner.height == cfg_h { "YES" } else { "NO" },
                        );
                    }
                    match engine.render() {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Outdated) => {
                            let inner = window.inner_size();
                            let t = ts(&self.t0, &mut self.event_seq);
                            eprintln!("{} [REDRAW] SurfaceError::Outdated! \
                                inner_size={}x{} scale={:.4} — forcing resize",
                                t, inner.width, inner.height, window.scale_factor());
                            engine.resize(inner.width, inner.height);
                        }
                        Err(wgpu::SurfaceError::Lost) => {
                            let inner = window.inner_size();
                            let t = ts(&self.t0, &mut self.event_seq);
                            eprintln!("{} [REDRAW] SurfaceError::Lost! \
                                inner_size={}x{} — forcing resize",
                                t, inner.width, inner.height);
                            engine.resize(inner.width, inner.height);
                        }
                        Err(e) => {
                            let t = ts(&self.t0, &mut self.event_seq);
                            eprintln!("{} [REDRAW] render error: {:?}", t, e);
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

            WindowEvent::Moved(position) => {
                // Log window movement — fires when dragging between monitors
                // and may correlate with scale_factor changes
                if let Some(window) = &self.window {
                    let inner = window.inner_size();
                    let scale = window.scale_factor();
                    let monitor = window.current_monitor();
                    let t = ts(&self.t0, &mut self.event_seq);
                    eprintln!("{} [MOVED] position={:?} inner_size={}x{} scale={:.4} monitor={:?}",
                        t, position,
                        inner.width, inner.height,
                        scale,
                        monitor.as_ref().map(|m| (m.size(), m.scale_factor(), m.name())),
                    );
                }
            }

            WindowEvent::Occluded(occluded) => {
                if let Some(window) = &self.window {
                    let t = ts(&self.t0, &mut self.event_seq);
                    eprintln!("{} [OCCLUDED] occluded={} inner_size={:?} scale={:.4}",
                        t, occluded, window.inner_size(), window.scale_factor());
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
