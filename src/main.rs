use std::{sync::Arc, time::Instant};

use viso::engine::ProteinRenderEngine;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct RenderApp {
    window: Option<Arc<Window>>,
    engine: Option<ProteinRenderEngine>,
    last_mouse_pos: (f32, f32),
    last_frame_time: Instant,
    cif_path: String,
}

impl RenderApp {
    fn new(cif_path: String) -> Self {
        Self {
            window: None,
            engine: None,
            last_mouse_pos: (0.0, 0.0),
            last_frame_time: Instant::now(),
            cif_path,
        }
    }
}

impl ApplicationHandler for RenderApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let monitor = event_loop
                .primary_monitor()
                .or_else(|| event_loop.available_monitors().next());
            let attrs = if let Some(mon) = &monitor {
                let mon_size = mon.size();
                let scale = mon.scale_factor();
                let logical_w = (mon_size.width as f64 / scale * 0.75) as u32;
                let logical_h = (mon_size.height as f64 / scale * 0.75) as u32;
                Window::default_attributes()
                    .with_title("Viso")
                    .with_inner_size(winit::dpi::LogicalSize::new(
                        logical_w, logical_h,
                    ))
            } else {
                Window::default_attributes().with_title("Viso")
            };
            let window = Arc::new(event_loop.create_window(attrs).unwrap());

            let size = window.inner_size();
            let scale = window.scale_factor();
            let (width, height) = (size.width, size.height);

            let mut engine =
                pollster::block_on(ProteinRenderEngine::new_with_path(
                    window.clone(),
                    (width, height),
                    scale,
                    &self.cif_path,
                ));

            // Kick off background scene processing so colors and geometry are
            // ready
            engine.sync_scene_to_renderers(None);

            window.request_redraw();
            self.window = Some(window);
            self.engine = Some(engine);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(event_size) => {
                if let Some(engine) = &mut self.engine {
                    engine.resize(event_size.width, event_size.height);
                }
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let (Some(window), Some(engine)) =
                    (&self.window, &mut self.engine)
                {
                    engine.set_scale_factor(scale_factor);
                    let inner = window.inner_size();
                    engine.resize(inner.width, inner.height);
                }
            }

            WindowEvent::RedrawRequested => {
                if let (Some(window), Some(engine)) =
                    (&self.window, &mut self.engine)
                {
                    let now = Instant::now();
                    let dt =
                        now.duration_since(self.last_frame_time).as_secs_f32();
                    self.last_frame_time = now;
                    engine.update_camera_animation(dt);

                    engine.apply_pending_scene();
                    match engine.render() {
                        Ok(()) => {}
                        Err(
                            wgpu::SurfaceError::Outdated
                            | wgpu::SurfaceError::Lost,
                        ) => {
                            let inner = window.inner_size();
                            engine.resize(inner.width, inner.height);
                        }
                        Err(e) => {
                            log::error!("render error: {:?}", e);
                        }
                    }
                    window.request_redraw();
                }
            }

            WindowEvent::MouseInput { button, state, .. } => {
                if let Some(engine) = &mut self.engine {
                    let pressed = state == ElementState::Pressed;
                    engine.handle_mouse_button(button, pressed);
                    if button == winit::event::MouseButton::Left && !pressed {
                        engine.handle_mouse_up();
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let delta_x = position.x as f32 - self.last_mouse_pos.0;
                let delta_y = position.y as f32 - self.last_mouse_pos.1;

                if let Some(engine) = &mut self.engine {
                    engine.handle_mouse_move(delta_x, delta_y);
                    engine.handle_mouse_position(
                        position.x as f32,
                        position.y as f32,
                    );
                }

                self.last_mouse_pos = (position.x as f32, position.y as f32);

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(engine) = &mut self.engine {
                    match delta {
                        MouseScrollDelta::LineDelta(_, y) => {
                            engine.handle_mouse_wheel(y)
                        }
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
                        use winit::keyboard::PhysicalKey;
                        if let PhysicalKey::Code(code) = event.physical_key {
                            let key_str = format!("{code:?}");
                            if let Some(action) =
                                engine.options().keybindings.lookup(&key_str)
                            {
                                action.execute(engine);
                            }
                        }
                    }
                }
            }

            _ => (),
        }
    }
}

fn resolve_structure_path(input: &str) -> Result<String, String> {
    if std::path::Path::new(input).exists() {
        return Ok(input.to_string());
    }

    if input.len() == 4 && input.chars().all(|c| c.is_ascii_alphanumeric()) {
        let pdb_id = input.to_lowercase();
        let models_dir = std::path::Path::new("assets/models");
        let local_path = models_dir.join(format!("{}.cif", pdb_id));

        if local_path.exists() {
            return Ok(local_path.to_string_lossy().to_string());
        }

        if !models_dir.exists() {
            std::fs::create_dir_all(models_dir).map_err(|e| {
                format!("Failed to create models directory: {}", e)
            })?;
        }

        let url = format!("https://files.rcsb.org/download/{}.cif", pdb_id);
        log::info!("Downloading {} from RCSB...", pdb_id.to_uppercase());

        let content = ureq::get(&url)
            .call()
            .map_err(|e| format!("Failed to download {}: {}", pdb_id, e))?
            .into_body()
            .read_to_string()
            .map_err(|e| format!("Failed to read response: {}", e))?;

        std::fs::write(&local_path, &content)
            .map_err(|e| format!("Failed to save CIF file: {}", e))?;

        log::info!("Downloaded to {}", local_path.display());
        return Ok(local_path.to_string_lossy().to_string());
    }

    Err(format!(
        "File not found and not a valid PDB code: {}",
        input
    ))
}

fn main() {
    env_logger::init();

    let input = match std::env::args().nth(1) {
        Some(arg) => arg,
        None => {
            log::error!("Usage: viso <PDB_ID or path>");
            std::process::exit(1);
        }
    };

    let cif_path = match resolve_structure_path(&input) {
        Ok(path) => path,
        Err(e) => {
            log::error!("{}", e);
            std::process::exit(1);
        }
    };

    let mut renderer = RenderApp::new(cif_path);
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut renderer).expect("Event loop error");
}
