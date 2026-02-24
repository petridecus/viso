//! Standalone visualization window backed by winit.
//!
//! When the `gui` feature is enabled, a wry webview panel is created
//! alongside the 3D viewport for the schema-driven options UI.
//!
//! ```no_run
//! # use viso::Viewer;
//! Viewer::builder()
//!     .with_path("assets/models/4pnk.cif")
//!     .build()
//!     .run()
//!     .unwrap();
//! ```

use std::{sync::Arc, time::Instant, time::Duration};

use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use crate::{
    error::VisoError, options::Options, InputEvent, MouseButton,
    ProteinRenderEngine,
};

// ── Builder ──────────────────────────────────────────────────────────────

/// Fluent builder for [`Viewer`].
pub struct ViewerBuilder {
    path: Option<String>,
    options: Option<Options>,
    title: String,
}

impl ViewerBuilder {
    /// Create a builder with sensible defaults (title "Viso", no path,
    /// default options).
    fn new() -> Self {
        Self {
            path: None,
            options: None,
            title: "Viso".into(),
        }
    }

    /// Set the structure file path (`.cif` or `.pdb`).
    #[must_use]
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Override the default options.
    #[must_use]
    pub fn with_options(mut self, options: Options) -> Self {
        self.options = Some(options);
        self
    }

    /// Set the window title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Consume the builder and produce a [`Viewer`].
    #[must_use]
    pub fn build(self) -> Viewer {
        Viewer {
            path: self.path,
            options: self.options,
            title: self.title,
        }
    }
}

// ── Viewer ───────────────────────────────────────────────────────────────

/// A standalone window that displays a protein structure.
///
/// Construct via [`Viewer::builder`], then call [`run`](Self::run) to
/// enter the event loop.
pub struct Viewer {
    path: Option<String>,
    options: Option<Options>,
    title: String,
}

impl Viewer {
    /// Start a new builder.
    #[must_use]
    pub fn builder() -> ViewerBuilder {
        ViewerBuilder::new()
    }

    /// Open the window and run the event loop. Blocks until the window is
    /// closed.
    pub fn run(self) -> Result<(), VisoError> {
        let event_loop =
            EventLoop::new().map_err(|e| VisoError::Viewer(e.to_string()))?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut app = ViewerApp {
            window: None,
            engine: None,
            last_frame_time: Instant::now(),
            path: self.path,
            options: self.options,
            title: self.title,
            #[cfg(feature = "gui")]
            webview: None,
            #[cfg(feature = "gui")]
            action_rx: None,
            #[cfg(feature = "gui")]
            last_stats_push: Instant::now(),
        };

        event_loop
            .run_app(&mut app)
            .map_err(|e| VisoError::Viewer(e.to_string()))
    }
}

// ── Winit app ────────────────────────────────────────────────────────────

/// Internal winit application handler.
struct ViewerApp {
    window: Option<Arc<Window>>,
    engine: Option<ProteinRenderEngine>,
    last_frame_time: Instant,
    path: Option<String>,
    options: Option<Options>,
    title: String,
    #[cfg(feature = "gui")]
    webview: Option<wry::WebView>,
    #[cfg(feature = "gui")]
    action_rx: Option<std::sync::mpsc::Receiver<crate::gui::webview::UiAction>>,
    #[cfg(feature = "gui")]
    last_stats_push: Instant,
}

/// Compute the wgpu surface size, reserving panel width when gui is
/// enabled.
fn viewport_size(inner: winit::dpi::PhysicalSize<u32>) -> (u32, u32) {
    #[cfg(feature = "gui")]
    {
        let panel_w = crate::gui::webview::PANEL_WIDTH;
        let vp_w = inner.width.saturating_sub(panel_w);
        (vp_w.max(1), inner.height.max(1))
    }
    #[cfg(not(feature = "gui"))]
    {
        (inner.width.max(1), inner.height.max(1))
    }
}

/// Drain IPC actions from the webview and apply to the engine.
#[cfg(feature = "gui")]
fn drain_ui_actions(
    rx: &std::sync::mpsc::Receiver<crate::gui::webview::UiAction>,
    engine: &mut ProteinRenderEngine,
) {
    use crate::gui::webview::UiAction;

    while let Ok(action) = rx.try_recv() {
        match action {
            UiAction::SetOption { path, field, value } => {
                let mut opts = engine.options().clone();
                let Ok(mut root) = serde_json::to_value(&opts) else {
                    continue;
                };
                if let Some(section) = root.get_mut(&path) {
                    section[&field] = value;
                }
                if let Ok(updated) = serde_json::from_value(root) {
                    opts = updated;
                }
                engine.set_options(opts);
            }
            UiAction::LoadFile { path } => {
                log::info!("load_file action: {path}");
                // TODO: implement runtime file loading
            }
        }
    }
}

impl ApplicationHandler for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let monitor = event_loop
            .primary_monitor()
            .or_else(|| event_loop.available_monitors().next());
        let attrs = if let Some(mon) = &monitor {
            let mon_size = mon.size();
            let scale = mon.scale_factor();
            #[allow(clippy::cast_possible_truncation)]
            let logical_w = (mon_size.width as f64 / scale * 0.75) as u32;
            #[allow(clippy::cast_possible_truncation)]
            let logical_h = (mon_size.height as f64 / scale * 0.75) as u32;
            Window::default_attributes()
                .with_title(&self.title)
                .with_inner_size(winit::dpi::LogicalSize::new(
                    logical_w, logical_h,
                ))
        } else {
            Window::default_attributes().with_title(&self.title)
        };

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("Failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        let inner = window.inner_size();
        let scale = window.scale_factor();
        let (vp_w, vp_h) = viewport_size(inner);

        let engine_result = if let Some(ref path) = self.path {
            pollster::block_on(ProteinRenderEngine::new_with_path(
                window.clone(),
                (vp_w, vp_h),
                scale,
                path,
            ))
        } else {
            pollster::block_on(ProteinRenderEngine::new(
                window.clone(),
                (vp_w, vp_h),
                scale,
            ))
        };

        let mut engine = match engine_result {
            Ok(e) => e,
            Err(e) => {
                log::error!("Failed to initialize engine: {e}");
                event_loop.exit();
                return;
            }
        };

        if let Some(opts) = self.options.take() {
            engine.set_options(opts);
        }

        engine.sync_scene_to_renderers(None);

        // Create the wry webview panel (gui feature only)
        #[cfg(feature = "gui")]
        {
            match crate::gui::webview::create_webview(
                window.as_ref(),
                inner.width,
                inner.height,
            ) {
                Ok((wv, rx)) => {
                    crate::gui::webview::push_schema(&wv, engine.options());
                    self.webview = Some(wv);
                    self.action_rx = Some(rx);
                }
                Err(e) => {
                    log::error!("Failed to create webview: {e}");
                    // Continue without GUI panel
                }
            }
        }

        window.request_redraw();
        self.window = Some(window);
        self.engine = Some(engine);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        if matches!(event, WindowEvent::CloseRequested) {
            event_loop.exit();
            return;
        }

        // Guard: both window and engine must be initialised.
        if self.window.is_none() || self.engine.is_none() {
            return;
        }

        match event {
            WindowEvent::Resized(event_size) => {
                let (vp_w, vp_h) = viewport_size(event_size);
                if let Some(engine) = &mut self.engine {
                    engine.resize(vp_w, vp_h);
                }
                #[cfg(feature = "gui")]
                if let Some(ref wv) = self.webview {
                    let _ = wv.set_bounds(crate::gui::webview::panel_bounds(
                        event_size.width,
                        event_size.height,
                    ));
                }
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                #[allow(clippy::cast_possible_truncation)]
                let render_scale = if scale_factor < 2.0 { 2 } else { 1 };
                let inner = self.window.as_ref().map(|w| w.inner_size());
                if let Some(engine) = &mut self.engine {
                    engine.set_render_scale(render_scale);
                    if let Some(inner) = inner {
                        let (vp_w, vp_h) = viewport_size(inner);
                        engine.resize(vp_w, vp_h);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                #[cfg(feature = "gui")]
                if let (Some(rx), Some(engine)) =
                    (&self.action_rx, &mut self.engine)
                {
                    drain_ui_actions(rx, engine);
                }

                let now = Instant::now();
                let dt = now.duration_since(self.last_frame_time).as_secs_f32();
                self.last_frame_time = now;

                if let Some(engine) = &mut self.engine {
                    engine.update(dt);
                    match engine.render() {
                        Ok(()) => {}
                        Err(
                            wgpu::SurfaceError::Outdated
                            | wgpu::SurfaceError::Lost,
                        ) => {
                            if let Some(w) = &self.window {
                                let inner = w.inner_size();
                                let (vp_w, vp_h) = viewport_size(inner);
                                engine.resize(vp_w, vp_h);
                            }
                        }
                        Err(e) => {
                            log::error!("render error: {:?}", e);
                        }
                    }

                    // Push FPS stats to webview at ~4 Hz
                    #[cfg(feature = "gui")]
                    if let Some(ref wv) = self.webview {
                        if now.duration_since(self.last_stats_push)
                            >= Duration::from_millis(250)
                        {
                            crate::gui::webview::push_stats(
                                wv,
                                engine.frame_timing.fps(),
                            );
                            self.last_stats_push = now;
                        }
                    }
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            WindowEvent::MouseInput { button, state, .. } => {
                let pressed = state == ElementState::Pressed;
                if let Some(engine) = &mut self.engine {
                    let _ = engine.handle_input(InputEvent::MouseButton {
                        button: MouseButton::from(button),
                        pressed,
                    });
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if let Some(engine) = &mut self.engine {
                    #[allow(clippy::cast_possible_truncation)]
                    let _ = engine.handle_input(InputEvent::CursorMoved {
                        x: position.x as f32,
                        y: position.y as f32,
                    });
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                #[allow(clippy::cast_possible_truncation)]
                let scroll_delta = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                if let Some(engine) = &mut self.engine {
                    let _ = engine.handle_input(InputEvent::Scroll {
                        delta: scroll_delta,
                    });
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            WindowEvent::ModifiersChanged(modifiers) => {
                if let Some(engine) = &mut self.engine {
                    let _ = engine.handle_input(InputEvent::ModifiersChanged {
                        shift: modifiers.state().shift_key(),
                    });
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                use winit::keyboard::PhysicalKey;
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                let key_str = format!("{code:?}");
                if let Some(engine) = &mut self.engine {
                    if let Some(action) =
                        engine.options().keybindings.lookup(&key_str)
                    {
                        action.execute(engine);
                    }
                }
            }

            _ => (),
        }
    }
}
