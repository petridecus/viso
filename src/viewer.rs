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

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::error::VisoError;
use crate::options::Options;
use crate::{InputEvent, MouseButton, ProteinRenderEngine};

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
    ///
    /// # Errors
    ///
    /// Returns [`VisoError::Viewer`] if the event loop or render engine
    /// fails to initialise.
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
            panel: crate::gui::panel::PanelController::new(),
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
    panel: crate::gui::panel::PanelController,
}

/// Compute the wgpu surface size — always the full window dimensions.
///
/// The webview options panel overlays the right edge of the window;
/// the surface must cover the entire window to avoid stretching.
fn viewport_size(inner: winit::dpi::PhysicalSize<u32>) -> (u32, u32) {
    (inner.width.max(1), inner.height.max(1))
}

/// Create a [`ProteinRenderEngine`], optionally loading a structure file.
fn create_engine(
    window: Arc<Window>,
    size: (u32, u32),
    scale: f64,
    path: Option<&str>,
) -> Result<ProteinRenderEngine, VisoError> {
    let result = if let Some(path) = path {
        pollster::block_on(ProteinRenderEngine::new_with_path(
            window, size, scale, path,
        ))
    } else {
        pollster::block_on(ProteinRenderEngine::new(window, size, scale))
    };
    result.map_err(|e| VisoError::Viewer(e.to_string()))
}

impl ViewerApp {
    fn handle_resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        let (vp_w, vp_h) = viewport_size(size);
        let Some(engine) = &mut self.engine else {
            return;
        };
        engine.resize(vp_w, vp_h);
        let Some(window) = &self.window else { return };
        #[cfg(feature = "gui")]
        self.panel.apply_layout(window);
    }

    fn handle_scale_factor_changed(&mut self, scale_factor: f64) {
        let Some(window) = &self.window else { return };
        let inner = window.inner_size();
        let Some(engine) = &mut self.engine else {
            return;
        };
        #[allow(clippy::cast_possible_truncation)]
        let render_scale = if scale_factor < 2.0 { 2 } else { 1 };
        engine.set_render_scale(render_scale);
        let (vp_w, vp_h) = viewport_size(inner);
        engine.resize(vp_w, vp_h);
    }

    /// Drain GUI actions, update, render, and push stats.
    fn handle_redraw(&mut self) {
        #[cfg(feature = "gui")]
        if let (Some(window), Some(engine)) = (&self.window, &mut self.engine) {
            self.panel.drain_and_apply(engine, window);
        }

        let now = Instant::now();
        let dt = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        let Some(engine) = &mut self.engine else {
            return;
        };

        engine.update(dt);

        match engine.render() {
            Ok(()) => {}
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                let Some(w) = &self.window else { return };
                let inner = w.inner_size();
                let (vp_w, vp_h) = viewport_size(inner);
                engine.resize(vp_w, vp_h);
            }
            Err(e) => {
                log::error!("render error: {e:?}");
            }
        }

        #[cfg(feature = "gui")]
        self.panel.push_stats_if_due(now, engine);

        let Some(w) = &self.window else { return };
        w.request_redraw();
    }

    fn handle_mouse_input(
        &mut self,
        button: winit::event::MouseButton,
        state: ElementState,
    ) {
        let Some(engine) = &mut self.engine else {
            return;
        };
        let pressed = state == ElementState::Pressed;
        let _ = engine.handle_input(InputEvent::MouseButton {
            button: MouseButton::from(button),
            pressed,
        });
    }

    fn handle_cursor_moved(
        &mut self,
        position: winit::dpi::PhysicalPosition<f64>,
    ) {
        let Some(engine) = &mut self.engine else {
            return;
        };
        #[allow(clippy::cast_possible_truncation)]
        let _ = engine.handle_input(InputEvent::CursorMoved {
            x: position.x as f32,
            y: position.y as f32,
        });

        let Some(window) = &self.window else { return };
        #[cfg(feature = "gui")]
        self.panel.update_peek(position.x as f32, window);
        window.request_redraw();
    }

    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        #[allow(clippy::cast_possible_truncation)]
        let scroll_delta = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
        };
        let Some(engine) = &mut self.engine else {
            return;
        };
        let _ = engine.handle_input(InputEvent::Scroll {
            delta: scroll_delta,
        });
        let Some(w) = &self.window else { return };
        w.request_redraw();
    }

    fn handle_modifiers_changed(&mut self, modifiers: winit::event::Modifiers) {
        let Some(engine) = &mut self.engine else {
            return;
        };
        let _ = engine.handle_input(InputEvent::ModifiersChanged {
            shift: modifiers.state().shift_key(),
        });
    }

    fn handle_keyboard_input(&mut self, event: &winit::event::KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }
        use winit::keyboard::PhysicalKey;
        let PhysicalKey::Code(code) = event.physical_key else {
            return;
        };

        #[cfg(feature = "gui")]
        if code == winit::keyboard::KeyCode::Backslash {
            self.panel.toggle();
            let Some(window) = &self.window else { return };
            self.panel.apply_layout(window);
            return;
        }

        let Some(engine) = &mut self.engine else {
            return;
        };
        let key_str = format!("{code:?}");
        if let Some(action) = engine.options().keybindings.lookup(&key_str) {
            action.execute(engine);
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

        let mut engine = match create_engine(
            window.clone(),
            (vp_w, vp_h),
            scale,
            self.path.as_deref(),
        ) {
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

        #[cfg(feature = "gui")]
        self.panel.init_webview(
            window.as_ref(),
            inner.width,
            inner.height,
            &engine,
        );

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
            WindowEvent::Resized(size) => self.handle_resize(size),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.handle_scale_factor_changed(scale_factor);
            }
            WindowEvent::RedrawRequested => self.handle_redraw(),
            WindowEvent::MouseInput { button, state, .. } => {
                self.handle_mouse_input(button, state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.handle_cursor_moved(position);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.handle_mouse_wheel(delta);
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.handle_modifiers_changed(modifiers);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard_input(&event);
            }
            _ => (),
        }
    }
}
