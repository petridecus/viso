//! GUI panel controller — owns the wry webview and its state.
//!
//! Extracted from `viewer.rs` so that `ViewerApp` holds a
//! single `PanelController` field instead of six `#[cfg(feature = "gui")]`
//! fields.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use winit::window::Window;

use super::webview::{self, UiAction};
use crate::ProteinRenderEngine;

/// Owns the webview panel and all associated state.
pub(crate) struct PanelController {
    webview: Option<wry::WebView>,
    action_rx: Option<mpsc::Receiver<UiAction>>,
    last_stats_push: Instant,
    /// Whether the options panel is pinned open (visible).
    pinned: bool,
    /// Whether the panel is temporarily revealed by a mouse hover.
    peek: bool,
    /// Current panel width in physical pixels.
    width: u32,
}

// ── Constants ────────────────────────────────────────────────────────────

impl PanelController {
    /// Margin around the panel when floating (not pinned).
    const PANEL_MARGIN: u32 = 10;
    /// Minimum panel width for resize.
    const MIN_PANEL_WIDTH: u32 = 220;
    /// Maximum panel width for resize.
    const MAX_PANEL_WIDTH: u32 = 700;
}

// ── Construction ─────────────────────────────────────────────────────────

impl PanelController {
    /// Create a new controller with default state (pinned, no webview yet).
    pub(crate) fn new() -> Self {
        Self {
            webview: None,
            action_rx: None,
            last_stats_push: Instant::now(),
            pinned: true,
            peek: false,
            width: webview::PANEL_WIDTH,
        }
    }

    /// Create the wry webview and push the initial schema to it.
    pub(crate) fn init_webview(
        &mut self,
        window: &Window,
        width: u32,
        height: u32,
        engine: &ProteinRenderEngine,
    ) {
        match webview::create_webview(window, width, height, self.width) {
            Ok((wv, rx)) => {
                webview::push_schema(&wv, engine.options());
                webview::push_panel_pinned(&wv, self.pinned);
                self.webview = Some(wv);
                self.action_rx = Some(rx);
            }
            Err(e) => {
                log::error!("Failed to create webview: {e}");
                // Continue without GUI panel
            }
        }
    }
}

// ── Runtime ──────────────────────────────────────────────────────────────

impl PanelController {
    /// Toggle pinned state and push it to the webview.
    pub(crate) fn toggle(&mut self) {
        self.pinned = !self.pinned;
        self.peek = false;
        if let Some(ref wv) = self.webview {
            webview::push_panel_pinned(wv, self.pinned);
        }
    }

    /// Position the webview according to the current pinned/peek state.
    pub(crate) fn apply_layout(&self, window: &Window) {
        let inner = window.inner_size();
        let Some(ref wv) = self.webview else {
            return;
        };

        let visible = self.pinned || self.peek;
        if visible {
            let bounds = if self.pinned {
                webview::panel_bounds(inner.width, inner.height, self.width)
            } else {
                webview::panel_bounds_floating(
                    inner.width,
                    inner.height,
                    self.width,
                    Self::PANEL_MARGIN,
                )
            };
            let _ = wv.set_bounds(bounds);
        } else {
            // Park off-screen to the right.
            use wry::dpi;
            let _ = wv.set_bounds(wry::Rect {
                position: dpi::Position::Physical(dpi::PhysicalPosition::new(
                    inner.width as i32,
                    0,
                )),
                size: dpi::Size::Physical(dpi::PhysicalSize::new(
                    self.width,
                    inner.height,
                )),
            });
        }
    }

    /// Check if the mouse is near the right edge and temporarily reveal the
    /// panel.
    pub(crate) fn update_peek(&mut self, mouse_x: f32, window: &Window) {
        if self.pinned {
            return;
        }
        let inner = window.inner_size();
        let edge_zone = 6.0;

        let near_edge = mouse_x >= (inner.width as f32 - edge_zone);
        let in_panel = mouse_x
            >= (inner.width as f32
                - self.width as f32
                - Self::PANEL_MARGIN as f32);

        let should_peek = near_edge || (self.peek && in_panel);

        if should_peek != self.peek {
            self.peek = should_peek;
            self.apply_layout(window);
        }
    }

    /// Drain IPC actions from the webview, apply them to the engine, and
    /// handle panel toggle/resize.  Returns `true` if the panel layout
    /// changed (so the caller can request a redraw of the layout).
    pub(crate) fn drain_and_apply(
        &mut self,
        engine: &mut ProteinRenderEngine,
        window: &Window,
    ) {
        let Some(ref rx) = self.action_rx else {
            return;
        };

        let mut toggled = false;
        let mut resize_width: Option<u32> = None;

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
                UiAction::TogglePanel => {
                    toggled = true;
                }
                UiAction::ResizePanel { width } => {
                    resize_width = Some(width);
                }
            }
        }

        if toggled {
            self.toggle();
            self.apply_layout(window);
        }
        if let Some(w) = resize_width {
            let clamped = w.clamp(Self::MIN_PANEL_WIDTH, Self::MAX_PANEL_WIDTH);
            if clamped != self.width {
                self.width = clamped;
                self.apply_layout(window);
            }
        }
    }

    /// Push FPS and GPU buffer stats to the webview at ~4 Hz.
    pub(crate) fn push_stats_if_due(
        &mut self,
        now: Instant,
        engine: &ProteinRenderEngine,
    ) {
        let Some(ref wv) = self.webview else {
            return;
        };
        if now.duration_since(self.last_stats_push)
            >= Duration::from_millis(250)
        {
            let buffers = engine.gpu_buffer_stats();
            webview::push_stats(wv, engine.frame_timing.fps(), &buffers);
            self.last_stats_push = now;
        }
    }
}
