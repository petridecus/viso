//! GUI panel controller — owns the wry webview and its state.
//!
//! Extracted from `viewer.rs` so that `ViewerApp` holds a
//! single `PanelController` field instead of six `#[cfg(feature = "gui")]`
//! fields.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use winit::window::Window;
use wry::dpi;

use super::webview::{self, UiAction};
use crate::VisoEngine;

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
    /// Animated x-position of the panel (physical px, left edge).
    /// Only used for the unpinned slide; pinned mode ignores this.
    slide_x: f32,
    /// Receiver for background PDB fetch results (path or error message).
    load_rx: Option<mpsc::Receiver<Result<String, String>>>,
}

// ── Constants ────────────────────────────────────────────────────────────

impl PanelController {
    /// Margin around the panel when floating (not pinned).
    const PANEL_MARGIN: u32 = 10;
    /// Minimum panel width for resize.
    const MIN_PANEL_WIDTH: u32 = 220;
    /// Maximum panel width for resize.
    const MAX_PANEL_WIDTH: u32 = 700;
    /// How many pixels from the right edge trigger a peek.
    const EDGE_ZONE: f32 = 24.0;
    /// Extra grace pixels past the panel's left edge before it hides.
    const PEEK_GRACE: f32 = 40.0;
    /// Lerp speed for the peek slide (fraction per second).
    const SLIDE_SPEED: f32 = 12.0;
    /// Snap when within this many pixels of the target.
    const SNAP_PX: f32 = 0.5;
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
            slide_x: 0.0,
            load_rx: None,
        }
    }

    /// Create the wry webview and push the initial schema to it.
    pub(crate) fn init_webview(
        &mut self,
        window: &Window,
        width: u32,
        height: u32,
        engine: &VisoEngine,
    ) {
        match webview::create_webview(window, width, height, self.width) {
            Ok((wv, rx)) => {
                webview::push_schema(&wv, &engine.options);
                webview::push_panel_pinned(&wv, self.pinned);
                self.webview = Some(wv);
                self.action_rx = Some(rx);
                // Start the slide position at the correct spot.
                self.slide_x = width as f32;
                self.apply_layout(window);
            }
            Err(e) => {
                log::error!("Failed to create webview: {e}");
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

    /// Set the webview bounds directly.  Used by pinned mode and resize.
    pub(crate) fn apply_layout(&self, window: &Window) {
        let inner = window.inner_size();
        let Some(ref wv) = self.webview else {
            return;
        };

        if self.pinned {
            let _ = wv.set_bounds(webview::panel_bounds(
                inner.width,
                inner.height,
                self.width,
            ));
        }
        // Unpinned positioning is handled by `tick_slide`.
    }

    /// Advance the unpinned slide animation by `dt` seconds.
    /// Call every frame from `handle_redraw`.  Does nothing when pinned.
    pub(crate) fn tick_slide(&mut self, dt: f32, window: &Window) {
        if self.pinned {
            return;
        }
        let inner = window.inner_size();
        let Some(ref wv) = self.webview else {
            return;
        };

        let w = inner.width as f32;
        let margin = Self::PANEL_MARGIN as f32;

        // Where the panel should end up.
        let target = if self.peek {
            w - self.width as f32 - margin
        } else {
            w // off-screen right
        };

        // Lerp toward target.
        let diff = target - self.slide_x;
        if diff.abs() < Self::SNAP_PX {
            self.slide_x = target;
        } else {
            self.slide_x += diff * (Self::SLIDE_SPEED * dt).min(1.0);
        }

        #[allow(clippy::cast_possible_truncation)]
        let x = self.slide_x.round() as i32;
        let margin_i = Self::PANEL_MARGIN as i32;

        let _ = wv.set_bounds(wry::Rect {
            position: dpi::Position::Physical(dpi::PhysicalPosition::new(
                x, margin_i,
            )),
            size: dpi::Size::Physical(dpi::PhysicalSize::new(
                self.width.min(inner.width),
                inner.height.saturating_sub(Self::PANEL_MARGIN * 2),
            )),
        });
    }

    /// Check if the mouse is near the right edge and temporarily reveal the
    /// panel.
    pub(crate) fn update_peek(&mut self, mouse_x: f32, window: &Window) {
        if self.pinned {
            return;
        }
        let w = window.inner_size().width as f32;

        let near_edge = mouse_x >= (w - Self::EDGE_ZONE);
        let in_panel = mouse_x
            >= (w
                - self.width as f32
                - Self::PANEL_MARGIN as f32
                - Self::PEEK_GRACE);

        self.peek = near_edge || (self.peek && in_panel);
        // Slide is driven by `tick_slide` each frame.
    }

    /// Drain IPC actions from the webview, apply them to the engine, and
    /// handle panel toggle/resize.
    pub(crate) fn drain_and_apply(
        &mut self,
        engine: &mut VisoEngine,
        window: &Window,
    ) {
        // Poll background fetch result.
        self.poll_load_result(engine);

        let actions: Vec<UiAction> = self
            .action_rx
            .as_ref()
            .map(|rx| std::iter::from_fn(|| rx.try_recv().ok()).collect())
            .unwrap_or_default();

        let mut toggled = false;
        let mut resize_width: Option<u32> = None;

        for action in actions {
            match action {
                UiAction::SetOption { path, field, value } => {
                    log::debug!("SetOption: {path}.{field} = {value}");
                    let mut opts = engine.options.clone();
                    let Ok(mut root) = serde_json::to_value(&opts) else {
                        log::warn!("Failed to serialize options to JSON");
                        continue;
                    };
                    if let Some(section) = root.get_mut(&path) {
                        section[&field] = value;
                    } else {
                        log::warn!(
                            "Section '{path}' not found in options JSON"
                        );
                    }
                    match serde_json::from_value(root) {
                        Ok(updated) => opts = updated,
                        Err(e) => {
                            log::warn!("Options deserialization failed: {e}");
                        }
                    }
                    engine.set_options(opts);
                }
                UiAction::LoadFile { path } => {
                    self.load_local_file(engine, &path);
                }
                UiAction::FetchPdb { id, source } => {
                    self.start_fetch_pdb(&id, &source);
                }
                UiAction::OpenFileDialog => {
                    self.open_file_dialog(engine);
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
        engine: &VisoEngine,
    ) {
        let Some(ref wv) = self.webview else {
            return;
        };
        if now.duration_since(self.last_stats_push)
            >= Duration::from_millis(250)
        {
            let mut buffers = Vec::new();
            buffers.extend(engine.gpu.renderers.buffer_info());
            buffers.extend(engine.gpu.pick.selection.buffer_info());
            buffers.extend(engine.gpu.pick.residue_colors.buffer_info());
            webview::push_stats(wv, engine.fps(), &buffers);
            self.last_stats_push = now;
        }
    }
}

// ── Load handlers ────────────────────────────────────────────────────────

impl PanelController {
    /// Parse a local file and load its entities into the engine.
    fn load_local_file(&self, engine: &mut VisoEngine, path: &str) {
        log::info!("Loading local file: {path}");
        match parse_and_load(engine, path) {
            Ok(()) => {
                if let Some(ref wv) = self.webview {
                    let name =
                        std::path::Path::new(path).file_name().map_or_else(
                            || path.to_owned(),
                            |n| n.to_string_lossy().into_owned(),
                        );
                    webview::push_load_status(
                        wv,
                        "loaded",
                        &format!("Loaded {name}"),
                    );
                }
            }
            Err(msg) => {
                log::error!("Failed to load {path}: {msg}");
                if let Some(ref wv) = self.webview {
                    webview::push_load_status(wv, "error", &msg);
                }
            }
        }
    }

    /// Open a native file dialog and load the selected file.
    fn open_file_dialog(&self, engine: &mut VisoEngine) {
        let dialog = rfd::FileDialog::new()
            .add_filter("Structure", &["cif", "pdb"])
            .set_title("Open Structure File");

        if let Some(path) = dialog.pick_file() {
            let path_str = path.to_string_lossy().into_owned();
            self.load_local_file(engine, &path_str);
        }
    }

    /// Validate and start a background PDB fetch.
    fn start_fetch_pdb(&mut self, id: &str, source: &str) {
        let id = id.trim().to_lowercase();
        if id.len() != 4 || !id.chars().all(|c| c.is_ascii_alphanumeric()) {
            if let Some(ref wv) = self.webview {
                webview::push_load_status(
                    wv,
                    "error",
                    "PDB ID must be exactly 4 alphanumeric characters",
                );
            }
            return;
        }

        if let Some(ref wv) = self.webview {
            webview::push_load_status(
                wv,
                "loading",
                &format!("Fetching {}", id.to_uppercase()),
            );
        }

        let (tx, rx) = mpsc::channel();
        self.load_rx = Some(rx);

        let source = source.to_owned();
        let _ = std::thread::Builder::new().name("pdb-fetch".into()).spawn(
            move || {
                let _ = tx.send(fetch_pdb_blocking(&id, &source));
            },
        );
    }

    /// Poll the background fetch channel for a result.
    fn poll_load_result(&mut self, engine: &mut VisoEngine) {
        let Some(ref rx) = self.load_rx else {
            return;
        };
        let result = match rx.try_recv() {
            Ok(r) => r,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.load_rx = None;
                return;
            }
        };
        self.load_rx = None;

        match result {
            Ok(path) => self.load_local_file(engine, &path),
            Err(msg) => {
                log::error!("PDB fetch failed: {msg}");
                if let Some(ref wv) = self.webview {
                    webview::push_load_status(wv, "error", &msg);
                }
            }
        }
    }
}

// ── Helpers (free functions) ─────────────────────────────────────────────

/// Parse a structure file via `foldit-conv` and load entities into the
/// engine, replacing the current scene.
fn parse_and_load(engine: &mut VisoEngine, path: &str) -> Result<(), String> {
    use foldit_conv::adapters::pdb::structure_file_to_coords;
    use foldit_conv::types::entity::split_into_entities;

    let coords = structure_file_to_coords(std::path::Path::new(path))
        .map_err(|e| format!("Parse error: {e}"))?;
    let entities = split_into_entities(&coords);
    if entities.is_empty() {
        return Err("No entities found in file".into());
    }

    let _ = engine.replace_scene(entities);
    Ok(())
}

/// Download a PDB structure to the local cache. Returns the cached path.
///
/// Runs on a background thread — must not touch the engine or webview.
fn fetch_pdb_blocking(id: &str, source: &str) -> Result<String, String> {
    let models_dir = std::path::Path::new("assets/models");

    let (url, filename) = match source {
        "pdb-redo" => (
            format!("https://pdb-redo.eu/db/{id}/{id}_final.cif"),
            format!("{id}_redo.cif"),
        ),
        _ => (
            format!("https://files.rcsb.org/download/{id}.cif"),
            format!("{id}.cif"),
        ),
    };

    let local_path = models_dir.join(&filename);

    // Return cached copy if it exists.
    if local_path.exists() {
        return Ok(local_path.to_string_lossy().into_owned());
    }

    if !models_dir.exists() {
        std::fs::create_dir_all(models_dir)
            .map_err(|e| format!("Failed to create models directory: {e}"))?;
    }

    log::info!("Downloading {} from {source}...", id.to_uppercase());

    let agent = ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .timeout_global(Some(Duration::from_secs(30)))
            .build(),
    );
    let content = agent
        .get(&url)
        .call()
        .map_err(|e| format!("Network error downloading {id}: {e}"))?
        .into_body()
        .with_config()
        .limit(50 * 1024 * 1024)
        .read_to_string()
        .map_err(|e| format!("Failed to read response body: {e}"))?;

    std::fs::write(&local_path, &content)
        .map_err(|e| format!("I/O error saving CIF file: {e}"))?;

    log::info!("Downloaded to {}", local_path.display());
    Ok(local_path.to_string_lossy().into_owned())
}
