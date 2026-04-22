//! GUI panel controller — owns the wry webview and its state.
//!
//! Extracted from `viewer.rs` so that `ViewerApp` holds a
//! single `PanelController` field instead of six `#[cfg(feature = "gui")]`
//! fields.

use std::sync::mpsc;
use std::time::Duration;

use web_time::Instant;
use winit::window::Window;

use super::webview;
use crate::bridge::{self, PanelAxis, UiAction};
use crate::VisoEngine;

/// Owns the webview panel and all associated state.
pub(crate) struct PanelController {
    webview: Option<wry::WebView>,
    action_rx: Option<mpsc::Receiver<UiAction>>,
    last_stats_push: Instant,
    /// Whether the panel content is collapsed (only arrow visible).
    collapsed: bool,
    /// Current panel size in CSS (logical) pixels (width for Right,
    /// height for Bottom).  Converted to physical at the point of
    /// `set_bounds()`.
    size: u32,
    /// Current panel axis (right sidebar or bottom bar).
    axis: PanelAxis,
    /// Receiver for background PDB fetch results (path or error message).
    load_rx: Option<mpsc::Receiver<Result<String, String>>>,
}

// ── Constants ────────────────────────────────────────────────────────────

impl PanelController {
    /// Size of the arrow strip in CSS pixels (matches `.arrow-column`
    /// width/height in style.css).
    const ARROW_SIZE_CSS: u32 = 32;
}

// ── Construction ─────────────────────────────────────────────────────────

impl PanelController {
    /// Create a new controller with default state (expanded, no webview yet).
    pub(crate) fn new() -> Self {
        Self {
            webview: None,
            action_rx: None,
            last_stats_push: Instant::now(),
            collapsed: false,
            size: bridge::DEFAULT_PANEL_SIZE,
            axis: PanelAxis::Right,
            load_rx: None,
        }
    }

    /// Convert CSS panel size to physical pixels.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn physical_size(&self, scale: f64) -> u32 {
        (f64::from(self.size) * scale).round() as u32
    }

    /// Create the wry webview and push the initial schema to it.
    pub(crate) fn init_webview(
        &mut self,
        window: &Window,
        width: u32,
        height: u32,
        engine: &VisoEngine,
    ) {
        let physical = self.physical_size(window.scale_factor());
        self.axis = PanelAxis::from_dimensions(width, height);
        match webview::create_webview(window, width, height, physical) {
            Ok((wv, rx)) => {
                webview::push_schema(&wv, &engine.options);
                webview::push_orientation(&wv, self.axis);
                webview::push_panel_size_css(&wv, self.size);
                self.webview = Some(wv);
                self.action_rx = Some(rx);
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
    /// Toggle collapsed state. The arrow strip stays visible; only the
    /// settings content is hidden/shown.
    pub(crate) fn toggle(&mut self) {
        self.collapsed = !self.collapsed;
    }

    /// Recompute the axis from window dimensions and set the webview
    /// bounds. When collapsed, the webview shrinks to just the arrow
    /// strip so it doesn't block canvas input.
    pub(crate) fn apply_layout(&mut self, window: &Window) {
        let inner = window.inner_size();
        let new_axis = PanelAxis::from_dimensions(inner.width, inner.height);

        if new_axis != self.axis {
            self.axis = new_axis;
            if let Some(ref wv) = self.webview {
                webview::push_orientation(wv, self.axis);
            }
        }

        let Some(ref wv) = self.webview else {
            return;
        };

        let panel_physical = if self.collapsed {
            self.arrow_physical_size(window.scale_factor())
        } else {
            self.physical_size(window.scale_factor())
        };

        let _ = wv.set_bounds(webview::panel_bounds(
            inner.width,
            inner.height,
            panel_physical,
            self.axis,
        ));
    }

    /// Arrow strip size in physical pixels.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::unused_self
    )]
    fn arrow_physical_size(&self, scale: f64) -> u32 {
        (f64::from(Self::ARROW_SIZE_CSS) * scale).round() as u32
    }

    /// Drain IPC actions from the webview, apply them to the engine, and
    /// handle panel toggle/resize.
    pub(crate) fn drain_and_apply(
        &mut self,
        app: &mut crate::app::VisoApp,
        engine: &mut VisoEngine,
        window: &Window,
    ) {
        // Poll background fetch result.
        self.poll_load_result(app, engine);

        let actions: Vec<UiAction> = self
            .action_rx
            .as_ref()
            .map(|rx| std::iter::from_fn(|| rx.try_recv().ok()).collect())
            .unwrap_or_default();

        let mut toggled = false;
        let mut resize_width: Option<u32> = None;

        for action in actions {
            match action {
                UiAction::TogglePanel => toggled = true,
                UiAction::ResizePanel { size } => resize_width = Some(size),
                _ => self.apply_action(action, app, engine),
            }
        }

        if toggled {
            self.toggle();
            self.apply_layout(window);
        }
        if let Some(w) = resize_width {
            // The webview sends CSS pixels — clamp and store in CSS
            // units.  Conversion to physical happens at set_bounds().
            let clamped =
                w.clamp(bridge::MIN_PANEL_SIZE, bridge::MAX_PANEL_SIZE);
            if clamped != self.size {
                self.size = clamped;
                self.apply_layout(window);
                if let Some(ref wv) = self.webview {
                    webview::push_panel_size_css(wv, self.size);
                }
            }
        }
    }

    /// Apply a single UI action to the engine.
    fn apply_action(
        &mut self,
        action: UiAction,
        app: &mut crate::app::VisoApp,
        engine: &mut VisoEngine,
    ) {
        match action {
            UiAction::SetOption { path, field, value } => {
                log::debug!("SetOption: {path}.{field} = {value}");
                let mut opts = engine.options.clone();
                let Ok(mut root) = serde_json::to_value(&opts) else {
                    log::warn!("Failed to serialize options to JSON");
                    return;
                };
                // Build a JSON pointer from the dot-separated path + field.
                let pointer = format!(
                    "/{}",
                    path.split('.')
                        .chain(std::iter::once(field.as_str()))
                        .collect::<Vec<_>>()
                        .join("/")
                );
                if let Some(target) = root.pointer_mut(&pointer) {
                    *target = value;
                } else {
                    log::warn!("Option path not found: {pointer}");
                }
                match serde_json::from_value(root) {
                    Ok(updated) => opts = updated,
                    Err(e) => {
                        log::warn!("Options deserialization failed: {e}");
                    }
                }
                engine.set_options(opts);
                if let Some(ref wv) = self.webview {
                    webview::push_options(wv, &engine.options);
                }
                // Surface kind/opacity changes affect entity summaries
                self.push_scene_entities(engine);
            }
            UiAction::LoadFile { path } => {
                self.load_local_file(app, engine, &path);
            }
            UiAction::FetchPdb { id, source } => {
                self.start_fetch_pdb(&id, &source);
            }
            UiAction::OpenFileDialog => {
                self.open_file_dialog(app, engine);
            }
            UiAction::KeyPress { key } => {
                if let Some(cmd) =
                    crate::input::KeyBindings::default().lookup(&key)
                {
                    let _ = engine.execute(cmd);
                }
            }
            UiAction::Command(cmd) => {
                let _ = engine.execute(cmd);
                self.push_scene_entities(engine);
            }
            UiAction::SetEntityOption {
                entity_id,
                field,
                value,
            }
            | UiAction::SetEntityAppearance {
                entity_id,
                field,
                value,
            } => {
                Self::apply_entity_option(engine, entity_id, &field, &value);
                self.push_scene_entities(engine);
            }
            UiAction::ClearEntityOption { entity_id } => {
                engine.clear_entity_appearance(entity_id);
                self.push_scene_entities(engine);
            }
            UiAction::SetEntitySurface { entity_id, kind } => {
                let default_color = [0.7, 0.7, 0.7, 0.35];
                match kind.as_str() {
                    "gaussian" => {
                        engine.add_gaussian_surface(entity_id, default_color);
                    }
                    "ses" => {
                        engine.add_ses_surface(entity_id, default_color);
                    }
                    _ => {
                        engine.remove_entity_surface(entity_id);
                    }
                }
                self.push_scene_entities(engine);
            }
            UiAction::SetSurfaceOption {
                entity_id,
                field,
                value,
            } => {
                if let Some(v) = value.as_f64() {
                    let ch = match field.as_str() {
                        "color_r" => Some(0),
                        "color_g" => Some(1),
                        "color_b" => Some(2),
                        "opacity" => Some(3),
                        _ => None,
                    };
                    if let Some(ch) = ch {
                        engine
                            .set_surface_color_channel(entity_id, ch, v as f32);
                    }
                }
                self.push_scene_entities(engine);
            }
            UiAction::SetDensityOption { id, field, value } => {
                Self::apply_density_option(engine, id, &field, &value);
                self.push_density_maps(engine);
            }
            UiAction::RemoveDensityMap { id } => {
                engine.remove_density_map(id);
                self.push_density_maps(engine);
            }
            UiAction::ToggleDensityVisibility { id } => {
                let vis = engine.density.get(id).is_some_and(|e| !e.visible);
                engine.set_density_visible(id, vis);
                self.push_density_maps(engine);
            }
            UiAction::TogglePanel | UiAction::ResizePanel { .. } => {
                // Handled in drain_and_apply directly
            }
        }
    }

    /// Apply a single density option field.
    fn apply_density_option(
        engine: &mut VisoEngine,
        id: u32,
        field: &str,
        value: &serde_json::Value,
    ) {
        match field {
            "threshold" => {
                if let Some(v) = value.as_f64() {
                    engine.set_density_threshold(id, v as f32);
                }
            }
            "opacity" => {
                if let Some(v) = value.as_f64() {
                    engine.set_density_opacity(id, v as f32);
                }
            }
            "color_r" | "color_g" | "color_b" => {
                if let Some(v) = value.as_f64() {
                    let mut color = engine
                        .density
                        .get(id)
                        .map_or([0.3, 0.5, 0.8], |e| e.color);
                    match field {
                        "color_r" => color[0] = v as f32,
                        "color_g" => color[1] = v as f32,
                        "color_b" => color[2] = v as f32,
                        _ => {}
                    }
                    engine.set_density_color(id, color);
                }
            }
            _ => log::warn!("Unknown density field: {field}"),
        }
    }

    /// Apply a single per-entity appearance override field.
    fn apply_entity_option(
        engine: &mut VisoEngine,
        entity_id: u32,
        field: &str,
        value: &serde_json::Value,
    ) {
        let mut ovr = engine
            .entity_appearance(entity_id)
            .cloned()
            .unwrap_or_default();
        if let Err(unknown) = ovr.apply_json_field(field, value) {
            log::warn!("Unknown entity appearance field: {unknown}");
            return;
        }
        if ovr.is_empty() {
            engine.clear_entity_appearance(entity_id);
        } else {
            engine.set_entity_appearance(entity_id, ovr);
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

    /// Push the current entity list to the webview.
    pub(crate) fn push_scene_entities(&self, engine: &VisoEngine) {
        let Some(ref wv) = self.webview else {
            return;
        };
        let summaries = bridge::entity_summaries(engine);
        let json = serde_json::to_string(&summaries).unwrap_or_default();
        webview::push_scene_entities(wv, &json);
    }

    /// Push the current density map list to the webview.
    fn push_density_maps(&self, engine: &VisoEngine) {
        let Some(ref wv) = self.webview else {
            return;
        };
        let maps = bridge::density_summaries(engine);
        let json = serde_json::to_string(&maps).unwrap_or_default();
        webview::push_density_maps(wv, &json);
    }
}

// ── Load handlers ────────────────────────────────────────────────────────

impl PanelController {
    /// Parse a local file and load its entities into the engine.
    fn load_local_file(
        &self,
        app: &mut crate::app::VisoApp,
        engine: &mut VisoEngine,
        path: &str,
    ) {
        log::info!("Loading local file: {path}");
        match parse_and_load(app, engine, path) {
            Ok(()) => {
                self.push_scene_entities(engine);
                self.push_density_maps(engine);
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
    fn open_file_dialog(
        &self,
        app: &mut crate::app::VisoApp,
        engine: &mut VisoEngine,
    ) {
        let dialog = rfd::FileDialog::new()
            .add_filter("Structure", &["cif", "pdb", "ent", "bcif"])
            .add_filter("Density Map", &["mrc", "map", "ccp4"])
            .add_filter(
                "All Supported",
                &["cif", "pdb", "ent", "bcif", "mrc", "map", "ccp4"],
            )
            .set_title("Open File");

        if let Some(path) = dialog.pick_file() {
            let path_str = path.to_string_lossy().into_owned();
            self.load_local_file(app, engine, &path_str);
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
    fn poll_load_result(
        &mut self,
        app: &mut crate::app::VisoApp,
        engine: &mut VisoEngine,
    ) {
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
            Ok(path) => self.load_local_file(app, engine, &path),
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

/// Parse a file (structure or density) and load it into the engine.
fn parse_and_load(
    app: &mut crate::app::VisoApp,
    engine: &mut VisoEngine,
    path: &str,
) -> Result<(), String> {
    use crate::bridge;

    let ext = std::path::Path::new(path)
        .extension()
        .map_or("", |e| e.to_str().unwrap_or(""));

    if bridge::is_density_extension(ext) {
        let map = molex::adapters::mrc::mrc_file_to_density(
            std::path::Path::new(path),
        )
        .map_err(|e| format!("Density parse error: {e}"))?;
        let _ = engine.load_density_map(map);
        Ok(())
    } else {
        use molex::adapters::pdb::structure_file_to_entities;

        let entities = structure_file_to_entities(std::path::Path::new(path))
            .map_err(|e| format!("Parse error: {e}"))?;
        if entities.is_empty() {
            return Err("No entities found in file".into());
        }

        let _ = app.replace_scene(engine, entities);
        Ok(())
    }
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
