//! GPU-accelerated 3D protein visualization engine built on wgpu.
//!
//! Viso renders protein structures with real-time animation support,
//! post-processing (SSAO, bloom, FXAA), and GPU-based residue picking.
//!
//! # Key types
//!
//! - [`VisoEngine`] — the main rendering engine
//! - [`VisoCommand`] — action vocabulary (camera, selection, focus, etc.)
//! - [`VisoOptions`](options::VisoOptions) — runtime configuration (display,
//!   lighting, camera, colors)
//! - [`VisoError`] — error type
//! - [`InputProcessor`] — optional convenience for translating raw input events
//!   into [`VisoCommand`]s
//!
//! # Architecture
//!
//! The engine runs a background scene-processor thread that generates mesh
//! data off the main thread, delivering results via a lock-free triple
//! buffer. The main thread uploads prepared geometry to the GPU and
//! orchestrates a multi-pass pipeline: geometry → SSAO → bloom → composite
//! → FXAA.
//!
//! For integration guides and deep dives, see the companion mdBook
//! documentation.

pub(crate) mod animation;
pub(crate) mod camera;
pub(crate) mod engine;
pub(crate) mod error;
pub(crate) mod gpu;
pub(crate) mod input;
pub(crate) mod renderer;
pub(crate) mod scene;
pub(crate) mod util;

/// Runtime display, lighting, camera, and color options.
pub mod options;

#[cfg(feature = "viewer")]
pub mod viewer;

#[cfg(feature = "gui")]
pub mod gui;

// Animation (preset constructors only)
pub use animation::transition::Transition;
pub use engine::command::{BandInfo, BandType, PullInfo, VisoCommand};
pub use engine::VisoEngine;
pub use error::VisoError;
pub use gpu::render_context::RenderContext;
pub use gpu::texture::RenderTarget;
#[cfg(feature = "gui")]
pub use gui::webview::UiAction;
// Input (optional convenience)
pub use input::{InputEvent, InputProcessor, KeyBindings, MouseButton};
// Feature-gated
#[cfg(feature = "viewer")]
pub use viewer::{Viewer, ViewerBuilder};
