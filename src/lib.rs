//! GPU-accelerated 3D protein visualization engine built on wgpu.
//!
//! Viso renders protein structures with real-time animation support,
//! post-processing (SSAO, bloom, FXAA), and GPU-based residue picking.
//!
//! # Key entry points
//!
//! - [`engine::ProteinRenderEngine`] — the main rendering engine
//! - [`scene::Scene`] — the scene graph holding entity groups
//! - [`options::Options`] — runtime configuration (display, lighting, camera, colors)
//! - [`animation`] — behavior-driven structural animation system
//!
//! # Architecture
//!
//! The engine runs a background [`scene::processor::SceneProcessor`] thread that
//! generates mesh data off the main thread, delivering results via a lock-free
//! triple buffer. The main thread uploads prepared geometry to the GPU and
//! orchestrates a multi-pass pipeline: geometry → SSAO → bloom → composite → FXAA.
//!
//! For integration guides and deep dives, see the companion mdBook documentation.

pub mod animation;
pub mod camera;
pub mod engine;
pub mod gpu;
pub mod input;
pub mod options;
pub mod picking;
pub mod renderer;
pub mod scene;
pub mod util;
