// -- Lint policy ---------------------------------------------------------
// This is the single source of truth for crate-wide lints.

// Broad lint groups (covers all, pedantic, nursery sub-lints)
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(clippy::nursery)]
// Cargo lints (warn, not deny since cargo lints can be noisy)
#![deny(clippy::cargo)]
// Documentation
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::private_intra_doc_links)]
#![deny(rustdoc::bare_urls)]
// Restriction lints (opt-in, not covered by the groups above)
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::dbg_macro)]
#![deny(clippy::print_stdout)]
#![deny(clippy::print_stderr)]
#![deny(clippy::str_to_string)]
// Rustc lints
#![deny(unused_results)]
#![deny(unused_qualifications)]
#![deny(trivial_casts)]
#![deny(trivial_numeric_casts)]
// ── GPU / graphics allowances ──
// Casts: GPU code constantly converts between usize/u32/f32/f64 for indices,
// buffer sizes, and shader uniforms.  These are intentional and safe.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
// Float comparison: graphics math frequently compares against 0.0, 1.0, etc.
#![allow(clippy::float_cmp)]
// Multiple crate versions: transitive deps (objc2, syn, etc.) — not actionable
#![allow(clippy::multiple_crate_versions)]
// ── Pedantic allowances ──
// mul_add: wgpu shader math reads more clearly as a*b+c
#![allow(clippy::suboptimal_flops)]
// const fn: adding const to simple accessors adds noise for no runtime benefit
#![allow(clippy::missing_const_for_fn)]
// Pipeline/texture descriptor defaults: Default::default() is idiomatic wgpu
#![allow(clippy::default_trait_access)]
// Identifier-like words in doc comments don't all need backticks
#![allow(clippy::doc_markdown)]
// Similar names are common in graphics code (e.g. view / view_matrix)
#![allow(clippy::similar_names)]
// Items after statements (helper fns inside long impls)
#![allow(clippy::items_after_statements)]
// Self vs type name in return position
#![allow(clippy::use_self)]
// pub(crate) inside private modules is a stylistic choice about explicit
// visibility
#![allow(clippy::redundant_pub_crate)]

//! GPU-accelerated 3D protein visualization engine built on wgpu.
//!
//! Viso renders protein structures with real-time animation support,
//! post-processing (SSAO, bloom, FXAA), and GPU-based residue picking.
//!
//! # Key entry points
//!
//! - [`ProteinRenderEngine`] — the main rendering engine
//! - [`Scene`] — the scene graph holding entities
//! - [`Options`](options::Options) — runtime configuration (display, lighting,
//!   camera, colors)
//! - [`InputEvent`] — platform-agnostic input forwarding
//! - [`Transition`] — animation transition for scene sync
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
/// Core rendering engine and frame loop.
pub mod engine;
/// Crate-level error types.
pub mod error;
pub(crate) mod gpu;
#[cfg(feature = "gui")]
/// Native-side GUI: wry webview hosting the viso-ui WASM bundle.
pub mod gui;
pub(crate) mod input;
/// Runtime display, lighting, camera, and color options.
pub mod options;
pub(crate) mod picking;
pub(crate) mod renderer;
/// Scene graph holding entities.
pub mod scene;
pub(crate) mod util;
#[cfg(feature = "viewer")]
/// Standalone visualization window backed by winit.
pub mod viewer;

// ── Public re-exports from internal modules ──

// Input types — platform-agnostic event API
// Animation types for scene sync and custom behaviors
pub use animation::{
    behaviors::{
        shared, AnimationBehavior, BackboneThenExpand, Cascade, CollapseExpand,
        PreemptionStrategy, SharedBehavior, SmoothInterpolation, Snap,
    },
    transition::Transition,
};
// Convenience re-exports
pub use engine::ProteinRenderEngine;
pub use error::VisoError;
/// GPU render context (device, queue, surface, config).
pub use gpu::render_context::RenderContext;
pub use gpu::texture::RenderTarget;
#[cfg(feature = "gui")]
pub use gui::webview::UiAction;
pub use input::{InputEvent, KeyAction, MouseButton};
// Renderer data types needed by consumers for band/pull updates
pub use renderer::molecular::band::BandRenderInfo;
pub use renderer::molecular::pull::PullRenderInfo;
pub use scene::{Focus, Scene, SceneEntity};
pub use util::easing::EasingFunction;
#[cfg(feature = "viewer")]
pub use viewer::{Viewer, ViewerBuilder};
