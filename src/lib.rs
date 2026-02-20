// -- Lint policy ---------------------------------------------------------
// This is the single source of truth for crate-wide lints.

// Broad lint groups
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(clippy::nursery)]
// Documentation
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::private_intra_doc_links)]
#![deny(rustdoc::bare_urls)]
// No panicking in library code
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
// No debug/print artifacts
#![deny(clippy::dbg_macro)]
#![deny(clippy::print_stdout)]
#![deny(clippy::print_stderr)]
// Import hygiene
#![deny(clippy::wildcard_imports)]
// Complexity limits (thresholds in clippy.toml)
#![deny(clippy::cognitive_complexity)]
#![deny(clippy::too_many_lines)]
#![deny(clippy::excessive_nesting)]
// Function signature hygiene
#![deny(clippy::too_many_arguments)]
#![deny(clippy::fn_params_excessive_bools)]
// Clone / pass-by-value hygiene
#![deny(clippy::needless_pass_by_value)]
#![deny(clippy::implicit_clone)]
// String hygiene
#![deny(clippy::inefficient_to_string)]
#![deny(clippy::redundant_closure_for_method_calls)]
#![deny(clippy::manual_string_new)]
#![deny(clippy::str_to_string)]
// Cargo lints (warn, not deny since cargo lints can be noisy)
#![warn(clippy::cargo)]
// Unused / redundant code
#![deny(unused_results)]
#![deny(unused_qualifications)]
// Cast hygiene
#![deny(trivial_casts)]
#![deny(trivial_numeric_casts)]

//! GPU-accelerated 3D protein visualization engine built on wgpu.
//!
//! Viso renders protein structures with real-time animation support,
//! post-processing (SSAO, bloom, FXAA), and GPU-based residue picking.
//!
//! # Key entry points
//!
//! - [`engine::ProteinRenderEngine`] - the main rendering engine
//! - [`scene::Scene`] - the scene graph holding entity groups
//! - [`options::Options`] - runtime configuration (display, lighting, camera,
//!   colors)
//! - [`animation`] - behavior-driven structural animation system
//!
//! # Architecture
//!
//! The engine runs a background [`scene::processor::SceneProcessor`] thread
//! that generates mesh data off the main thread, delivering results via a
//! lock-free triple buffer. The main thread uploads prepared geometry to the
//! GPU and orchestrates a multi-pass pipeline: geometry → SSAO → bloom →
//! composite → FXAA.
//!
//! For integration guides and deep dives, see the companion mdBook
//! documentation.

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
