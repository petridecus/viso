//! Native-side GUI layer: wry webview hosting the viso-ui WASM bundle.
//!
//! The webview is created as a child of the winit window and communicates
//! with the engine via a minimal JSON IPC bridge.

/// Wry webview creation, IPC handler, and state push helpers.
pub mod webview;
