//! Native-side GUI layer: wry webview hosting the viso-ui WASM bundle.
//!
//! The webview is created as a child of the winit window and communicates
//! with the engine via a minimal JSON IPC bridge.

/// GUI panel controller that owns the webview and its state.
pub(crate) mod panel;

/// Wry webview creation, IPC handler, and state push helpers.
pub mod webview;
