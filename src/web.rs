//! WASM entry point for running Viso in a browser via WebGPU.
//!
//! Exports two `wasm_bindgen` functions:
//! - [`init`] — sets up logging and panic hooks.
//! - [`start`] — creates a `VisoEngine` from an `HtmlCanvasElement` and
//!   structure file bytes, then enters a `requestAnimationFrame` render loop.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

use crate::gpu::RenderContext;
use crate::input::{InputEvent, InputProcessor, MouseButton};
use crate::VisoEngine;

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/// Initialize the WASM environment: console logging and panic hook.
///
/// Call this once before [`start`].
#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
}

// ---------------------------------------------------------------------------
// Engine start
// ---------------------------------------------------------------------------

/// Create and start a `VisoEngine` rendering to the given canvas.
///
/// `format_hint` is the file extension used to select the parser:
/// `"cif"`, `"pdb"`, or `"bcif"`.
///
/// # Errors
///
/// Returns a `JsValue` error string if engine initialization fails.
#[wasm_bindgen]
pub async fn start(
    canvas: HtmlCanvasElement,
    structure_bytes: &[u8],
    format_hint: &str,
) -> Result<(), JsValue> {
    let width = canvas.client_width().max(1) as u32;
    let height = canvas.client_height().max(1) as u32;

    // Parse structure from bytes
    let entities = parse_structure_bytes(structure_bytes, format_hint)
        .map_err(|e| JsValue::from_str(&e))?;

    // Create RenderContext from canvas via wgpu::SurfaceTarget::Canvas.
    let surface_target = wgpu::SurfaceTarget::Canvas(canvas.clone());
    let context = RenderContext::new(surface_target, (width, height))
        .await
        .map_err(|e| JsValue::from_str(&format!("GPU init failed: {e}")))?;

    // Build engine with empty scene, then load entities
    let mut engine = VisoEngine::new_empty(context)
        .map_err(|e| JsValue::from_str(&format!("Engine init failed: {e}")))?;

    let _ids = engine.load_entities(entities, true);

    // Kick off the rAF loop
    let engine = Rc::new(RefCell::new(engine));
    let input = Rc::new(RefCell::new(InputProcessor::new()));

    attach_input_listeners(&canvas, Rc::clone(&engine), Rc::clone(&input));
    request_animation_frame_loop(Rc::clone(&engine));

    Ok(())
}

// ---------------------------------------------------------------------------
// Structure parsing from in-memory bytes
// ---------------------------------------------------------------------------

/// Parse a structure file from raw bytes, selecting the parser based on the
/// file extension hint.
fn parse_structure_bytes(
    bytes: &[u8],
    format_hint: &str,
) -> Result<Vec<molex::types::entity::MoleculeEntity>, String> {
    let hint = format_hint.to_ascii_lowercase();
    let hint = hint.trim_start_matches('.');
    match hint {
        "cif" | "mmcif" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|e| format!("Invalid UTF-8 in CIF: {e}"))?;
            molex::adapters::pdb::mmcif_str_to_entities(text)
                .map_err(|e| format!("CIF parse error: {e}"))
        }
        "pdb" | "ent" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|e| format!("Invalid UTF-8 in PDB: {e}"))?;
            molex::adapters::pdb::pdb_str_to_entities(text)
                .map_err(|e| format!("PDB parse error: {e}"))
        }
        "bcif" => molex::adapters::bcif::bcif_to_entities(bytes)
            .map_err(|e| format!("BinaryCIF parse error: {e}")),
        other => Err(format!(
            "Unsupported format '{other}'. Use 'cif', 'pdb', or 'bcif'."
        )),
    }
}

// ---------------------------------------------------------------------------
// requestAnimationFrame render loop
// ---------------------------------------------------------------------------

/// Start a `requestAnimationFrame` loop that drives `engine.update()` +
/// `engine.render()` each frame.
fn request_animation_frame_loop(engine: Rc<RefCell<VisoEngine>>) {
    // The closure must be able to schedule itself for the next frame.
    // We use an Rc<RefCell<Option<Closure>>> to allow the closure to
    // reference itself indirectly.
    let closure_holder: Rc<RefCell<Option<Closure<dyn FnMut()>>>> =
        Rc::new(RefCell::new(None));
    let holder_for_closure = Rc::clone(&closure_holder);

    let dt = 1.0 / 60.0_f32; // approximate; real timing can be added later

    let cb = Closure::<dyn FnMut()>::new(move || {
        {
            let mut eng = engine.borrow_mut();
            eng.update(dt);
            match eng.render() {
                Ok(()) => {}
                Err(e) => {
                    log::error!("render error: {e:?}");
                }
            }
        }
        // Schedule next frame
        let holder = holder_for_closure.borrow();
        if let Some(ref cb) = *holder {
            let _ = request_animation_frame(cb);
        }
    });

    // Kick off the first frame
    let _ = request_animation_frame(&cb);

    // Store the closure so it is not dropped
    *closure_holder.borrow_mut() = Some(cb);
}

/// Schedule a callback via `window.requestAnimationFrame`.
fn request_animation_frame(
    callback: &Closure<dyn FnMut()>,
) -> Result<i32, JsValue> {
    web_sys::window()
        .ok_or_else(|| JsValue::from_str("no global window"))?
        .request_animation_frame(callback.as_ref().unchecked_ref())
}

// ---------------------------------------------------------------------------
// DOM input event forwarding
// ---------------------------------------------------------------------------

/// Attach mouse/wheel/keyboard listeners on the canvas and forward them
/// through the `InputProcessor` into the `VisoEngine`.
fn attach_input_listeners(
    canvas: &HtmlCanvasElement,
    engine: Rc<RefCell<VisoEngine>>,
    input: Rc<RefCell<InputProcessor>>,
) {
    // ── Mouse move ──
    {
        let engine = Rc::clone(&engine);
        let input = Rc::clone(&input);
        let cb =
            Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
                let x = event.offset_x() as f32;
                let y = event.offset_y() as f32;
                let evt = InputEvent::CursorMoved { x, y };
                let mut eng = engine.borrow_mut();
                eng.set_cursor_pos(x, y);
                if let Some(cmd) =
                    input.borrow_mut().handle_event(evt, eng.hovered_target())
                {
                    let _ = eng.execute(cmd);
                }
            });
        let _ = canvas.add_event_listener_with_callback(
            "mousemove",
            cb.as_ref().unchecked_ref(),
        );
        cb.forget();
    }

    // ── Mouse down ──
    {
        let engine = Rc::clone(&engine);
        let input = Rc::clone(&input);
        let cb =
            Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
                let button = match event.button() {
                    2 => MouseButton::Right,
                    1 => MouseButton::Middle,
                    _ => MouseButton::Left,
                };
                let evt = InputEvent::MouseButton {
                    button,
                    pressed: true,
                };
                let mut eng = engine.borrow_mut();
                if let Some(cmd) =
                    input.borrow_mut().handle_event(evt, eng.hovered_target())
                {
                    let _ = eng.execute(cmd);
                }
            });
        let _ = canvas.add_event_listener_with_callback(
            "mousedown",
            cb.as_ref().unchecked_ref(),
        );
        cb.forget();
    }

    // ── Mouse up ──
    {
        let engine = Rc::clone(&engine);
        let input = Rc::clone(&input);
        let cb =
            Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
                let button = match event.button() {
                    2 => MouseButton::Right,
                    1 => MouseButton::Middle,
                    _ => MouseButton::Left,
                };
                let evt = InputEvent::MouseButton {
                    button,
                    pressed: false,
                };
                let mut eng = engine.borrow_mut();
                if let Some(cmd) =
                    input.borrow_mut().handle_event(evt, eng.hovered_target())
                {
                    let _ = eng.execute(cmd);
                }
            });
        let _ = canvas.add_event_listener_with_callback(
            "mouseup",
            cb.as_ref().unchecked_ref(),
        );
        cb.forget();
    }

    // ── Wheel (scroll/zoom) ──
    {
        let engine = Rc::clone(&engine);
        let input = Rc::clone(&input);
        let cb =
            Closure::<dyn FnMut(_)>::new(move |event: web_sys::WheelEvent| {
                event.prevent_default();
                let delta = -(event.delta_y() as f32) * 0.01;
                let evt = InputEvent::Scroll { delta };
                let mut eng = engine.borrow_mut();
                if let Some(cmd) =
                    input.borrow_mut().handle_event(evt, eng.hovered_target())
                {
                    let _ = eng.execute(cmd);
                }
            });
        // Use non-passive listener so we can `preventDefault`
        let opts = web_sys::AddEventListenerOptions::new();
        opts.set_passive(false);
        let _ = canvas
            .add_event_listener_with_callback_and_add_event_listener_options(
                "wheel",
                cb.as_ref().unchecked_ref(),
                &opts,
            );
        cb.forget();
    }

    // ── Context menu (prevent default right-click menu) ──
    {
        let cb =
            Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
                event.prevent_default();
            });
        let _ = canvas.add_event_listener_with_callback(
            "contextmenu",
            cb.as_ref().unchecked_ref(),
        );
        cb.forget();
    }
}
