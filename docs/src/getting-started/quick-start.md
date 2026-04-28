# Quick Start

Viso is a library first. With no feature flags enabled, it gives you
`VisoEngine` — a self-contained rendering engine you embed in your own
event loop. The optional `viewer` feature adds a standalone winit
window for quick prototyping; `gui` adds an embedded webview options
panel; `binary` (default) builds the CLI.

## Using Viso as a Library

Add viso to your `Cargo.toml`:

```toml
[dependencies]
viso = { path = "../viso", default-features = false }
pollster = "0.4"  # for blocking on async GPU init
```

The minimal integration has three parts: build a `VisoApp` (which owns
the authoritative `Assembly`), build a `VisoEngine` from the matching
`AssemblyConsumer`, and run a render loop.

### 1. Create the App + Engine Pair

`VisoApp` is the host of the structural state. The engine reads
snapshots through an `AssemblyConsumer` triple buffer.

```rust
use viso::{RenderContext, VisoApp, VisoEngine};
use viso::options::VisoOptions;

// Empty scene — useful when you want to load entities later.
let (mut app, consumer) = VisoApp::new_empty();

// Or load straight from a structure file (.cif / .pdb / .bcif):
let (mut app, consumer) = VisoApp::from_file("path/to/structure.cif")?;

// Or from in-memory bytes with a format hint:
let (mut app, consumer) = VisoApp::from_bytes(&bytes, "cif")?;

let context = pollster::block_on(
    RenderContext::new(window.clone(), (width, height))
)?;
let mut engine = VisoEngine::new(context, consumer, VisoOptions::default())?;
```

### 2. Load or Replace Entities

All structural mutations route through `VisoApp`, which publishes
snapshots that the engine picks up on the next `update()`.

```rust
// entities: Vec<MoleculeEntity> from molex or your own pipeline
let ids = app.load_entities(&mut engine, entities, true); // fit camera

// Replace the whole scene:
app.replace_scene(&mut engine, new_entities);
```

### 3. Render Loop

Each frame, call `update` then `render`:

```rust
engine.update(dt);       // poll assembly snapshots, advance animation,
                         // apply pending background mesh data
match engine.render() {
    Ok(()) => {}
    Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
        engine.resize(width, height);
    }
    Err(e) => log::error!("render error: {e:?}"),
}
```

The engine handles background mesh generation, animation, and the full
post-processing pipeline internally. You own the event loop and the
window.

### Input (Optional)

`InputProcessor` is a convenience layer that translates raw input
events into `VisoCommand` values. You can use it or wire commands
directly:

```rust
use viso::{InputProcessor, InputEvent};

let mut input = InputProcessor::new();

// In your event handler:
let event = InputEvent::CursorMoved { x, y };
if let Some(cmd) = input.handle_event(event, engine.hovered_target()) {
    let _ = engine.execute(cmd);
}
```

## Standalone Viewer

For quick prototyping, enable the `viewer` feature:

```toml
[dependencies]
viso = { path = "../viso", features = ["viewer"] }
```

This pulls in `winit` and `pollster` and gives you `Viewer`, which
handles window creation, the event loop, input wiring, and the render
loop:

```rust
use viso::Viewer;

Viewer::builder()
    .with_path("assets/models/4pnk.cif")
    .build()
    .run()?;
```

### Running the CLI

The `binary` feature (enabled by default) builds a standalone CLI that
can download structures from RCSB by PDB code:

```sh
cargo run -p viso -- 1ubq
```

This downloads the CIF file, caches it in `assets/models/`, and opens
a viewer window. You can also pass a local file path:

```sh
cargo run -p viso -- path/to/structure.cif
```
