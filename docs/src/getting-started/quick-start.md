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

The minimal integration has three parts: build a `VisoEngine`, push a
`molex::Assembly` snapshot to it, and run a render loop. You own the
`Assembly` directly using molex's APIs.

### 1. Build the Engine and Push an Assembly

```rust
use std::sync::Arc;
use viso::{RenderContext, VisoEngine};
use viso::options::VisoOptions;
use molex::{Assembly, MoleculeEntity};

let context = pollster::block_on(
    RenderContext::new(window.clone(), (width, height))
)?;
let mut engine = VisoEngine::new(context, VisoOptions::default())?;

// You own the Assembly. After every mutation, push the latest
// snapshot via engine.set_assembly. The engine drains it on the
// next update tick.
let mut assembly = Assembly::new(entities);
engine.set_assembly(Arc::new(assembly.clone()));
```

### 2. Mutate and Re-publish

Mutate your `Assembly` through molex's APIs (`add_entity`,
`remove_entity`, `update_positions`, etc.), then push the new snapshot
to viso:

```rust
assembly.add_entity(new_entity);
assembly.update_positions(eid, &new_coords);

engine.set_assembly(Arc::new(assembly.clone()));
```

The engine generation-checks each push, so re-publishing without an
actual change is a no-op.

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

## Standalone Viewer (separate use case)

If you want to run viso *as* a standalone application — not embed it
in your own library — there's a built-in `Viewer` for quick
prototyping. This is a separate use case from library embedding;
library users should not enable these features.

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

Internally, the standalone viewer uses a helper called `VisoApp` to
own its own `Assembly`. **`VisoApp` is not part of the library API**
— it exists solely so viso can be its own host when run standalone.
Library consumers own their own `molex::Assembly` and call
`engine.set_assembly` directly, never going through `VisoApp`.

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
