# Architecture Overview

This chapter provides a high-level view of viso's architecture: how subsystems relate to each other, how data flows from file to screen, and how threading is organized.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
│  (foldit-rs window.rs / viso main.rs)                           │
│                                                                 │
│  winit events ──► InputProcessor ──► VisoCommand                │
│  IPC messages ──► backend handler ──► engine API calls          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         VisoEngine                              │
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ Scene      │  │ Animator   │  │ Camera     │  │ Picking    │ │
│  │            │  │            │  │ Controller │  │ System     │ │
│  │ Groups     │  │ Backbone   │  │ Arcball    │  │ GPU read   │ │
│  │ Entities   │  │ Sidechain  │  │ Animation  │  │ Selection  │ │
│  │ Focus      │  │ Per-entity │  │ Frustum    │  │ Bit-array  │ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘ │
│        │               │               │               │        │
│        ▼               ▼               │               │        │
│  ┌───────────────────────────┐         │               │        │
│  │  SceneProcessor           │         │               │        │
│  │  (background thread)      │         │               │        │
│  │                           │         │               │        │
│  │  Per-group mesh cache     │         │               │        │
│  │  Tube/ribbon/sidechain    │         │               │        │
│  │  Ball-and-stick/NA gen    │         │               │        │
│  └─────────────┬─────────────┘         │               │        │
│                │ triple buffer         │               │        │
│                ▼                       ▼               ▼        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                      Renderers                            │  │
│  │                                                           │  │
│  │  Molecular:                   Post-Processing:            │  │
│  │  ├─ BackboneRenderer          ├─ SSAO                     │  │
│  │  │  (tubes + ribbons)         ├─ Bloom                    │  │
│  │  ├─ SidechainRenderer         ├─ Composite                │  │
│  │  ├─ BallAndStickRenderer      └─ FXAA                     │  │
│  │  ├─ BandRenderer                                          │  │
│  │  ├─ PullRenderer              ShaderComposer:             │  │
│  │  └─ NucleicAcidRenderer       └─ naga_oil composition     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│                    ┌──────────────┐                             │
│                    │ RenderContext│                             │
│                    │ wgpu device  │                             │
│                    │ queue        │                             │
│                    │ surface      │                             │
│                    └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## High-Level Data Flow

```
┌───────────────────────────────────────────────────────────┐
│                     INITIALIZATION                        │
│                                                           │
│  File path (.cif/.pdb/.bcif)  ──or──  Vec<MoleculeEntity> │
│         │                                    │            │
│         ▼                                    │            │
│  foldit_conv::parse ──► Vec<MoleculeEntity> ◄┘            │
│                                │                          │
│                                ▼                          │
│                     Engine stores as                      │
│                    SOURCE OF TRUTH                        │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                         SCENE                             │
│                                                           │
│  The "live" renderable state of the world.                │
│  Positions, SS types, colors, sidechain topology —        │
│  everything needed to produce geometry.                   │
│                                                           │
│  Dirty-flagged: only rebuilds geometry when changed.      │
│  During animation: reflects interpolated state.           │
│  When animation completes: matches source of truth.       │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                       RENDERER                            │
│                                                           │
│  Consumes Scene read-only, produces GPU data.             │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ renderer::geometry                                  │  │
│  │                                                     │  │
│  │ Scene data ──► meshes + impostor instances          │  │
│  │ (tubes, ribbons, capsules, ball-and-stick, NA)      │  │
│  └──────────────────────┬──────────────────────────────┘  │
│                         │                                 │
│                         ▼                                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ GPU Passes                                          │  │
│  │                                                     │  │
│  │ 1. Geometry pass (color + normals + depth)          │  │
│  │ 2. Picking pass (object ID readback)                │  │
│  │ 3. Post-process (SSAO, bloom, fog, FXAA)            │  │
│  │           │                                         │  │
│  │           ▼                                         │  │
│  │ Final 2D screen-space texture                       │  │
│  └─────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                   OUTPUT / EMBEDDING                      │
│                                                           │
│  The final texture is consumed by the host:               │
│    • winit window (current viewer)                        │
│    • HTML canvas (wasm / web embed)                       │
│    • PNG snapshot (headless)                              │
│    • dioxus / egui / any framework with a texture slot    │
│                                                           │
│  The engine produces a texture; the consumer decides      │
│  what to do with it.                                      │
└───────────────────────────────────────────────────────────┘
```

## Data Flow: File to Screen

### 1. Parsing

```
PDB/CIF file → foldit_conv → Coords → MoleculeEntity
```

The `foldit_conv` crate parses mmCIF files into `Coords` structs (atom positions, names, chains, residue info). These are wrapped in `MoleculeEntity` values with a molecule type classification.

### 2. Scene Organization

```
MoleculeEntity → EntityGroup → Scene
```

Entities are grouped into `EntityGroup` values. The scene maintains insertion order, visibility state, focus tracking, and a generation counter for dirty detection.

### 3. Background Mesh Generation

```
Scene → PerGroupData → SceneProcessor → PreparedScene
```

When the scene is dirty, `per_group_data()` collects render data for each visible group. This is submitted to the background thread via `SceneRequest::FullRebuild`. The processor generates (or retrieves cached) meshes for each group, concatenates them, and writes a `PreparedScene` to the triple buffer.

### 4. GPU Upload

```
PreparedScene → queue.write_buffer() → GPU buffers
```

The main thread picks up the `PreparedScene` and writes raw byte arrays directly to GPU buffers. This is a memcpy-level operation, typically under 1ms.

### 5. Rendering

```
GPU buffers → Geometry Pass → Post-Processing → Swapchain
```

All molecular renderers draw to HDR render targets. The post-processing stack applies SSAO, bloom, compositing (outlines, fog, tone mapping), and FXAA before presenting to the swapchain.

## Threading Model

Viso uses two threads with lock-free communication:

### Main Thread

Owns all GPU resources and runs the render loop. Responsibilities:

- Processing input events (mouse, keyboard, IPC)
- Managing the scene (add/remove groups, update entities)
- Running animation (update each frame, get interpolated state)
- Submitting scene requests to the background thread (non-blocking)
- Picking up completed meshes from the triple buffer (non-blocking)
- Uploading data to the GPU
- Executing the render pipeline
- Handling GPU picking readback

The main thread **never blocks** on the background thread. If meshes aren't ready, it renders the previous frame's data.

### Background Thread

Owns the mesh cache and performs CPU-intensive work:

- Receiving scene requests via `mpsc::Receiver` (blocks when idle)
- Generating tube, ribbon, sidechain, ball-and-stick, and nucleic acid meshes
- Maintaining a per-group mesh cache with version-based invalidation
- Coalescing queued requests to skip stale intermediates
- Writing results to triple buffers (non-blocking)

### Lock-Free Bridge

| Mechanism               | Direction         | Semantics                                           |
| ----------------------- | ----------------- | --------------------------------------------------- |
| `mpsc::channel`         | Main → Background | Submit requests (non-blocking send)                 |
| `triple_buffer` (scene) | Background → Main | Latest `PreparedScene` (non-blocking read)          |
| `triple_buffer` (anim)  | Background → Main | Latest `PreparedAnimationFrame` (non-blocking read) |

Triple buffers guarantee:

- The writer always has a buffer to write to (never blocks)
- The reader always gets the latest completed result
- No data races or mutex contention

## Module Structure

```
viso/src/
├── lib.rs                  # Public API (flat re-exports only)
├── main.rs                 # Standalone viewer binary
├── engine/                 # Core coordinator: frame loop, command dispatch, subsystem wiring
├── scene/                  # Entity storage, groups, visibility, SS overrides, dirty flagging
├── animation/              # Structural animation system
│   ├── behaviors/          # How animations look (snap, smooth, collapse/expand, cascade)
│   └── animator/           # State machines that execute animations (runner, controller, preemption)
├── input/                  # Raw window events → VisoCommand conversion
├── options/                # TOML-serializable runtime options (lighting, camera, colors, display)
├── camera/                 # Orbital camera controller, animated transitions, frustum culling
├── gpu/                    # wgpu device/surface init, dynamic buffers, lighting, shader composition
├── renderer/               # GPU rendering pipeline
│   ├── geometry/           # Scene data → mesh/impostor generation
│   ├── picking/            # GPU-based object picking + readback
│   └── postprocess/        # SSAO, bloom, composite, FXAA
├── viewer.rs               # Standalone winit viewer (feature-gated)
├── gui/                    # Webview options panel (feature-gated)
└── util/                   # Easing curves, frame timing, trajectory playback, bond topology
```

## Key Design Decisions

### Why Background Processing?

Mesh generation for complex proteins (>1000 residues) can take 20-40ms. At 60fps, that's most of the frame budget. By offloading to a background thread:

- The main thread maintains smooth rendering
- GPU upload is <1ms (raw buffer writes)
- The background thread can take as long as it needs without dropping frames

### Why Triple Buffers?

Triple buffers provide lock-free communication:

- The background thread always has a buffer to write to
- The main thread always reads the latest result
- No mutexes, no contention, no blocking on either side

The cost is memory (3x the buffer size), but mesh data is typically 1-10MB, so this is negligible.

### Why Per-Group Caching?

Molecular scenes often have multiple groups where only one changes at a time (e.g., Rosetta updates one group while others stay static). Per-group caching with version-based invalidation means only the changed group's meshes are regenerated. For a 3-group scene, this can save 60-80% of generation time.

### Why Capsule Impostors?

Sidechains and ball-and-stick atoms use ray-marched impostor rendering instead of mesh-based spheres and cylinders:

- **Memory**: a capsule is 48 bytes vs. hundreds of bytes for a mesh sphere
- **Quality**: impostors are pixel-perfect at any zoom level
- **Performance**: GPU ray-marching is efficient for the simple SDF shapes (spheres, capsules, cones)
