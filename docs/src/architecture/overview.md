# Architecture Overview

This chapter provides a high-level view of viso's architecture: how subsystems relate to each other, how data flows from file to screen, and how threading is organized.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│  (foldit-rs window.rs / viso main.rs)                           │
│                                                                  │
│  winit events → input handling → engine API calls                │
│  IPC messages → backend handler → engine API calls               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ProteinRenderEngine                            │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐  │
│  │  Scene    │  │ Animator  │  │  Camera    │  │   Picking    │  │
│  │          │  │          │  │ Controller │  │              │  │
│  │ Groups   │  │ Behaviors│  │ Arcball    │  │ GPU readback │  │
│  │ Entities │  │ Interp.  │  │ Animation  │  │ Selection    │  │
│  │ Focus    │  │ Sidechain│  │ Frustum    │  │ Bit-array    │  │
│  └────┬─────┘  └────┬─────┘  └────┬───────┘  └──────┬───────┘  │
│       │              │             │                  │          │
│       ▼              ▼             │                  │          │
│  ┌─────────────────────────┐      │                  │          │
│  │    SceneProcessor       │      │                  │          │
│  │    (background thread)  │      │                  │          │
│  │                         │      │                  │          │
│  │  Per-group mesh cache   │      │                  │          │
│  │  Tube/ribbon/sidechain  │      │                  │          │
│  │  Ball-and-stick/NA gen  │      │                  │          │
│  └───────────┬─────────────┘      │                  │          │
│              │ triple buffer      │                  │          │
│              ▼                    ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Renderers                             │   │
│  │                                                         │   │
│  │  Molecular:                    Post-Processing:         │   │
│  │  ├─ TubeRenderer              ├─ SSAO                   │   │
│  │  ├─ RibbonRenderer            ├─ Bloom                  │   │
│  │  ├─ CapsuleSidechainRenderer  ├─ Composite              │   │
│  │  ├─ BallAndStickRenderer      └─ FXAA                   │   │
│  │  ├─ BandRenderer                                        │   │
│  │  ├─ PullRenderer              ShaderComposer:           │   │
│  │  └─ NucleicAcidRenderer       └─ naga_oil composition   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│                    ┌─────────────┐                              │
│                    │ RenderContext│                              │
│                    │ wgpu device │                              │
│                    │ queue       │                              │
│                    │ surface     │                              │
│                    └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
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

| Mechanism | Direction | Semantics |
|-----------|-----------|-----------|
| `mpsc::channel` | Main → Background | Submit requests (non-blocking send) |
| `triple_buffer` (scene) | Background → Main | Latest `PreparedScene` (non-blocking read) |
| `triple_buffer` (anim) | Background → Main | Latest `PreparedAnimationFrame` (non-blocking read) |

Triple buffers guarantee:
- The writer always has a buffer to write to (never blocks)
- The reader always gets the latest completed result
- No data races or mutex contention

## Module Structure

```
viso/src/
├── lib.rs                  # Module declarations
├── main.rs                 # Standalone viewer binary
├── engine/
│   ├── mod.rs              # ProteinRenderEngine (core orchestrator)
│   ├── animation.rs        # Pose animation methods
│   ├── input.rs            # Input handling and selection
│   ├── options.rs          # Runtime options application
│   ├── queries.rs          # State queries
│   ├── scene_management.rs # Group management, bands, pulls
│   └── scene_sync.rs       # Background processor communication
├── scene/
│   ├── mod.rs              # Scene, EntityGroup, Focus, AggregatedRenderData
│   └── processor.rs        # SceneProcessor background thread
├── animation/
│   ├── mod.rs              # Module re-exports
│   ├── preferences.rs      # AnimationAction, AnimationPreferences
│   ├── interpolation.rs    # InterpolationContext, lerp utilities
│   ├── sidechain_state.rs  # Sidechain animation tracking
│   ├── behaviors/
│   │   ├── traits.rs       # AnimationBehavior trait
│   │   ├── snap.rs         # Instant snap
│   │   ├── smooth.rs       # Eased interpolation
│   │   ├── cascade.rs      # Per-residue staggered animation
│   │   ├── collapse_expand.rs  # Two-phase mutation animation
│   │   ├── backbone_then_expand.rs  # Backbone-first animation
│   │   └── state.rs        # ResidueVisualState
│   └── animator/
│       ├── mod.rs          # StructureAnimator (top-level API)
│       ├── controller.rs   # AnimationController (preemption)
│       ├── runner.rs       # AnimationRunner (single animation)
│       └── state.rs        # StructureState (current/target)
├── camera/
│   ├── mod.rs              # Module declarations
│   ├── core.rs             # Camera struct
│   ├── controller.rs       # CameraController (input + animation)
│   ├── frustum.rs          # Frustum culling
│   └── input_state.rs      # Click detection state machine
├── picking/
│   ├── mod.rs              # Module re-exports
│   ├── picking.rs          # Picking, SelectionBuffer
│   └── picking_state.rs    # Picking bind group management
├── gpu/
│   ├── render_context.rs   # wgpu device, queue, surface
│   ├── shader_composer.rs  # naga_oil shader composition
│   ├── dynamic_buffer.rs   # Growable GPU buffers
│   └── residue_color.rs    # Per-residue color GPU buffer
├── renderer/
│   ├── mod.rs              # MolecularRenderer trait
│   ├── pipeline_util.rs    # Shared pipeline configuration
│   ├── molecular/
│   │   ├── tube.rs         # Backbone tubes
│   │   ├── ribbon.rs       # Helices and sheets
│   │   ├── capsule_sidechain.rs  # Sidechain capsules
│   │   ├── ball_and_stick.rs     # Ligands, ions, waters
│   │   ├── band.rs         # Constraint bands
│   │   ├── pull.rs         # Active pull visualization
│   │   ├── nucleic_acid.rs # DNA/RNA backbone
│   │   ├── draw_context.rs # DrawBindGroups
│   │   └── capsule_instance.rs  # Shared CapsuleInstance struct
│   └── postprocess/
│       ├── post_process.rs # PostProcessStack
│       ├── ssao.rs         # Screen-space ambient occlusion
│       ├── bloom.rs        # Bloom extraction and blur
│       ├── composite.rs    # Final compositing
│       └── fxaa.rs         # Anti-aliasing
└── util/
    ├── options.rs          # Options, DisplayOptions, ColorOptions, etc.
    ├── lighting.rs         # Lighting uniform management
    ├── frame_timing.rs     # Frame rate tracking
    ├── score_color.rs      # Energy score → RGB conversion
    ├── sheet_adjust.rs     # Sheet surface offset adjustment
    ├── easing.rs           # Easing functions
    ├── bond_topology.rs    # Bond inference utilities
    └── trajectory.rs       # DCD trajectory playback
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
