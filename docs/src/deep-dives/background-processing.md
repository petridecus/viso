# Background Scene Processing

Mesh generation for molecular structures is CPU-intensive — generating
backbone splines, ribbon surfaces, and sidechain capsule instances
can take 20–40ms for complex structures. Viso offloads this to a
background thread so the main thread can continue rendering at full
frame rate.

## Architecture

```
Main Thread                          Background Thread
  ├─ SceneProcessor::submit()          ├─ Blocks on mpsc::Receiver
  │   → mpsc::Sender                   ├─ Processes request
  │                                    ├─ Generates / caches per-entity meshes
  │                                    ├─ Concatenates into PreparedRebuild
  ├─ try_recv_rebuild()                └─ Writes to triple_buffer::Output
  │   ← triple_buffer::Input
  ├─ GPU upload (<1ms)
  └─ Render
```

### Communication Channels

| Channel | Type | Direction | Purpose |
|---------|------|-----------|---------|
| Request | `mpsc::Sender<SceneRequest>` | Main → Background | Submit work |
| Rebuild result | `triple_buffer` | Background → Main | Completed `PreparedRebuild` |
| Animation result | `triple_buffer` | Background → Main | Completed `PreparedAnimationFrame` |

Triple buffers are lock-free: the writer always has a buffer to write
to, and the reader always gets the latest completed result. No
blocking on either side.

## SceneProcessor

```rust
let processor = SceneProcessor::new()?; // Spawns background thread

// Submit work (non-blocking)
processor.submit(SceneRequest::FullRebuild(Box::new(body)));

// Check for results (non-blocking)
if let Some(prepared) = processor.try_recv_rebuild() {
    // Upload to GPU
}

// Shutdown
processor.shutdown(); // Sends Shutdown message, joins thread
```

## Request Types

```rust
pub(crate) enum SceneRequest {
    FullRebuild(Box<FullRebuildBody>),
    AnimationFrame(Box<AnimationFrameBody>),
    Shutdown,
}
```

### FullRebuild

A complete scene rebuild with per-entity render-ready snapshots:

```rust
pub(crate) struct FullRebuildBody {
    pub entities: Vec<FullRebuildEntity>,           // per-entity snapshots
    pub display: DisplayOptions,
    pub colors: ColorOptions,
    pub geometry: GeometryOptions,
    pub entity_options:
        FxHashMap<u32, (DisplayOptions, GeometryOptions)>, // per-entity overrides
    pub generation: u64,
}

pub(crate) struct FullRebuildEntity {
    pub id: EntityId,
    pub mesh_version: u64,
    pub drawing_mode: DrawingMode,
    pub topology: Arc<EntityTopology>,
    pub positions: Vec<Vec3>,
    pub ss_override: Option<Vec<SSType>>,
    pub per_residue_colors: Option<Vec<[f32; 3]>>,
    pub sheet_plane_normals: Vec<(u32, Vec3)>,
}
```

`FullRebuild` is submitted when:

- A new `Assembly` snapshot is consumed (entities added / removed /
  modified, scores or SS overrides changed).
- Display, color, or geometry options change.
- A scoped reset (e.g. `engine.reset_scene_local_state`) clears local
  state.

`mesh_version` per-entity is the cache key — entities whose version
hasn't changed since the previous rebuild reuse their cached mesh.

### AnimationFrame

Per-frame mesh regeneration during animation:

```rust
pub(crate) struct AnimationFrameBody {
    pub positions: EntityPositions,           // interpolated
    pub geometry: GeometryOptions,
    pub per_chain_lod: Option<Vec<(usize, usize)>>,  // per-chain detail override
    pub include_sidechains: bool,
    pub generation: u64,
}
```

This is submitted while animation is in progress. It regenerates
backbone meshes (and optionally sidechains) from interpolated
positions, reusing topology and other state cached from the last
`FullRebuild`.

### Shutdown

Terminates the background thread.

## Per-Entity Mesh Caching

The background thread maintains a cache of per-entity meshes, keyed on
`EntityId`:

```
FxHashMap<EntityId, CachedEntityMesh>
```

`CachedEntityMesh` stores GPU-ready byte buffers (backbone vertex /
index, sidechain instances, ball-and-stick spheres+capsules, nucleic
acid stems+rings) plus typed intermediates needed for index
concatenation.

### Cache Invalidation

When a `FullRebuild` arrives, the processor checks each entity's
`mesh_version` against the cached version:

1. **Same version** — reuse cached mesh (skip generation entirely).
2. **Different version** — regenerate and update cache.
3. **Entity removed** — evict from cache.

Version-based invalidation is cheap (a u64 comparison) and avoids
regenerating unchanged entities. For a scene with 3 entities where
only 1 changed, this saves ~70% of mesh generation time.

### Global vs Per-Entity Settings

Display, color, and geometry options affect mesh content. The
processor distinguishes geometry-affecting changes (which require
mesh regeneration) from color-only changes (which only update vertex
color buffers). A bumped `mesh_version` is the universal "regenerate
me" signal — option-change paths in the engine bump the affected
entities' versions before submitting the rebuild.

## Mesh Generation

For each entity, the processor generates whichever of the following
apply to its `drawing_mode`:

1. **Backbone mesh** — cubic Hermite splines with rotation-minimizing
   frames, with separate index ranges for tube and ribbon passes.
2. **Sidechain capsule instances** — packed capsule structs for the
   storage buffer.
3. **Ball-and-stick instances** — sphere and capsule instances for
   non-protein entities (and proteins drawn in BallAndStick mode).
4. **Nucleic acid instances** — stem capsules and ring polygons for
   DNA/RNA backbones.

### Mesh Concatenation

After generating (or retrieving from cache) all entity meshes, they're
concatenated into a single `PreparedRebuild`:

- Vertex buffers are appended.
- Index buffers are appended with per-entity index offset adjustment.
- Instance buffers are concatenated.
- A single `PickMap` is built mapping raw GPU pick IDs to typed pick
  targets.

## PreparedRebuild

The output of a `FullRebuild`, ready for GPU upload:

```rust
pub(crate) struct PreparedRebuild {
    pub generation: u64,
    pub backbone: BackboneMeshData,                 // verts + tube/ribbon idx
    pub sidechain_instances: Vec<u8>,
    pub sidechain_instance_count: u32,
    pub bns: BallAndStickInstances,                 // sphere + capsule instances
    pub na: NucleicAcidInstances,                   // stem + ring instances
    pub pick_map: PickMap,
}
```

All byte arrays are raw GPU buffer data (`bytemuck::cast_slice`),
ready for `queue.write_buffer()` with no further processing.

## PreparedAnimationFrame

The output of an `AnimationFrame` request, containing only the data
that changes during animation:

```rust
pub(crate) struct PreparedAnimationFrame {
    pub backbone: BackboneMeshData,
    pub sidechain_instances: Option<Vec<u8>>,
    pub sidechain_instance_count: u32,
    pub generation: u64,
}
```

Only backbone mesh and (optionally) sidechain instances are
regenerated during animation — ball-and-stick, nucleic-acid, and
isosurface meshes don't change.

## Stale Frame Discarding

When a scene is replaced (e.g. loading a new structure), in-flight
animation frames from the old scene become stale — their per-chain
LOD or topology assumptions may not match the new scene.

A monotonically increasing `generation` counter prevents this:

1. Each `FullRebuild` carries a new generation.
2. Each `AnimationFrame` carries the generation of the scene it was
   produced for.
3. **Background thread**: frames with `generation < last_rebuild_generation`
   are skipped before processing.
4. **Main thread**: stale animation frames are discarded before GPU
   upload.

This two-level check ensures stale frames are dropped both before
expensive mesh generation and before GPU upload, with no additional
synchronization primitives.

## Threading Model Summary

| Thread | Owns | Does |
|--------|------|------|
| **Main thread** | GPU resources, engine, scene | Input, render, GPU upload |
| **Mesh thread** | Per-entity mesh cache | CPU mesh generation |
| **Surface thread** | (none — short-lived) | Isosurface mesh regeneration |
| **Bridge** | Triple buffers, mpsc channels | Lock-free data transfer |

The main thread never blocks on the background threads. If meshes
aren't ready yet, the previous frame's meshes are rendered. This
ensures consistent frame rates even during expensive mesh
regeneration.
