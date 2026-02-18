# Background Scene Processing

Mesh generation for molecular structures is CPU-intensive -- generating tube splines, ribbon surfaces, and sidechain capsule instances can take 20-40ms for complex structures. Viso offloads this to a background thread so the main thread can continue rendering at full frame rate.

## Architecture

```
Main Thread                          Background Thread
  ├─ SceneProcessor::submit()          ├─ Blocks on mpsc::Receiver
  │   → mpsc::Sender                   ├─ Processes request
  │                                    ├─ Generates/caches per-group meshes
  │                                    ├─ Concatenates into PreparedScene
  ├─ try_recv_scene()                  └─ Writes to triple_buffer::Output
  │   ← triple_buffer::Input
  ├─ GPU upload (<1ms)
  └─ Render
```

### Communication Channels

| Channel | Type | Direction | Purpose |
|---------|------|-----------|---------|
| Request | `mpsc::Sender<SceneRequest>` | Main → Background | Submit work |
| Scene result | `triple_buffer` | Background → Main | Completed `PreparedScene` |
| Animation result | `triple_buffer` | Background → Main | Completed `PreparedAnimationFrame` |

Triple buffers are lock-free: the writer always has a buffer to write to, and the reader always gets the latest completed result. No blocking on either side.

## SceneProcessor

```rust
let processor = SceneProcessor::new(); // Spawns background thread

// Submit work (non-blocking)
processor.submit(SceneRequest::FullRebuild { /* ... */ });

// Check for results (non-blocking)
if let Some(prepared) = processor.try_recv_scene() {
    // Upload to GPU
}

// Shutdown
processor.shutdown(); // Sends Shutdown message, joins thread
```

## Request Types

### FullRebuild

A complete scene rebuild with all visible group data:

```rust
SceneRequest::FullRebuild {
    groups: Vec<PerGroupData>,
    aggregated: Arc<AggregatedRenderData>,
    entity_actions: HashMap<GroupId, AnimationAction>,
    display: DisplayOptions,
    colors: ColorOptions,
}
```

This is submitted when:
- The scene is dirty (entities added/removed/modified)
- Display or color options change
- A targeted animation is requested

### AnimationFrame

Per-frame mesh regeneration during animation:

```rust
SceneRequest::AnimationFrame {
    backbone_chains: Vec<Vec<Vec3>>,
    sidechains: Option<AnimationSidechainData>,
    ss_types: Option<Vec<SSType>>,
    per_residue_colors: Option<Vec<[f32; 3]>>,
}
```

This is submitted every frame while animation is in progress. It regenerates tube and ribbon meshes from interpolated backbone positions.

### Shutdown

Terminates the background thread:

```rust
SceneRequest::Shutdown
```

## Per-Group Mesh Caching

The background thread maintains a cache of per-group meshes:

```
HashMap<GroupId, (u64, CachedGroupMesh)>
         │         │        │
         │         │        └─ Tube, ribbon, sidechain, BNS meshes
         │         └─ mesh_version at time of generation
         └─ Group identifier
```

### Cache Invalidation

When a `FullRebuild` arrives, the processor checks each group:

1. **Same version** -- reuse cached mesh (skip generation entirely)
2. **Different version** -- regenerate and update cache
3. **Group removed** -- evict from cache

Version-based invalidation is cheap (a u64 comparison) and avoids regenerating unchanged groups. For a scene with 3 groups where only 1 changed, this saves ~70% of mesh generation time.

### Global Settings Changes

Display and color options affect all meshes. The processor detects when these change and clears the entire cache, forcing full regeneration. This happens when:
- Backbone color mode changes
- Show/hide sidechains changes
- Tube radius or segment count changes

## Mesh Generation

For each group, the processor generates:

1. **Tube mesh** -- cubic Hermite splines with rotation-minimizing frames, filtered by SS type
2. **Ribbon mesh** -- B-spline interpolation for helices and sheets, with sheet offsets
3. **Sidechain capsule instances** -- packed `CapsuleInstance` structs for the storage buffer
4. **Ball-and-stick instances** -- sphere and capsule instances for non-protein entities
5. **Nucleic acid mesh** -- flat ribbon geometry for DNA/RNA backbones

### Mesh Concatenation

After generating (or retrieving from cache) all group meshes, they're concatenated into a single `PreparedScene`:

- Vertex buffers are appended
- Index buffers are appended with **index offset** adjustment (each group's indices are offset by the previous group's vertex count)
- Instance buffers are concatenated
- Passthrough data (backbone chains, sidechain positions, etc.) is merged with global index remapping

## PreparedScene

The output of a `FullRebuild`, ready for GPU upload:

```rust
pub struct PreparedScene {
    // Tube mesh
    pub tube_vertices: Vec<u8>,
    pub tube_indices: Vec<u8>,
    pub tube_index_count: u32,

    // Ribbon mesh
    pub ribbon_vertices: Vec<u8>,
    pub ribbon_indices: Vec<u8>,
    pub ribbon_index_count: u32,
    pub sheet_offsets: Vec<(u32, Vec3)>,

    // Sidechain capsule instances
    pub sidechain_instances: Vec<u8>,
    pub sidechain_instance_count: u32,

    // Ball-and-stick
    pub bns_sphere_instances: Vec<u8>,
    pub bns_sphere_count: u32,
    pub bns_capsule_instances: Vec<u8>,
    pub bns_capsule_count: u32,
    pub bns_picking_capsules: Vec<u8>,
    pub bns_picking_count: u32,

    // Nucleic acid mesh
    pub na_vertices: Vec<u8>,
    pub na_indices: Vec<u8>,
    pub na_index_count: u32,

    // Passthrough data for animation, camera, etc.
    pub backbone_chains: Vec<Vec<Vec3>>,
    pub sidechain_positions: Vec<Vec3>,
    pub ss_types: Option<Vec<SSType>>,
    pub per_residue_colors: Option<Vec<[f32; 3]>>,
    pub all_positions: Vec<Vec3>,
    pub entity_actions: HashMap<GroupId, AnimationAction>,
    pub entity_residue_ranges: Vec<(GroupId, u32, u32)>,
    // ... more passthrough fields
}
```

All byte arrays are raw GPU buffer data (`bytemuck::cast_slice`), ready for `queue.write_buffer()` with no further processing.

## PreparedAnimationFrame

The output of an `AnimationFrame` request, containing only the meshes that change during animation:

```rust
pub struct PreparedAnimationFrame {
    pub tube_vertices: Vec<u8>,
    pub tube_indices: Vec<u8>,
    pub tube_index_count: u32,
    pub ribbon_vertices: Vec<u8>,
    pub ribbon_indices: Vec<u8>,
    pub ribbon_index_count: u32,
    pub sheet_offsets: Vec<(u32, Vec3)>,
    pub sidechain_instances: Option<Vec<u8>>,
    pub sidechain_instance_count: u32,
}
```

Only tube, ribbon, and (optionally) sidechain meshes are regenerated during animation -- ball-and-stick, bands, pulls, and nucleic acid meshes don't change.

## Request Coalescing

If the background thread is busy and multiple requests queue up, it drains the channel and keeps only the latest request of each type. This prevents a backlog during rapid updates:

```
Queue: [FullRebuild, FullRebuild, AnimFrame, AnimFrame, AnimFrame]
                                          ↓ drain
Process: [FullRebuild (latest), AnimFrame (latest)]
```

This means the background thread always works on the most current data, never wasting time on stale intermediate states.

## Threading Model Summary

| Thread | Owns | Does |
|--------|------|------|
| **Main thread** | GPU resources, engine, scene | Input, render, GPU upload |
| **Background thread** | Mesh cache | CPU mesh generation |
| **Bridge** | Triple buffers, mpsc channel | Lock-free data transfer |

The main thread never blocks on the background thread. If meshes aren't ready yet, the previous frame's meshes are rendered. This ensures consistent frame rates even during expensive mesh regeneration.
