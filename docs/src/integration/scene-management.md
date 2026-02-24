# Scene Management

The scene system organizes molecular data into groups, tracks visibility and focus, and provides the data structures that flow into the rendering pipeline.

## Core Types

### Scene

`Scene` is the authoritative container for all molecular data. It owns entity groups and tracks dirty state via a generation counter.

```rust
let mut scene = Scene::new();

// Add a group of entities
let group_id = scene.add_group(entities, "Loaded Structure");

// Check if scene changed since last render
if scene.is_dirty() {
    // Sync to renderers...
    scene.mark_rendered();
}
```

### EntityGroup

An `EntityGroup` is a collection of `MoleculeEntity` values loaded together -- typically from one file or one backend operation. Each group has:

- A unique `GroupId`
- A human-readable name
- Visibility state
- A `mesh_version` counter for cache invalidation
- Optional per-residue scores and secondary structure overrides

```rust
// Read access
let group = engine.group(group_id).unwrap();
println!("{}: {} entities", group.name(), group.entities().len());

// Write access (bumps mesh_version, invalidates caches)
let group = engine.group_mut(group_id).unwrap();
group.set_entities(new_entities);
group.invalidate_render_cache();
```

### MoleculeEntity

A `MoleculeEntity` represents a single molecular chain or component. It contains atomic coordinates (`Coords`), a molecule type, and an entity ID.

### GroupId

A monotonically increasing u64 identifier. Each call to `scene.add_group()` or `engine.load_entities()` produces a new unique ID.

## Group Management

### Loading Entities

```rust
// Load entities into a new group, optionally fitting the camera
let group_id = engine.load_entities(entities, "My Structure", true);
```

The `fit_camera` parameter controls whether the camera animates to show the new group.

### Visibility

```rust
engine.set_group_visible(group_id, false); // Hide
engine.set_group_visible(group_id, true);  // Show
```

Hidden groups are excluded from aggregated render data, picking, and camera fitting.

### Removal

```rust
let removed = engine.remove_group(group_id); // Returns Option<EntityGroup>
engine.clear_scene(); // Remove all groups
```

### Iteration

```rust
let ids = engine.group_ids(); // Ordered by insertion
let count = engine.group_count();

// Iterate over all groups (read-only)
for group in engine.scene.iter() {
    println!("{}: visible={}", group.name(), group.visible);
}
```

## Focus System

Focus determines which entities are active for operations. It cycles through groups and focusable entities with tab:

```rust
pub enum Focus {
    Session,          // All groups (default)
    Group(GroupId),   // A specific group
    Entity(u32),      // A specific entity by entity_id
}
```

### Cycling Focus

```rust
let new_focus = engine.cycle_focus();
// Session -> Group1 -> Group2 -> ... -> focusable entities -> Session
```

### Querying Focus

```rust
let focus = engine.focus(); // Returns &Focus
```

## Aggregated Render Data

When the scene is dirty, the engine collects data from all visible groups into `AggregatedRenderData`. This is computed lazily and cached:

```rust
let aggregated = scene.aggregated(); // Returns Arc<AggregatedRenderData>
```

The aggregated data contains:

- **Backbone chains** -- all visible backbone atom positions, concatenated across groups
- **Sidechain data** -- positions, bonds, hydrophobicity, residue indices (global)
- **Secondary structure types** -- per-residue SS classification
- **Non-protein entities** -- ligands, ions, waters, lipids
- **Nucleic acid chains** -- P-atom chains and nucleotide rings
- **All positions** -- for camera fitting

Global residue indices are remapped during aggregation so that the first group's residues start at 0, the second group's residues follow, etc.

## Per-Group Data for Background Processing

When syncing to renderers, the scene produces `PerGroupData` for each visible group:

```rust
let per_group = scene.per_group_data(); // Vec<PerGroupData>
```

Each `PerGroupData` contains:

- Group ID and mesh version (for cache invalidation)
- Backbone chains and residue data
- Sidechain atoms, bonds, and backbone-sidechain bonds
- Secondary structure overrides
- Per-residue scores
- Non-protein entities and nucleic acid data

The background processor uses `mesh_version` to decide whether to regenerate or reuse cached meshes for each group.

## Syncing Changes

After modifying the scene, sync to the rendering pipeline:

```rust
// Full sync with optional transition
engine.sync_scene_to_renderers(Some(Transition::smooth()));

// Targeted sync -- only specific entities get animation, others snap
engine.sync_scene_to_renderers_targeted(
    HashMap::from([(design_id, Transition::with_behavior(
        BackboneThenExpand::new(
            Duration::from_millis(400),
            Duration::from_millis(600),
        ))
        .allowing_size_change()
        .suppressing_initial_sidechains()
    )])
);
```

The sync submits a `SceneRequest::FullRebuild` to the background thread. On the next frame, `apply_pending_scene()` picks up the results.

## Backend Coordinate Updates

For Rosetta integration where multiple groups may be updated atomically:

```rust
// Combine coords from all visible groups
let result = scene.combined_coords_for_backend();
// result.bytes: ASSEM01 format for Rosetta
// result.chain_ids_per_group: chain ID mapping for splitting results

// Apply combined update from Rosetta
scene.apply_combined_update(&coords_bytes, &chain_ids_per_group)?;

// Update a single group's protein coordinates
scene.update_group_protein_coords(group_id, new_coords);
```
