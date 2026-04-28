# Scene Management

Viso's structural state is owned by `molex::Assembly`. The host
application (or `VisoApp` in standalone deployments) owns the
authoritative `Assembly` and publishes snapshots through a triple
buffer; the engine reads them through an `AssemblyConsumer` and
rederives its render state.

There is no "group" abstraction in viso — every entity lives directly
in the `Assembly` and is identified by an opaque `EntityId`.

## The Assembly Channel

```
┌──────────────────────┐                  ┌─────────────────┐
│   VisoApp / host     │  Arc<Assembly>   │   VisoEngine    │
│                      │ ──triple-buffer→ │                 │
│  Assembly            │                  │  AssemblyConsumer
│  AssemblyPublisher   │                  │  Scene + derived│
└──────────────────────┘                  └─────────────────┘
```

- `VisoApp` mutates the `Assembly` and calls `publisher.commit(...)`.
- `VisoEngine::update(dt)` polls the consumer; if a new generation is
  ready, the engine rederives its per-entity state and submits a
  full-rebuild request to the background mesh processor.

In standalone deployments (`feature = "viewer" / "gui" / "web"`),
`VisoApp` plays the host role. In `foldit-rs`, the real host owns the
publisher and feeds the consumer into `VisoEngine::new` directly.

## Mutating the Scene (via `VisoApp`)

All structural mutation methods take `&mut VisoEngine` so viso-side
bookkeeping (animation transitions, camera fit, per-entity
annotations) can update atomically alongside the `Assembly` mutation
and the `publisher.commit`.

```rust
// Add entities. Returns the assigned raw u32 ids.
let ids: Vec<u32> = app.load_entities(&mut engine, entities, fit_camera);

// Replace the entire scene.
app.replace_scene(&mut engine, new_entities);

// Remove all entities.
app.clear_scene(&mut engine);

// Update one or many entities (matched by id), with a default
// transition for entities lacking a per-entity behavior override.
app.update_entity(&mut engine, entity, Transition::smooth())?;
app.update_entities(&mut engine, updated, &Transition::smooth());

// Update just the atom coordinates of an existing entity.
app.update_entity_coords(&mut engine, id, &coords, Transition::smooth());

// Reconcile: add new ids, remove missing ids, update existing ones.
app.sync_entities(&mut engine, entities, &Transition::smooth());

// Remove a single entity by id.
app.remove_entity(&mut engine, id);

// Per-entity visibility (also syncs ambient-type display flags for
// water/ion/solvent so the renderer safety net stays consistent).
app.set_entity_visible(&mut engine, id, true);

// Per-entity scoring (drives color-by-score; pass None to clear).
app.set_per_residue_scores(&mut engine, id, Some(scores));

// Per-entity SS override (used for puzzle annotations).
app.set_ss_override(&mut engine, id, ss_types);
```

Internally each of these republishes the `Assembly` and calls
`engine.sync_now()`, which submits a full-rebuild request to the
background mesh processor. None of the cost is paid synchronously on
the main thread — mesh generation happens off-thread.

## Engine-Side Annotations

Some per-entity state is purely a viso concern (it doesn't belong on
the molecular structure itself). Those live on
`EntityAnnotations`, mutated through engine methods:

```rust
// Animation behavior overrides (keyed by EntityId, not raw u32).
let eid = engine.entity_id(raw_id).expect("known entity");
engine.set_entity_behavior(eid, Transition::collapse_expand(/* ... */));
engine.clear_entity_behavior(eid);

// Per-entity appearance overrides (drawing mode, color scheme,
// helix/sheet style, surface kind, palette, etc.).
let mut overrides = DisplayOverrides::default();
overrides.color_scheme = Some(ColorScheme::SecondaryStructure);
engine.set_entity_appearance(eid, overrides);
engine.clear_entity_appearance(eid);
```

`set_entity_appearance` diffs against the previous overrides and
dispatches only the invalidations that matter — a `surface_kind`
change triggers surface regeneration, a `color_scheme` change triggers
color recomputation, and so on.

## Looking Up Entities

The engine exposes a small read-only surface for looking entities up:

```rust
// Translate a raw u32 (from IPC, TOML, CLI) to an opaque EntityId.
let eid: Option<EntityId> = engine.entity_id(raw_id);

// Walk the current Assembly snapshot directly.
for entity in engine.assembly().entities() {
    println!("{:?}: {}", entity.id(), entity.molecule_type());
}

// Total entity count.
let n = engine.entity_count();
```

`entity_id` is the canonical "boundary translator" — wire formats
carry raw `u32` ids; viso-internal APIs use `EntityId`. Translate
once at the boundary and pass `EntityId` through.

## Focus

Focus determines what the camera follows and what the user is
"working on". It cycles through entities with `Tab`:

```rust
pub enum Focus {
    Session,             // All visible entities (default)
    Entity(EntityId),    // A specific entity
}
```

```rust
// Cycle: Session → Entity₁ → … → EntityN → Session
engine.execute(VisoCommand::CycleFocus);

// Focus a specific entity by raw id.
engine.execute(VisoCommand::FocusEntity { id });

// Reset to session-wide focus.
engine.execute(VisoCommand::ResetFocus);

// Read current focus state.
let focus: Focus = engine.focus();
let focused_entity: Option<EntityId> = engine.focused_entity();
```

## What Happens During Sync

When a new `Assembly` snapshot arrives:

1. **Rederive per-entity state.** For each entity, the engine builds
   render-ready derived data (backbone chains, sidechain topology, SS
   types, residue color metadata).
2. **Submit a `FullRebuild`.** The background processor receives a
   `Vec<FullRebuildEntity>` plus the active display, color, and
   geometry options. Per-entity mesh caching means only entities whose
   `mesh_version` changed are regenerated.
3. **Triple-buffer the result.** When the processor finishes, the
   resulting `PreparedRebuild` is written to a triple buffer.
4. **Apply on the next frame.** `engine.update(dt)` calls
   `apply_pending_scene`, which uploads the GPU buffers in a memcpy
   and rebuilds picking bind groups.

The main thread never blocks. If the new meshes aren't ready by the
next frame, the previous frame's data continues to render until they
are.
