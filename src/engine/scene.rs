//! Assembly consumption + derived per-entity state.
//!
//! [`Scene`] bundles the one coherent data flow that makes the engine
//! a consumer of the host's [`Assembly`](molex::Assembly): the
//! triple-buffer read end, the latest snapshot, the generation
//! tracker, and the render-ready derived state
//! ([`SceneRenderState`] + per-entity [`EntityView`] +
//! [`EntityPositions`]).
//!
//! `sync_from_assembly` reads the three ingest fields and writes the
//! four derived fields, so they're kept together. The `next_mesh_version`
//! dispenser also lives here — it's the counter that stamps freshness
//! onto `EntityView.mesh_version`, and must survive `reset_scene_local_state`
//! so a Vacant insert never collides with a stale worker `MeshCache`
//! entry for the same `EntityId`.
//!
//! Every overlay / command entry point that arrives with a raw `u32`
//! entity ID translates once via [`Scene::entity_id`] and then passes
//! the opaque [`EntityId`] down. Internal code never walks the
//! assembly looking up u32s.

use std::sync::Arc;

use molex::entity::molecule::id::EntityId;
use molex::{Assembly, MoleculeEntity};
use rustc_hash::FxHashMap;

use super::assembly_consumer::AssemblyConsumer;
use super::entity_view::EntityView;
use super::positions::EntityPositions;
use super::scene_state::SceneRenderState;

/// Assembly consumption + derived per-entity state.
pub(crate) struct Scene {
    // ── Ingest ────────────────────────────────────────────────────
    /// Triple-buffer reader for the host-owned [`Assembly`].
    pub consumer: AssemblyConsumer,
    /// Last `Assembly` snapshot applied by `sync_from_assembly`. Held
    /// for read-only queries (entity metadata for the UI panel, atom
    /// positions for the picking pipeline). Always up to date with
    /// `entity_state` / `positions`.
    pub current: Arc<Assembly>,
    /// Generation of the most recently consumed snapshot. Initialized
    /// to `u64::MAX` so the first snapshot (generation 0) always
    /// triggers a sync.
    pub last_seen_generation: u64,

    // ── Derived ───────────────────────────────────────────────────
    /// Cross-entity rendering data (disulfide + H-bond endpoints)
    /// rederived on every sync.
    pub render_state: Arc<SceneRenderState>,
    /// Per-entity rendering state (topology + drawing mode + SS
    /// override + mesh version) rederived on every sync.
    pub entity_state: FxHashMap<EntityId, EntityView>,
    /// Per-entity animator write surface; renderer reads back each
    /// frame. Seeded from the assembly's reference positions when
    /// new entities appear, animated locally thereafter.
    pub positions: EntityPositions,

    /// Monotonic mesh-version dispenser. Survives
    /// `reset_scene_local_state` so a Vacant insert never collides
    /// with a stale worker `MeshCache` entry for the same `EntityId`
    /// (fresh per-file allocators reuse low ids across
    /// `replace_scene`).
    pub next_mesh_version: u64,
}

impl Scene {
    /// Empty scene with a valid (but empty) assembly snapshot. The
    /// caller provides the triple-buffer reader.
    pub fn new(consumer: AssemblyConsumer) -> Self {
        Self {
            consumer,
            current: Arc::new(Assembly::new(Vec::new())),
            last_seen_generation: u64::MAX,
            render_state: Arc::new(SceneRenderState::new()),
            entity_state: FxHashMap::default(),
            positions: EntityPositions::new(),
            next_mesh_version: 1,
        }
    }

    /// Pull a fresh, never-reused `mesh_version` from the dispenser.
    /// All writes to `EntityView.mesh_version` must route through
    /// this so a worker `MeshCache` entry for an `EntityId` is never
    /// invalidated by a colliding version.
    pub fn bump_mesh_version(&mut self) -> u64 {
        let v = self.next_mesh_version;
        self.next_mesh_version = self.next_mesh_version.wrapping_add(1);
        v
    }

    /// Look up the opaque [`EntityId`] for a raw `u32` id. Returns
    /// `None` if no entity with that raw id exists. Callers that hold
    /// a raw id should translate here *once* and then pass `EntityId`
    /// down.
    #[must_use]
    pub fn entity_id(&self, raw: u32) -> Option<EntityId> {
        self.current
            .entities()
            .iter()
            .map(MoleculeEntity::id)
            .find(|eid| eid.raw() == raw)
    }

    /// Reset all scene-local derived state (positions, entity_state).
    /// Keeps the consumer and `next_mesh_version` (the dispenser must
    /// outlive `replace_scene` so fresh per-file entity IDs don't
    /// collide with stale worker mesh cache entries). Also resets
    /// `last_seen_generation` to `u64::MAX` so the next snapshot
    /// triggers a sync unconditionally.
    pub fn reset_local_state(&mut self) {
        self.entity_state.clear();
        self.positions = EntityPositions::new();
        self.render_state = Arc::new(SceneRenderState::new());
        self.current = Arc::new(Assembly::new(Vec::new()));
        self.last_seen_generation = u64::MAX;
    }
}
