# Viso — Improvement Plan

Findings from adversarial code quality analysis, ordered by effort.

## Completed work

- Engine decomposition: 18→10 fields, GpuPipeline/AnimationState/
  ConstraintSpecs/Bootstrap extracted. All files <500 lines.
- F-07 (pick ID f32 aliasing), F-13 (split mutation path),
  F-08 (frustum magic constant) — fixed by prior refactoring.
- F-15 (CLI timeout/size cap), F-05 (CLI error handling),
  F-09 (present mode), F-12 (HiDPI scale) — quick fixes.
- H2 (FxHasher), H4 (entity ID index), F-10 (resolve_atom_ref O(1)),
  H3 (clone storm), M2 (dead helpers), M4 (iterator returns) — small
  refactors.
- F-11 (adapter info logging + software rasterizer warning),
  F-06 (DynamicBuffer hysteresis shrink), M1 (encase already correct),
  M3 (pub→pub(crate) visibility audit across camera, animation, engine,
  renderer, postprocess) — medium refactors.
- F-02 (mesh cache granular invalidation) — geometry vs instance-only
  invalidation in MeshCache. Color/display changes no longer regenerate
  backbone meshes.
- H1 (shared WGSL modules) — 4 shared modules (pbr, selection,
  highlight, ray) + orchestrating shade module extracted. 12 modules
  registered in ShaderComposer. All consuming shaders use #import.

---

## Phase 1 — Test coverage (F-14, F-01)

**Problem:** Zero tests anywhere in the codebase.

**Approach:** The sub-structs (EntityStore, SceneTopology, VisualState,
AnimationState, ConstraintSpecs) are all independently constructable
without a GPU — no wrapper struct or mock needed. Tests construct the
pieces they need directly. The thin engine dispatch layer (3-8 line
methods) gets coverage implicitly; if a method's logic is complex
enough to need its own test, it belongs on the sub-struct.

**Test targets**, roughly ordered from easiest to hardest:

| Module | What to test | Constructs |
|--------|-------------|------------|
| `util/easing.rs` | Curve evaluation, boundary values | Pure functions |
| `animation/runner.rs` | Phase sequencing, lerp range, completion | `AnimationRunner` |
| `camera/frustum.rs` | Plane extraction, sphere containment | `Frustum` |
| `renderer/picking/pick_map.rs` | ID→PickTarget resolution | `PickMap` |
| `renderer/geometry/backbone/spline.rs` | Interpolation correctness | Pure functions |
| `engine/entity_store.rs` | Insert/remove/lookup, id_index consistency, generation bumps | `EntityStore` |
| `engine/scene.rs` | Topology rebuild, backbone_chain_offsets, sidechain atom_index | `SceneTopology` |
| `engine/constraint.rs` | resolve_atom_ref with known chain/sidechain data | Free functions + `SceneTopology` |
| `animation/state.rs` | Animation setup, tick, transition pending | `AnimationState` |

---

## Phase 2 — Generalize molex (F-03)

### 2.1. Rename and generalize the crate

**Problem:** `molex` (formerly `foldit_conv`) was named after and
structured around the FoldIt ecosystem. Its types (`MoleculeEntity`,
`SSType`, `Atom`, residue/chain structures) are general-purpose
molecular data, but the original crate name and packaging tied it to
FoldIt.

**Goal:** Publish `molex` as a standalone, general-purpose Rust
molecular data crate. This turns viso's tight coupling from a
liability into a strength — viso becomes the reference consumer of an
open community standard.

**Approach:** Same treatment as foldit-render → viso. Should be easier
since the data crate is smaller and less architecturally complex than
the renderer was.

**Scope:** Lives in the `molex` repo, not in viso. Work required:

1. ~~**Bootstrap project scaffolding** — copy and adapt from viso~~
   ✓ Done: `rustfmt.toml`, `clippy.toml`, `deny.toml`, `[lints]` in
   `Cargo.toml`
2. ~~Rename crate: `foldit_conv` → `molex`~~ ✓ Done
3. Audit public API for FoldIt-specific assumptions and generalize
4. Strip FoldIt-specific game/puzzle concepts from the core types
5. Clean up and document the public API for community consumption
6. Make all checks pass (`just check-all`)
7. Publish to crates.io
8. ~~Update viso's `Cargo.toml` to depend on `molex`~~ ✓ Done
9. ~~Update all `use foldit_conv::` imports across foldit-rs~~ ✓ Done

**Impact:** Establishes the Rust molecular data standard before anyone
else does. Viso + molex becomes a compelling open-source stack.

**Estimated effort:** Multi-day. Scaffolding + rename is fast (day 1).
API audit and generalization is the real work.

**Strategic note:** This is the finding that matters most long-term.
Whoever publishes the first good general-purpose Rust molecule crate
wins the ecosystem. Better to do it yourself than wait for someone
else to fork and generalize it.

---

## Future work (not from findings)

- Input triple buffer (lock-free per-entity input channel)
- Animation feature-gating
- Cascade per-residue staggering
- Molecular surfaces (MSMS / SES)
- Selection language
