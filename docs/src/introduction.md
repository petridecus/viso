# Introduction

Viso is a GPU-accelerated 3D protein visualization engine built in Rust on top of [wgpu](https://wgpu.rs/). It powers the molecular graphics in [foldit-rs](https://github.com/foldit/foldit-rs), rendering proteins, ligands, nucleic acids, and constraint visualizations at interactive frame rates.

## What Viso Does

- **Multiple representations** -- ribbons, tubes, ball-and-stick, capsule sidechains, nucleic acid backbones
- **Post-processing pipeline** -- SSAO, bloom, FXAA, depth-based outlines, fog, tone mapping
- **Interactive camera** -- arcball rotation, panning, zoom, animated transitions, auto-rotate
- **GPU picking** -- click to select residues, double-click for secondary structure segments, triple-click for chains, shift-click for multi-select
- **Animation system** -- smooth interpolation, cascading reveals, collapse/expand mutations, per-entity targeted animation
- **Background scene processing** -- CPU-heavy mesh generation on a dedicated thread with triple-buffered results
- **TOML presets** -- display, lighting, color, geometry, and camera options with load/save support

## Relationship to Rustdoc

This book covers **concepts, architecture, and integration patterns**. For the full API reference (every struct, method, and field), see the [rustdoc documentation](../target/doc/viso/index.html) generated with `cargo doc`.

The two are complementary:

| This Book | Rustdoc |
|-----------|---------|
| How the render loop works | Every method on `ProteinRenderEngine` |
| What animation behaviors exist and when to use them | Exact signatures, trait bounds, field types |
| How the background processor threads data | All public and `pub(crate)` items |
| Integration patterns from foldit-rs | Exhaustive API surface |

## Where to Start

**If you want to embed viso in your own application:**

1. [Quick Start](./getting-started/quick-start.md) -- walk through the standalone `main.rs`
2. [Engine Lifecycle](./integration/engine-lifecycle.md) -- creation, initialization, shutdown
3. [The Render Loop](./integration/render-loop.md) -- per-frame sequence
4. [Handling Input](./integration/handling-input.md) -- mouse and keyboard wiring

**If you want to understand how foldit-rs uses viso:**

1. [Scene Management](./integration/scene-management.md) -- groups, entities, focus
2. [Dynamic Structure Updates](./integration/dynamic-updates.md) -- Rosetta and ML integration
3. [Options and Presets](./configuration/options-and-presets.md) -- TOML configuration

**If you want to understand viso internals:**

1. [Architecture Overview](./architecture/overview.md) -- system diagram and data flow
2. [Rendering Pipeline](./deep-dives/rendering-pipeline.md) -- geometry pass and post-processing
3. [Background Scene Processing](./deep-dives/background-processing.md) -- threading model
4. [Animation System](./deep-dives/animation-system.md) -- three-layer architecture
