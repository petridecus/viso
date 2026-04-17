//! Background scene processor for non-blocking geometry generation.
//!
//! Moves all CPU-heavy mesh/instance generation off the main thread.
//! The main thread only does GPU uploads (<1ms) and render passes.
//!
//! Supports **per-entity mesh caching**: when an entity's `mesh_version`
//! hasn't changed between frames, its cached mesh is reused instead of
//! being regenerated. Global settings changes (view mode, display,
//! colors) clear the entire cache.

use std::sync::mpsc;

use rustc_hash::{FxHashMap, FxHashSet};

use super::prepared::{
    CachedEntityMesh, PreparedAnimationFrame, PreparedScene, SceneRequest,
};
use crate::options::{ColorOptions, DisplayOptions, GeometryOptions};

// ---------------------------------------------------------------------------
// Platform-abstracted background thread spawn
// ---------------------------------------------------------------------------

/// Handle to a background worker. On native this is a joinable OS thread;
/// on WASM it is a no-op because the worker runs on a rayon pool thread
/// (backed by web workers via `wasm-bindgen-rayon` + `SharedArrayBuffer`)
/// and exits when the channel disconnects.
#[cfg(not(target_arch = "wasm32"))]
type WorkerHandle = Option<std::thread::JoinHandle<()>>;
#[cfg(target_arch = "wasm32")]
type WorkerHandle = ();

/// Spawn a long-lived closure on a background thread.
///
/// - **Native:** dedicated OS thread via `std::thread::Builder`.
/// - **WASM:** `rayon::spawn` onto the `wasm-bindgen-rayon` pool.
fn spawn_background(
    f: impl FnOnce() + Send + 'static,
) -> Result<WorkerHandle, std::io::Error> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::Builder::new()
            .name("scene-processor".into())
            .spawn(f)
            .map(Some)
    }
    #[cfg(target_arch = "wasm32")]
    {
        rayon::spawn(f);
        Ok(())
    }
}

/// Join a background worker, blocking until it finishes.
fn join_background(handle: &mut WorkerHandle) {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(h) = handle.take() {
        let _ = h.join();
    }
    #[cfg(target_arch = "wasm32")]
    let _ = handle;
}

// ---------------------------------------------------------------------------
// SceneProcessor
// ---------------------------------------------------------------------------

/// Background thread that generates CPU-side geometry from scene data.
pub struct SceneProcessor {
    request_tx: mpsc::Sender<SceneRequest>,
    scene_result: triple_buffer::Output<Option<PreparedScene>>,
    anim_result: triple_buffer::Output<Option<PreparedAnimationFrame>>,
    worker: WorkerHandle,
    /// Monotonically increasing scene generation counter. Bumped each
    /// time a `FullRebuild` is submitted. Animation frame results with
    /// a lower generation are discarded as stale.
    scene_generation: u64,
    /// True between `FullRebuild` submission and `PreparedScene`
    /// consumption. While set, the backbone renderer's cached chains are
    /// stale — LOD must not read them.
    scene_pending: bool,
}

impl SceneProcessor {
    /// Spawn the background scene processing thread.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::Error`] if the background thread fails to spawn.
    pub fn new() -> Result<Self, std::io::Error> {
        let (request_tx, request_rx) = mpsc::channel::<SceneRequest>();
        let (scene_input, scene_output) = triple_buffer::triple_buffer(&None);
        let (anim_input, anim_output) = triple_buffer::triple_buffer(&None);

        let worker = spawn_background(move || {
            Self::thread_loop(request_rx, scene_input, anim_input);
        })?;

        Ok(Self {
            request_tx,
            scene_result: scene_output,
            anim_result: anim_output,
            worker,
            scene_generation: 0,
            scene_pending: false,
        })
    }

    /// Increment and return the next scene generation counter.
    ///
    /// Also sets `scene_pending`, which prevents LOD from reading the
    /// backbone renderer's stale cached chains until the corresponding
    /// `PreparedScene` is consumed.
    pub fn next_generation(&mut self) -> u64 {
        self.scene_generation += 1;
        self.scene_pending = true;
        self.scene_generation
    }

    /// Current scene generation counter.
    pub fn generation(&self) -> u64 {
        self.scene_generation
    }

    /// Submit a scene request (non-blocking send).
    pub fn submit(&self, request: SceneRequest) {
        let _ = self.request_tx.send(request);
    }

    /// Non-blocking check for completed full scene rebuild.
    ///
    /// Discards results whose generation is older than the current scene
    /// generation, preventing stale geometry from a previous structure
    /// from being uploaded after `replace_scene()`.
    ///
    /// Clears `scene_pending` on successful consumption so that LOD
    /// submission (gated by [`Self::is_scene_pending`]) resumes with the
    /// now-correct backbone renderer cache.
    pub fn try_recv_scene(&mut self) -> Option<PreparedScene> {
        let _ = self.scene_result.update();
        let prepared = self.scene_result.output_buffer_mut().take()?;
        if prepared.generation < self.scene_generation {
            log::debug!(
                "try_recv_scene: DISCARDING stale scene (gen {} < current {})",
                prepared.generation,
                self.scene_generation,
            );
            return None;
        }
        log::debug!(
            "try_recv_scene: ACCEPTED scene gen={} (current={})",
            prepared.generation,
            self.scene_generation,
        );
        self.scene_pending = false;
        Some(prepared)
    }

    /// Whether a `FullRebuild` has been submitted but its `PreparedScene`
    /// has not yet been consumed.
    ///
    /// While true, the backbone renderer's cached chains are stale —
    /// callers that read the cache to build `AnimationFrame` requests
    /// (notably LOD) must skip submission.
    pub fn is_scene_pending(&self) -> bool {
        self.scene_pending
    }

    /// Non-blocking check for completed animation frame.
    ///
    /// Discards frames whose generation is older than the current scene
    /// generation. As soon as a new `FullRebuild` is submitted (bumping
    /// `scene_generation`), all prior animation frames become stale —
    /// even before the rebuild result arrives on the main thread.
    pub fn try_recv_animation(&mut self) -> Option<PreparedAnimationFrame> {
        let _ = self.anim_result.update();
        let prepared = self.anim_result.output_buffer_mut().take()?;
        if prepared.generation < self.scene_generation {
            log::debug!(
                "Discarding stale animation frame (gen {} < current {})",
                prepared.generation,
                self.scene_generation,
            );
            return None;
        }
        Some(prepared)
    }

    /// Shut down the background thread and wait for it to finish.
    pub fn shutdown(&mut self) {
        let _ = self.request_tx.send(SceneRequest::Shutdown);
        join_background(&mut self.worker);
    }

    /// Background thread main loop with per-group mesh caching.
    #[allow(clippy::needless_pass_by_value)]
    fn thread_loop(
        request_rx: mpsc::Receiver<SceneRequest>,
        mut scene_input: triple_buffer::Input<Option<PreparedScene>>,
        mut anim_input: triple_buffer::Input<Option<PreparedAnimationFrame>>,
    ) {
        let mut cache = MeshCache::new();
        // Generation of the last FullRebuild processed on this thread.
        let mut last_rebuild_generation: u64 = 0;

        while let Ok(request) = request_rx.recv() {
            let latest = drain_latest(request, &request_rx);

            match latest {
                SceneRequest::Shutdown => break,
                SceneRequest::FullRebuild {
                    entities,
                    display,
                    colors,
                    geometry,
                    entity_options,
                    generation,
                } => {
                    last_rebuild_generation = generation;
                    cache.cache_stable_data(&entities, &display);
                    let entity_meshes = cache.update(
                        &entities,
                        &display,
                        &colors,
                        &geometry,
                        &entity_options,
                    );
                    let mut prepared =
                        super::mesh_concat::concatenate_meshes(&entity_meshes);
                    prepared.generation = generation;
                    scene_input.write(Some(prepared));
                }
                SceneRequest::AnimationFrame {
                    backbone_chains,
                    na_chains,
                    sidechains,
                    ss_types,
                    per_residue_colors,
                    geometry,
                    per_chain_lod,
                    generation,
                } => {
                    // Skip animation frames from a scene that has already
                    // been superseded by a newer FullRebuild.
                    if generation < last_rebuild_generation {
                        continue;
                    }

                    let na =
                        na_chains.as_deref().unwrap_or(&cache.cached_na_chains);
                    let ss = ss_types
                        .as_deref()
                        .or(cache.cached_ss_types.as_deref());
                    let colors = per_residue_colors
                        .as_deref()
                        .or(cache.cached_per_residue_colors.as_deref());
                    let prepared = super::mesh_gen::process_animation_frame(
                        &super::mesh_gen::AnimationFrameInput {
                            backbone_chains: &backbone_chains,
                            na_chains: na,
                            sidechains: sidechains.as_ref(),
                            ss_types: ss,
                            per_residue_colors: colors,
                            sheet_plane_normals: None,
                            na_base_colors: cache
                                .cached_na_base_colors
                                .as_deref(),
                            geometry: &geometry,
                            per_chain_lod: per_chain_lod.as_deref(),
                        },
                        generation,
                    );
                    anim_input.write(Some(prepared));
                }
            }
        }
    }
}

impl Drop for SceneProcessor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Per-entity mesh cache with settings-based invalidation.
///
/// Also caches stable per-scene data (NA chains, SS types, per-residue
/// colors) so that `AnimationFrame` requests can send `None` and avoid
/// cloning this data every frame.
struct MeshCache {
    meshes: FxHashMap<u32, (u64, CachedEntityMesh)>,
    last_display: Option<DisplayOptions>,
    last_colors: Option<ColorOptions>,
    last_geometry: Option<GeometryOptions>,
    /// Cached NA chains from the last `FullRebuild`.
    cached_na_chains: Vec<Vec<glam::Vec3>>,
    /// Cached SS types from the last `FullRebuild`.
    cached_ss_types: Option<Vec<molex::SSType>>,
    /// Cached per-residue colors from the last `FullRebuild`.
    cached_per_residue_colors: Option<Vec<[f32; 3]>>,
    /// Cached NA base colors from the last `FullRebuild`.
    cached_na_base_colors: Option<Vec<[f32; 3]>>,
}

impl MeshCache {
    fn new() -> Self {
        Self {
            meshes: FxHashMap::default(),
            last_display: None,
            last_colors: None,
            last_geometry: None,
            cached_na_chains: Vec::new(),
            cached_ss_types: None,
            cached_per_residue_colors: None,
            cached_na_base_colors: None,
        }
    }

    /// Cache stable per-scene data from entities so animation frames can
    /// reference it without cloning every frame.
    fn cache_stable_data(
        &mut self,
        entities: &[crate::engine::scene_data::PerEntityData],
        display: &DisplayOptions,
    ) {
        self.cached_na_chains = entities
            .iter()
            .flat_map(|e| e.nucleic_acid_chains.iter().cloned())
            .collect();
        // Only cache SS types and colors for Cartoon-mode entities —
        // these must align with `cached_chains` (also Cartoon-only) so
        // animation frames and LOD remeshes index correctly.
        let cartoon: Vec<_> = entities
            .iter()
            .filter(|e| e.drawing_mode == crate::options::DrawingMode::Cartoon)
            .cloned()
            .collect::<Vec<_>>();
        let cartoon_ranges =
            crate::engine::scene_data::compute_entity_residue_ranges(&cartoon);
        let ss: Vec<molex::SSType> =
            crate::engine::scene_data::concatenate_ss_types(
                &cartoon,
                &cartoon_ranges,
            );
        self.cached_ss_types = if ss.is_empty() { None } else { Some(ss) };
        let colors: Vec<[f32; 3]> = cartoon
            .iter()
            .flat_map(|e| {
                e.per_residue_colors.iter().flat_map(|c| c.iter().copied())
            })
            .collect();
        self.cached_per_residue_colors = if colors.is_empty() {
            None
        } else {
            Some(colors)
        };
        // Cache NA base colors for animation frames
        let na_colors: Vec<[f32; 3]> = if display.na_color_mode
            == crate::options::NaColorMode::BaseColor
        {
            entities
                .iter()
                .flat_map(|e| e.nucleic_acid_rings.iter().map(|r| r.color))
                .collect()
        } else {
            Vec::new()
        };
        self.cached_na_base_colors = if na_colors.is_empty() {
            None
        } else {
            Some(na_colors)
        };
    }

    /// Update cached meshes and return entity-ordered references for
    /// concatenation.
    fn update(
        &mut self,
        entities: &[crate::engine::scene_data::PerEntityData],
        display: &DisplayOptions,
        colors: &ColorOptions,
        geometry: &GeometryOptions,
        entity_options: &FxHashMap<u32, (DisplayOptions, GeometryOptions)>,
    ) -> Vec<(u32, &CachedEntityMesh)> {
        // Clamp geometry detail so the concatenated vertex
        // buffer stays under the wgpu 256 MB max.
        let total_residues: usize = entities
            .iter()
            .map(|e| {
                e.backbone_chains.iter().map(|c| c.len() / 3).sum::<usize>()
                    + e.nucleic_acid_chains.iter().map(Vec::len).sum::<usize>()
            })
            .sum();
        let geometry = geometry.clamped_for_residues(total_residues);

        // Any settings change (geometry, display, or colors) clears the
        // entire cache because backbone colors are baked into vertex data.
        let settings_changed = self.last_geometry.as_ref() != Some(&geometry)
            || self.last_display.as_ref() != Some(display)
            || self.last_colors.as_ref() != Some(colors);

        if settings_changed {
            self.meshes.clear();
        }
        self.last_display = Some(display.clone());
        self.last_colors = Some(colors.clone());
        self.last_geometry = Some(geometry.clone());

        // Generate or reuse per-entity meshes
        for e in entities {
            let needs_regen = self
                .meshes
                .get(&e.id)
                .is_none_or(|(v, _)| *v != e.mesh_version);
            if needs_regen {
                let (e_display, e_geometry) =
                    if let Some((d, g)) = entity_options.get(&e.id) {
                        (d, g.clamped_for_residues(total_residues))
                    } else {
                        (display, geometry.clone())
                    };
                let mesh = super::mesh_gen::generate_entity_mesh(
                    e,
                    e_display,
                    colors,
                    &e_geometry,
                );
                drop(self.meshes.insert(e.id, (e.mesh_version, mesh)));
            }
        }

        // Evict removed entities
        let active_ids: FxHashSet<u32> =
            entities.iter().map(|e| e.id).collect();
        self.meshes.retain(|id, _| active_ids.contains(id));

        // Collect references in entity order
        entities
            .iter()
            .filter_map(|e| {
                self.meshes.get(&e.id).map(|(_, mesh)| (e.id, mesh))
            })
            .collect()
    }
}

/// Drain queued requests, keeping only the latest.
///
/// Special case: a queued `AnimationFrame` does NOT replace a pending
/// `FullRebuild` — the rebuild must still run so the mesh cache is
/// populated before animation frames can reference it.
fn drain_latest(
    initial: SceneRequest,
    rx: &mpsc::Receiver<SceneRequest>,
) -> SceneRequest {
    let mut latest = initial;
    while let Ok(newer) = rx.try_recv() {
        match (&latest, &newer) {
            (
                SceneRequest::FullRebuild { .. },
                SceneRequest::AnimationFrame { .. },
            ) => {}
            _ => {
                latest = newer;
            }
        }
    }
    latest
}
