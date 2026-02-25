//! Background scene processor for non-blocking geometry generation.
//!
//! Moves all CPU-heavy mesh/instance generation off the main thread.
//! The main thread only does GPU uploads (<1ms) and render passes.
//!
//! Supports **per-entity mesh caching**: when an entity's `mesh_version`
//! hasn't changed between frames, its cached mesh is reused instead of
//! being regenerated. Global settings changes (view mode, display,
//! colors) clear the entire cache.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc;

use super::prepared::{
    CachedEntityMesh, PreparedAnimationFrame, PreparedScene, SceneRequest,
};
use crate::options::{ColorOptions, DisplayOptions, GeometryOptions};

/// Background thread that generates CPU-side geometry from scene data.
pub struct SceneProcessor {
    request_tx: mpsc::Sender<SceneRequest>,
    scene_result: triple_buffer::Output<Option<PreparedScene>>,
    anim_result: triple_buffer::Output<Option<PreparedAnimationFrame>>,
    thread: Option<std::thread::JoinHandle<()>>,
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

        let thread = std::thread::Builder::new()
            .name("scene-processor".into())
            .spawn(move || {
                Self::thread_loop(request_rx, scene_input, anim_input);
            })?;

        Ok(Self {
            request_tx,
            scene_result: scene_output,
            anim_result: anim_output,
            thread: Some(thread),
        })
    }

    /// Submit a scene request (non-blocking send).
    pub fn submit(&self, request: SceneRequest) {
        let _ = self.request_tx.send(request);
    }

    /// Non-blocking check for completed full scene rebuild.
    pub fn try_recv_scene(&mut self) -> Option<PreparedScene> {
        let _ = self.scene_result.update();
        self.scene_result.output_buffer_mut().take()
    }

    /// Non-blocking check for completed animation frame.
    pub fn try_recv_animation(&mut self) -> Option<PreparedAnimationFrame> {
        let _ = self.anim_result.update();
        self.anim_result.output_buffer_mut().take()
    }

    /// Shut down the background thread and wait for it to finish.
    pub fn shutdown(&mut self) {
        let _ = self.request_tx.send(SceneRequest::Shutdown);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }

    /// Background thread main loop with per-group mesh caching.
    #[allow(clippy::needless_pass_by_value)]
    fn thread_loop(
        request_rx: mpsc::Receiver<SceneRequest>,
        mut scene_input: triple_buffer::Input<Option<PreparedScene>>,
        mut anim_input: triple_buffer::Input<Option<PreparedAnimationFrame>>,
    ) {
        let mut cache = MeshCache::new();

        while let Ok(request) = request_rx.recv() {
            let latest = drain_latest(request, &request_rx);

            match latest {
                SceneRequest::Shutdown => break,
                SceneRequest::FullRebuild {
                    entities,
                    entity_transitions,
                    display,
                    colors,
                    geometry,
                } => {
                    let entity_meshes =
                        cache.update(&entities, &display, &colors, &geometry);
                    let prepared = super::mesh_concat::concatenate_meshes(
                        &entity_meshes,
                        entity_transitions,
                    );
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
                } => {
                    let prepared = super::mesh_gen::process_animation_frame(
                        &super::mesh_gen::AnimationFrameInput {
                            backbone_chains: &backbone_chains,
                            na_chains: &na_chains,
                            sidechains: sidechains.as_ref(),
                            ss_types: ss_types.as_deref(),
                            per_residue_colors: per_residue_colors.as_deref(),
                            geometry: &geometry,
                            per_chain_lod: per_chain_lod.as_deref(),
                        },
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
struct MeshCache {
    meshes: HashMap<u32, (u64, CachedEntityMesh)>,
    last_display: Option<DisplayOptions>,
    last_colors: Option<ColorOptions>,
    last_geometry: Option<GeometryOptions>,
}

impl MeshCache {
    fn new() -> Self {
        Self {
            meshes: HashMap::new(),
            last_display: None,
            last_colors: None,
            last_geometry: None,
        }
    }

    /// Update cached meshes and return entity-ordered references for
    /// concatenation.
    fn update(
        &mut self,
        entities: &[super::entity_data::PerEntityData],
        display: &DisplayOptions,
        colors: &ColorOptions,
        geometry: &GeometryOptions,
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

        // Clear cache if global settings changed
        let settings_changed = self.last_display.as_ref() != Some(display)
            || self.last_colors.as_ref() != Some(colors)
            || self.last_geometry.as_ref() != Some(&geometry);

        if settings_changed {
            self.meshes.clear();
            self.last_display = Some(display.clone());
            self.last_colors = Some(colors.clone());
            self.last_geometry = Some(geometry.clone());
        }

        // Generate or reuse per-entity meshes
        for e in entities {
            let needs_regen = self
                .meshes
                .get(&e.id)
                .is_none_or(|(v, _)| *v != e.mesh_version);
            if needs_regen {
                let mesh = super::mesh_gen::generate_entity_mesh(
                    e, display, colors, &geometry,
                );
                drop(self.meshes.insert(e.id, (e.mesh_version, mesh)));
            }
        }

        // Evict removed entities
        let active_ids: HashSet<u32> = entities.iter().map(|e| e.id).collect();
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
