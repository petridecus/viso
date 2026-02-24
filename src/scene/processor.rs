//! Background scene processor for non-blocking geometry generation.
//!
//! Moves all CPU-heavy mesh/instance generation off the main thread.
//! The main thread only does GPU uploads (<1ms) and render passes.
//!
//! Supports **per-entity mesh caching**: when an entity's `mesh_version`
//! hasn't changed between frames, its cached mesh is reused instead of
//! being regenerated. Global settings changes (view mode, display,
//! colors) clear the entire cache.

use std::{
    collections::{HashMap, HashSet},
    sync::mpsc,
};

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
    fn thread_loop(
        request_rx: mpsc::Receiver<SceneRequest>,
        mut scene_input: triple_buffer::Input<Option<PreparedScene>>,
        mut anim_input: triple_buffer::Input<Option<PreparedAnimationFrame>>,
    ) {
        let mut mesh_cache: HashMap<u32, (u64, CachedEntityMesh)> =
            HashMap::new();
        let mut last_display: Option<DisplayOptions> = None;
        let mut last_colors: Option<ColorOptions> = None;
        let mut last_geometry: Option<GeometryOptions> = None;

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
                    // Clamp geometry detail so the concatenated vertex
                    // buffer stays under the wgpu 256 MB max.
                    let total_residues: usize = entities
                        .iter()
                        .map(|e| {
                            e.backbone_chains
                                .iter()
                                .map(|c| c.len() / 3)
                                .sum::<usize>()
                                + e.nucleic_acid_chains
                                    .iter()
                                    .map(Vec::len)
                                    .sum::<usize>()
                        })
                        .sum();
                    let geometry =
                        geometry.clamped_for_residues(total_residues);

                    // Clear cache if global settings changed
                    let settings_changed = last_display.as_ref()
                        != Some(&display)
                        || last_colors.as_ref() != Some(&colors)
                        || last_geometry.as_ref() != Some(&geometry);

                    if settings_changed {
                        mesh_cache.clear();
                        last_display = Some(display.clone());
                        last_colors = Some(colors.clone());
                        last_geometry = Some(geometry.clone());
                    }

                    // Generate or reuse per-entity meshes
                    for e in &entities {
                        let needs_regen = match mesh_cache.get(&e.id) {
                            Some((cached_version, _)) => {
                                *cached_version != e.mesh_version
                            }
                            None => true,
                        };

                        if needs_regen {
                            let mesh = super::mesh_gen::generate_entity_mesh(
                                e, &display, &colors, &geometry,
                            );
                            let _ =
                                mesh_cache.insert(e.id, (e.mesh_version, mesh));
                        }
                    }

                    // Evict removed entities
                    let active_ids: HashSet<u32> =
                        entities.iter().map(|e| e.id).collect();
                    mesh_cache.retain(|id, _| active_ids.contains(id));

                    // Collect references in entity order
                    let entity_meshes: Vec<(u32, &CachedEntityMesh)> = entities
                        .iter()
                        .filter_map(|e| {
                            mesh_cache.get(&e.id).map(|(_, mesh)| (e.id, mesh))
                        })
                        .collect();

                    // Concatenate into PreparedScene
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
                        backbone_chains,
                        na_chains,
                        sidechains,
                        ss_types,
                        per_residue_colors,
                        &geometry,
                        per_chain_lod,
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
