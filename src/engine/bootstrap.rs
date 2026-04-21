//! Scene loading, GPU pipeline initialization, and engine assembly.

use std::sync::Arc;
use std::time::Duration;

use glam::Vec3;
#[cfg(not(target_arch = "wasm32"))]
use molex::adapters::pdb::structure_file_to_entities;
use molex::entity::molecule::protein::ProteinEntity;
use molex::{Assembly, MoleculeEntity};
use web_time::Instant;

use super::assembly_consumer::AssemblyConsumer;
use super::density_store::DensityStore;
use super::entity_store::EntityStore;
use super::scene::{SceneTopology, VisualState};
use super::viso_state::{EntityPositions, SceneRenderState};
use super::{ConstraintSpecs, VisoEngine};
use crate::animation::AnimationState;
use crate::camera::controller::CameraController;
use crate::error::VisoError;
use crate::gpu::lighting::Lighting;
use crate::gpu::residue_color::ResidueColorBuffer;
use crate::gpu::{RenderContext, ShaderComposer};
use crate::options::VisoOptions;
use crate::renderer::picking::{PickingSystem, SelectionBuffer};
use crate::renderer::pipeline::SceneProcessor;
use crate::renderer::postprocess::PostProcessStack;
use crate::renderer::{GpuPipeline, PipelineLayouts, Renderers};

// ---------------------------------------------------------------------------
// Host-owned Assembly channel
// ---------------------------------------------------------------------------
//
// Viso's engine is a read-only consumer of structural state. In library
// deployments a host application owns the `Assembly` and publishes
// snapshots to the engine through a triple buffer. When viso runs
// standalone, construction plays that role transiently: wrap the loaded
// entities in an `Assembly`, commit the initial snapshot, drop the
// publisher. The engine keeps the consumer handle and polls it each
// frame. Once mutation methods move off the engine, the publisher will
// gain a persistent home here.

/// Writer side of the host → viso assembly channel.
struct AssemblyPublisher {
    tx: triple_buffer::Input<Option<Arc<Assembly>>>,
}

impl AssemblyPublisher {
    fn commit(&mut self, assembly: Arc<Assembly>) {
        self.tx.write(Some(assembly));
    }
}

fn assembly_channel() -> (AssemblyPublisher, AssemblyConsumer) {
    let (tx, rx) = triple_buffer::triple_buffer(&None);
    (AssemblyPublisher { tx }, AssemblyConsumer { rx })
}

/// Build an `Assembly` from the store's current entities and publish the
/// initial snapshot. Returns the consumer handle for the engine.
fn publish_initial_assembly(store: &EntityStore) -> AssemblyConsumer {
    let entities: Vec<MoleculeEntity> = store
        .entities()
        .iter()
        .map(|se| se.entity.clone())
        .collect();
    let assembly = Arc::new(Assembly::new(entities));
    let (mut publisher, consumer) = assembly_channel();
    publisher.commit(assembly);
    consumer
}

/// Target FPS limit.
const TARGET_FPS: u32 = 300;

// ---------------------------------------------------------------------------
// FrameTiming
// ---------------------------------------------------------------------------

/// Frame timing with FPS calculation and optional frame limiting.
pub(crate) struct FrameTiming {
    target_fps: u32,
    min_frame_duration: Duration,
    last_frame: Instant,
    smoothed_fps: f32,
    smoothing: f32,
    start: Instant,
}

impl FrameTiming {
    /// Create a new frame timer with the given FPS target (0 = unlimited).
    pub(super) fn new(target_fps: u32) -> Self {
        let min_frame_duration = if target_fps > 0 {
            Duration::from_secs_f64(1.0 / f64::from(target_fps))
        } else {
            Duration::ZERO
        };
        Self {
            target_fps,
            min_frame_duration,
            last_frame: Instant::now(),
            smoothed_fps: 60.0,
            smoothing: 0.05,
            start: Instant::now(),
        }
    }

    /// Returns true if enough time has passed to render the next frame.
    pub(crate) fn should_render(&self) -> bool {
        if self.target_fps == 0 {
            return true;
        }
        self.last_frame.elapsed() >= self.min_frame_duration
    }

    /// Update timing after rendering a frame.
    pub(crate) fn end_frame(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame);
        self.last_frame = now;

        let frame_time = elapsed.as_secs_f32();
        if frame_time > 0.0 {
            let instant_fps = 1.0 / frame_time;
            self.smoothed_fps = self
                .smoothed_fps
                .mul_add(1.0 - self.smoothing, instant_fps * self.smoothing);
        }
    }

    /// Current smoothed FPS.
    pub(crate) fn fps(&self) -> f32 {
        self.smoothed_fps
    }

    /// Wall-clock seconds since engine start.
    pub(crate) fn elapsed_secs(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }
}

// ---------------------------------------------------------------------------
// GpuBootstrap — intermediate state for GPU pipeline initialization
// ---------------------------------------------------------------------------

/// Intermediate state holding all initialized GPU subsystems.
///
/// Produced by [`init_gpu_pipeline`] and consumed by
/// [`VisoEngine::assemble`] to build the final engine struct.
struct GpuBootstrap {
    shader_composer: ShaderComposer,
    camera_controller: CameraController,
    lighting: Lighting,
    renderers: Renderers,
    pick: PickingSystem,
    post_process: PostProcessStack,
    entities: EntityStore,
}

/// Initialize all shared GPU subsystems from entity data.
///
/// This is the common pipeline setup for both empty and loaded constructors.
fn init_gpu_pipeline(
    context: &RenderContext,
    entities: EntityStore,
) -> Result<GpuBootstrap, VisoError> {
    let mut shader_composer = ShaderComposer::new()?;
    let mut camera_controller = CameraController::new(context);
    let lighting = Lighting::new(context);

    // Estimate residue count from visible protein entities.
    let n = entities
        .entities()
        .iter()
        .filter_map(|se| {
            se.entity.as_protein().map(|p| {
                p.to_interleaved_segments()
                    .iter()
                    .map(|c| c.len() / 3)
                    .sum::<usize>()
            })
        })
        .sum::<usize>()
        .max(1);
    let selection = SelectionBuffer::new(&context.device, n);
    let residue_colors = ResidueColorBuffer::new(&context.device, n);
    let layouts = PipelineLayouts {
        camera: camera_controller.layout.clone(),
        lighting: lighting.layout.clone(),
        selection: selection.layout.clone(),
        color: residue_colors.layout.clone(),
    };
    let post_process = PostProcessStack::new(context, &mut shader_composer)?;
    let renderers = Renderers::new(
        context,
        &layouts,
        &mut shader_composer,
        &post_process.backface_depth_view,
    )?;
    let pick = PickingSystem::new(
        context,
        &camera_controller.layout,
        selection,
        residue_colors,
        &mut shader_composer,
    )?;
    camera_controller.fit_to_sphere(Vec3::ZERO, 0.0);

    Ok(GpuBootstrap {
        shader_composer,
        camera_controller,
        lighting,
        renderers,
        pick,
        post_process,
        entities,
    })
}

// ---------------------------------------------------------------------------
// Scene loading helpers
// ---------------------------------------------------------------------------

/// Load a structure from in-memory bytes, selecting the parser based on the
/// file extension hint (`"cif"`, `"pdb"`, or `"bcif"`).
///
/// Returns a populated [`EntityStore`].
pub(super) fn load_scene_from_bytes(
    bytes: &[u8],
    format_hint: &str,
) -> Result<EntityStore, VisoError> {
    let hint = format_hint.to_ascii_lowercase();
    let hint = hint.trim_start_matches('.');
    let entities = match hint {
        "cif" | "mmcif" => {
            let text = std::str::from_utf8(bytes).map_err(|e| {
                VisoError::StructureLoad(format!("Invalid UTF-8 in CIF: {e}"))
            })?;
            molex::adapters::cif::mmcif_str_to_entities(text)
                .map_err(|e| VisoError::StructureLoad(e.to_string()))?
        }
        "pdb" | "ent" => {
            let text = std::str::from_utf8(bytes).map_err(|e| {
                VisoError::StructureLoad(format!("Invalid UTF-8 in PDB: {e}"))
            })?;
            molex::adapters::pdb::pdb_str_to_entities(text)
                .map_err(|e| VisoError::StructureLoad(e.to_string()))?
        }
        "bcif" => molex::adapters::bcif::bcif_to_entities(bytes)
            .map_err(|e| VisoError::StructureLoad(e.to_string()))?,
        other => {
            return Err(VisoError::StructureLoad(format!(
                "Unsupported format '{other}'. Use 'cif', 'pdb', or 'bcif'."
            )));
        }
    };

    for e in &entities {
        log::debug!(
            "  entity {} — {:?}: {} atoms",
            e.id(),
            e.molecule_type(),
            e.atom_count()
        );
    }

    let mut store = EntityStore::new();
    let _entity_ids = store.add_entities(entities);

    Ok(store)
}

/// Load a structure file and split into entities, returning a populated
/// [`EntityStore`].
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn load_scene_from_file(
    cif_path: &str,
) -> Result<EntityStore, VisoError> {
    let entities = structure_file_to_entities(std::path::Path::new(cif_path))
        .map_err(|e| VisoError::StructureLoad(e.to_string()))?;

    for e in &entities {
        log::debug!(
            "  entity {} — {:?}: {} atoms",
            e.id(),
            e.molecule_type(),
            e.atom_count()
        );
    }

    let mut store = EntityStore::new();
    let _entity_ids = store.add_entities(entities);

    Ok(store)
}

/// Extract backbone chains from all protein entities in the store.
fn extract_backbone_chains_from_store(store: &EntityStore) -> Vec<Vec<Vec3>> {
    store
        .entities()
        .iter()
        .filter_map(|se| se.entity.as_protein())
        .flat_map(ProteinEntity::to_interleaved_segments)
        .collect()
}

/// Compute initial per-residue colors from chain hue ramp.
pub(super) fn initial_chain_colors(
    backbone_chains: &[Vec<Vec3>],
    total_residues: usize,
) -> Vec<[f32; 3]> {
    if backbone_chains.is_empty() {
        return vec![[0.5, 0.5, 0.5]; total_residues.max(1)];
    }
    let num_chains = backbone_chains.len();
    let mut colors = Vec::with_capacity(total_residues);
    for (chain_idx, chain) in backbone_chains.iter().enumerate() {
        let t = if num_chains > 1 {
            chain_idx as f32 / (num_chains - 1) as f32
        } else {
            0.0
        };
        let color = crate::options::score_color::chain_color(t);
        let n_residues = chain.len() / 3;
        colors.extend(std::iter::repeat_n(color, n_residues));
    }
    colors
}

// ---------------------------------------------------------------------------
// VisoEngine construction
// ---------------------------------------------------------------------------

impl VisoEngine {
    /// Engine with a default molecule path.
    ///
    /// # Errors
    ///
    /// Returns [`VisoError`] if GPU initialization
    /// or structure loading fails.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn new(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        size: (u32, u32),
        scale_factor: f64,
    ) -> Result<Self, VisoError> {
        Self::new_with_path(
            window,
            size,
            scale_factor,
            "assets/models/4pnk.cif",
        )
        .await
    }

    /// Engine with a specified molecule path.
    ///
    /// # Errors
    ///
    /// Returns [`VisoError`] if GPU initialization
    /// or structure loading fails.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn new_with_path(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        size: (u32, u32),
        scale_factor: f64,
        cif_path: &str,
    ) -> Result<Self, VisoError> {
        let mut context = RenderContext::new(window, size).await?;

        // 2x supersampling on standard-DPI displays to compensate for low pixel
        // density
        if scale_factor < 2.0 {
            context.render_scale = 2;
        }

        Self::init_with_context(context, cif_path)
    }

    /// Engine from a pre-built [`RenderContext`] (for embedding in dioxus,
    /// headless rendering, etc.).
    ///
    /// Use [`RenderContext::from_device`] to create a surface-less context
    /// from an externally-owned `wgpu::Device` and `wgpu::Queue`.
    ///
    /// # Errors
    ///
    /// Returns [`VisoError`] if structure loading
    /// fails.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_from_context(
        mut context: RenderContext,
        scale_factor: f64,
        cif_path: &str,
    ) -> Result<Self, VisoError> {
        if scale_factor < 2.0 {
            context.render_scale = 2;
        }

        Self::init_with_context(context, cif_path)
    }

    /// Engine from a pre-built [`RenderContext`] and in-memory structure
    /// bytes (for WASM / headless use).
    ///
    /// `format_hint` is the file extension: `"cif"`, `"pdb"`, or `"bcif"`.
    ///
    /// # Errors
    ///
    /// Returns [`VisoError`] if structure loading or GPU initialization
    /// fails.
    pub fn new_from_bytes(
        context: RenderContext,
        bytes: &[u8],
        format_hint: &str,
    ) -> Result<Self, VisoError> {
        let mut entities = load_scene_from_bytes(bytes, format_hint)?;
        let options = VisoOptions::default();
        entities.apply_type_visibility(&options.display);

        let backbone_chains = extract_backbone_chains_from_store(&entities);
        let mut bootstrap = init_gpu_pipeline(&context, entities)?;
        let colors = initial_chain_colors(
            &backbone_chains,
            backbone_chains.iter().map(|c| c.len() / 3).sum(),
        );
        bootstrap.pick.init_colors_and_groups(
            &context,
            &colors,
            &bootstrap.renderers,
        );
        if let Some((centroid, radius)) = bootstrap.entities.bounding_sphere() {
            bootstrap.camera_controller.fit_to_sphere(centroid, radius);
        }

        Self::assemble(context, options, bootstrap)
    }

    /// Engine with an empty scene (no entities loaded).
    ///
    /// Initializes all GPU resources but starts with no visible geometry.
    /// Entities can be loaded later via [`load_entities`](Self::load_entities)
    /// or [`sync_scene_to_renderers`](Self::sync_scene_to_renderers).
    ///
    /// # Errors
    ///
    /// Returns [`VisoError`] if GPU pipeline initialization fails.
    pub fn new_empty(context: RenderContext) -> Result<Self, VisoError> {
        Self::init_empty(context)
    }

    /// Shared empty-init logic.
    fn init_empty(context: RenderContext) -> Result<Self, VisoError> {
        let entities = EntityStore::new();
        let bootstrap = init_gpu_pipeline(&context, entities)?;
        let options = VisoOptions::default();
        Self::assemble(context, options, bootstrap)
    }

    /// Shared construction logic for both windowed and headless modes.
    #[cfg(not(target_arch = "wasm32"))]
    fn init_with_context(
        context: RenderContext,
        cif_path: &str,
    ) -> Result<Self, VisoError> {
        let mut entities = load_scene_from_file(cif_path)?;
        let options = VisoOptions::default();
        entities.apply_type_visibility(&options.display);

        let backbone_chains = extract_backbone_chains_from_store(&entities);
        let mut bootstrap = init_gpu_pipeline(&context, entities)?;
        let colors = initial_chain_colors(
            &backbone_chains,
            backbone_chains.iter().map(|c| c.len() / 3).sum(),
        );
        bootstrap.pick.init_colors_and_groups(
            &context,
            &colors,
            &bootstrap.renderers,
        );
        if let Some((centroid, radius)) = bootstrap.entities.bounding_sphere() {
            bootstrap.camera_controller.fit_to_sphere(centroid, radius);
        }

        Self::assemble(context, options, bootstrap)
    }

    /// Build the final `VisoEngine` from initialized GPU subsystems.
    fn assemble(
        context: RenderContext,
        options: VisoOptions,
        bootstrap: GpuBootstrap,
    ) -> Result<Self, VisoError> {
        let (density_tx, density_rx) = std::sync::mpsc::channel();
        let assembly_consumer = publish_initial_assembly(&bootstrap.entities);
        Ok(Self {
            gpu: GpuPipeline {
                context,
                renderers: bootstrap.renderers,
                pick: bootstrap.pick,
                scene_processor: SceneProcessor::new()
                    .map_err(VisoError::ThreadSpawn)?,
                post_process: bootstrap.post_process,
                lighting: bootstrap.lighting,
                cursor_pos: (0.0, 0.0),
                last_cull_camera_eye: Vec3::ZERO,
                shader_composer: bootstrap.shader_composer,
                density_tx,
                density_rx,
            },
            camera_controller: bootstrap.camera_controller,
            topology: SceneTopology::new(),
            visual: VisualState::new(),
            entities: bootstrap.entities,
            constraints: ConstraintSpecs {
                band_specs: Vec::new(),
                pull_spec: None,
            },
            animation: AnimationState::new(),
            options,
            active_preset: None,
            frame_timing: FrameTiming::new(TARGET_FPS),
            density: DensityStore::new(),
            entity_surfaces: rustc_hash::FxHashMap::default(),
            assembly_consumer,
            scene_state: Arc::new(SceneRenderState::new()),
            entity_state: rustc_hash::FxHashMap::default(),
            positions: EntityPositions::new(),
            last_seen_generation: u64::MAX,
        })
    }
}
