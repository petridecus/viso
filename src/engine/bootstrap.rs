//! Scene loading, GPU pipeline initialization, and engine assembly.

use std::time::{Duration, Instant};

use foldit_conv::adapters::pdb::structure_file_to_coords;
use foldit_conv::render::RenderCoords;
use foldit_conv::types::entity::split_into_entities;
use glam::Vec3;

use super::entity_store::EntityStore;
use super::scene::{SceneTopology, VisualState};
use super::scene_data::{get_residue_bonds, is_hydrophobic, SceneEntity};
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

/// Initialize all shared GPU subsystems from entity data and render coords.
///
/// This is the common pipeline setup for both empty and loaded constructors.
fn init_gpu_pipeline(
    context: &RenderContext,
    entities: EntityStore,
    render_coords: &RenderCoords,
) -> Result<GpuBootstrap, VisoError> {
    let mut shader_composer = ShaderComposer::new()?;
    let mut camera_controller = CameraController::new(context);
    let lighting = Lighting::new(context);

    let n = render_coords.residue_count().max(1);
    let selection = SelectionBuffer::new(&context.device, n);
    let residue_colors = ResidueColorBuffer::new(&context.device, n);
    let layouts = PipelineLayouts {
        camera: camera_controller.layout.clone(),
        lighting: lighting.layout.clone(),
        selection: selection.layout.clone(),
        color: residue_colors.layout.clone(),
    };
    let renderers = Renderers::new(
        context,
        &layouts,
        render_coords,
        &entities,
        &mut shader_composer,
    )?;
    let pick = PickingSystem::new(
        context,
        &camera_controller.layout,
        selection,
        residue_colors,
        &mut shader_composer,
    )?;
    let post_process = PostProcessStack::new(context, &mut shader_composer)?;
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

/// Load a structure file and split into entities, returning a populated
/// [`EntityStore`] and the derived protein `RenderCoords`.
pub(super) fn load_scene_from_file(
    cif_path: &str,
) -> Result<(EntityStore, RenderCoords), VisoError> {
    let coords = structure_file_to_coords(std::path::Path::new(cif_path))
        .map_err(|e| VisoError::StructureLoad(e.to_string()))?;

    let entities = split_into_entities(&coords);

    for e in &entities {
        log::debug!(
            "  entity {} — {:?}: {} atoms",
            e.entity_id,
            e.molecule_type,
            e.coords.num_atoms
        );
    }

    let mut store = EntityStore::new();
    let entity_ids = store.add_entities(entities);

    let render_coords = extract_render_coords(&store, &entity_ids);
    Ok((store, render_coords))
}

/// Derive protein `RenderCoords` from a populated entity store.
pub(super) fn extract_render_coords(
    store: &EntityStore,
    entity_ids: &[u32],
) -> RenderCoords {
    let protein_entity_id = entity_ids
        .iter()
        .find(|&&id| store.entity(id).is_some_and(SceneEntity::is_protein));

    if let Some(protein_coords) = protein_entity_id
        .and_then(|&id| store.entity(id).and_then(SceneEntity::protein_coords))
    {
        log::debug!("protein_coords: {} atoms", protein_coords.num_atoms);
        let protein_coords =
            foldit_conv::ops::transform::protein_only(&protein_coords);
        log::debug!("after protein_only: {} atoms", protein_coords.num_atoms);
        let rc = RenderCoords::from_coords_with_topology(
            &protein_coords,
            is_hydrophobic,
            |name| get_residue_bonds(name).map(<[(&str, &str)]>::to_vec),
        );
        log::debug!(
            "render_coords: {} backbone chains, {} residues",
            rc.backbone_chains.len(),
            rc.backbone_chains
                .iter()
                .map(|c| c.len() / 3)
                .sum::<usize>()
        );
        rc
    } else {
        log::debug!("no protein coords found");
        empty_render_coords()
    }
}

/// Build an empty `RenderCoords` (zero atoms, no topology).
pub(super) fn empty_render_coords() -> RenderCoords {
    let empty = foldit_conv::types::coords::Coords {
        num_atoms: 0,
        atoms: Vec::new(),
        chain_ids: Vec::new(),
        res_names: Vec::new(),
        res_nums: Vec::new(),
        atom_names: Vec::new(),
        elements: Vec::new(),
    };
    RenderCoords::from_coords_with_topology(&empty, is_hydrophobic, |name| {
        get_residue_bonds(name).map(<[(&str, &str)]>::to_vec)
    })
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
        let render_coords = empty_render_coords();
        let entities = EntityStore::new();
        let bootstrap = init_gpu_pipeline(&context, entities, &render_coords)?;
        let options = VisoOptions::default();
        Self::assemble(context, options, bootstrap)
    }

    /// Shared construction logic for both windowed and headless modes.
    fn init_with_context(
        context: RenderContext,
        cif_path: &str,
    ) -> Result<Self, VisoError> {
        let (entities, render_coords) = load_scene_from_file(cif_path)?;
        let options = VisoOptions::default();

        let mut bootstrap =
            init_gpu_pipeline(&context, entities, &render_coords)?;
        bootstrap.renderers.init_ball_and_stick_entities(
            &context,
            &bootstrap.entities,
            &options,
        );
        let colors = initial_chain_colors(
            &render_coords.backbone_chains,
            render_coords
                .backbone_chains
                .iter()
                .map(|c| c.len() / 3)
                .sum(),
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
        })
    }
}
