//! Free functions for scene loading and engine construction.

use foldit_conv::adapters::pdb::structure_file_to_coords;
use foldit_conv::render::RenderCoords;
use foldit_conv::types::entity::{split_into_entities, MoleculeEntity};
use glam::Vec3;

use super::picking_system::PickingSystem;
use super::renderers::Renderers;
use super::VisoEngine;
use crate::camera::controller::CameraController;
use crate::error::VisoError;
use crate::gpu::lighting::Lighting;
use crate::gpu::render_context::RenderContext;
use crate::gpu::residue_color::ResidueColorBuffer;
use crate::gpu::shader_composer::ShaderComposer;
use crate::options::VisoOptions;
use crate::renderer::geometry::ball_and_stick::BallAndStickRenderer;
use crate::renderer::picking::SelectionBuffer;
use crate::renderer::postprocess::post_process::PostProcessStack;
use crate::renderer::PipelineLayouts;
use crate::scene::{Scene, SceneEntity};
use crate::util::bond_topology::{get_residue_bonds, is_hydrophobic};

/// Load a structure file and split into entities, returning a populated Scene
/// and the derived protein `RenderCoords`.
pub(super) fn load_scene_from_file(
    cif_path: &str,
) -> Result<(Scene, RenderCoords), VisoError> {
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

    let mut scene = Scene::new();
    let entity_ids = scene.add_entities(entities);

    let render_coords = extract_render_coords(&scene, &entity_ids);
    Ok((scene, render_coords))
}

/// Derive protein `RenderCoords` from a populated scene.
pub(super) fn extract_render_coords(
    scene: &Scene,
    entity_ids: &[u32],
) -> RenderCoords {
    let protein_entity_id = entity_ids
        .iter()
        .find(|&&id| scene.entity(id).is_some_and(SceneEntity::is_protein));

    if let Some(protein_coords) = protein_entity_id
        .and_then(|&id| scene.entity(id).and_then(SceneEntity::protein_coords))
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
        let color = VisoEngine::chain_color(t);
        let n_residues = chain.len() / 3;
        colors.extend(std::iter::repeat_n(color, n_residues));
    }
    colors
}

/// Intermediate state holding all initialized GPU subsystems.
///
/// Produced by [`init_gpu_pipeline`] and consumed by
/// [`VisoEngine::assemble`](super::VisoEngine::assemble) to build the final
/// engine struct.
pub(super) struct GpuBootstrap {
    pub shader_composer: ShaderComposer,
    pub camera_controller: CameraController,
    pub lighting: Lighting,
    pub renderers: Renderers,
    pub pick: PickingSystem,
    pub post_process: PostProcessStack,
    pub scene: Scene,
}

/// Initialize all shared GPU subsystems from a scene and render coords.
///
/// This is the common pipeline setup for both empty and loaded constructors.
pub(super) fn init_gpu_pipeline(
    context: &RenderContext,
    scene: Scene,
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
        &scene,
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
    camera_controller.fit_to_positions(&[]);

    Ok(GpuBootstrap {
        shader_composer,
        camera_controller,
        lighting,
        renderers,
        pick,
        post_process,
        scene,
    })
}

/// Collect all atom positions for initial camera fit (protein + ligands + NA).
pub(super) fn collect_all_positions(
    render_coords: &RenderCoords,
    scene: &Scene,
    options: &VisoOptions,
) -> Vec<Vec3> {
    let mut positions = render_coords.all_positions.clone();
    let non_protein: Vec<MoleculeEntity> = scene
        .entities()
        .iter()
        .filter(|se| !se.is_protein())
        .map(|se| se.entity.clone())
        .collect();
    positions.extend(BallAndStickRenderer::collect_positions(
        &non_protein,
        &options.display,
    ));
    for chain in scene
        .nucleic_acid_entities()
        .iter()
        .flat_map(|se| se.entity.extract_p_atom_chains())
    {
        positions.extend(&chain);
    }
    positions
}
