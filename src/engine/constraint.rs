//! Constraint resolution: resolve band/pull specs to world-space
//! positions by walking
//! [`crate::renderer::entity_topology::EntityTopology`] +
//! [`super::positions::EntityPositions`].

use glam::{UVec2, Vec2, Vec3};

use super::command::{
    AtomRef, BandInfo, BandTarget, PullInfo, ResolvedBand, ResolvedPull,
};
use super::VisoEngine;
use crate::camera::controller::CameraController;

/// Resolve a single band spec to world-space endpoint positions.
pub(super) fn resolve_band(
    engine: &VisoEngine,
    band: &BandInfo,
) -> Option<ResolvedBand> {
    let endpoint_a = resolve_atom_ref(engine, &band.anchor_a)?;
    let endpoint_b = match &band.anchor_b {
        BandTarget::Atom(atom) => resolve_atom_ref(engine, atom)?,
        BandTarget::Position(pos) => *pos,
    };
    let is_space_pull = matches!(band.anchor_b, BandTarget::Position(_));

    Some(ResolvedBand {
        endpoint_a,
        endpoint_b,
        is_disabled: band.is_disabled,
        strength: band.strength,
        target_length: band.target_length,
        residue_idx: band.anchor_a.residue,
        is_space_pull,
        band_type: band.band_type,
        from_script: band.from_script,
    })
}

/// Resolve a pull spec to world-space atom and target positions.
pub(super) fn resolve_pull(
    engine: &VisoEngine,
    camera: &CameraController,
    viewport: (u32, u32),
    pull: &PullInfo,
) -> Option<ResolvedPull> {
    let atom_pos = resolve_atom_ref(engine, &pull.atom)?;
    let target_pos = camera.screen_to_world_at_depth(
        Vec2::new(pull.screen_target.0, pull.screen_target.1),
        UVec2::new(viewport.0, viewport.1),
        atom_pos,
    );

    Some(ResolvedPull {
        atom_pos,
        target_pos,
        residue_idx: pull.atom.residue,
    })
}

/// Public entry point for [`resolve_atom_ref`] (used by
/// [`VisoEngine::resolve_atom_position`]).
pub(super) fn resolve_atom_ref_pub(
    engine: &VisoEngine,
    atom: &AtomRef,
) -> Option<Vec3> {
    resolve_atom_ref(engine, atom)
}

/// Resolve an [`AtomRef`] to a world-space position.
///
/// The `residue` index is treated as a flat index across Cartoon-mode
/// protein entities in assembly order (matching the convention of
/// [`VisoEngine::concatenated_cartoon_ss`] and the GPU picking path
/// that produced it). Backbone atoms (`N`, `CA`, `C`) resolve against
/// each entity's topology `backbone_chain_layout`; other atom names
/// search the entity's sidechain layout by name.
fn resolve_atom_ref(engine: &VisoEngine, atom: &AtomRef) -> Option<Vec3> {
    let mut residues_seen: u32 = 0;
    for entity in engine.current_assembly.entities() {
        let eid = entity.id();
        if !engine.is_entity_visible(eid.raw()) {
            continue;
        }
        let state = engine.entity_state.get(&eid)?;
        if !state.topology.is_protein() {
            continue;
        }
        let residue_count = state.topology.residue_atom_ranges.len() as u32;
        if atom.residue >= residues_seen + residue_count {
            residues_seen += residue_count;
            continue;
        }
        let local_residue = atom.residue - residues_seen;
        let positions = engine.positions.get(eid)?;
        return resolve_atom_in_entity(
            state,
            positions,
            local_residue,
            &atom.atom_name,
        );
    }
    None
}

fn resolve_atom_in_entity(
    state: &super::entity_view::EntityView,
    positions: &[Vec3],
    local_residue: u32,
    atom_name: &str,
) -> Option<Vec3> {
    match atom_name {
        "N" | "CA" | "C" => {
            let range =
                state.topology.residue_atom_ranges.get(local_residue as usize)?;
            let offset = match atom_name {
                "N" => 0,
                "CA" => 1,
                "C" => 2,
                _ => return None,
            };
            let idx = range.start as usize + offset;
            positions.get(idx).copied()
        }
        other => {
            let layout = &state.topology.sidechain_layout;
            for (i, (&ri, name)) in layout
                .residue_indices
                .iter()
                .zip(layout.atom_names.iter())
                .enumerate()
            {
                if ri == local_residue && name == other {
                    let atom_idx = layout.atom_indices.get(i)?;
                    return positions.get(*atom_idx as usize).copied();
                }
            }
            None
        }
    }
}

// ── Engine-side wiring ──

impl VisoEngine {
    /// Resolve stored band/pull specs to world-space and update
    /// renderers.
    pub(super) fn resolve_and_render_constraints(&mut self) {
        let resolved_bands: Vec<_> = self
            .constraints
            .band_specs
            .iter()
            .filter_map(|b| resolve_band(self, b))
            .collect();
        self.gpu.renderers.band.update(
            &self.gpu.context.device,
            &self.gpu.context.queue,
            &resolved_bands,
            Some(&self.options.colors),
        );

        let viewport = (
            self.gpu.context.config.width,
            self.gpu.context.config.height,
        );
        let resolved_pull = self.constraints.pull_spec.as_ref().and_then(|p| {
            resolve_pull(self, &self.camera_controller, viewport, p)
        });
        self.gpu.renderers.pull.update(
            &self.gpu.context.device,
            &self.gpu.context.queue,
            resolved_pull.as_ref(),
        );
    }
}
