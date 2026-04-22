//! Per-entity annotations: user-authored state that rides alongside
//! the [`Assembly`](molex::Assembly) snapshot.
//!
//! Everything in [`EntityAnnotations`] is state the caller (UI, IPC,
//! command path) has expressed *about* entities — not state derived
//! from the Assembly itself. The Assembly-derived half of the engine
//! (topology, positions, render state) is rebuilt on every sync;
//! annotations persist across syncs and only get reconciled when an
//! entity leaves the assembly entirely.
//!
//! All maps are keyed on the opaque [`EntityId`] so lookups are O(1).
//! Callers arriving with a raw `u32` (UI commands, IPC) translate
//! *once* at the API boundary via [`VisoEngine::entity_id`] and then
//! pass the [`EntityId`] down. Engine-side mutation methods live at
//! the bottom of this file as impls on [`VisoEngine`].
//!
//! Reconciliation on [`Assembly`](molex::Assembly) sync happens via
//! [`EntityAnnotations::retain_entities`], which drops entries for
//! IDs that have left the assembly.
//!
//! The `Focus` arm holds an [`EntityId`] for the same reason the maps
//! do; the UI-side `u32` is recovered via `EntityId::raw()` when
//! needed by external consumers (the bridge, the webview).

use std::collections::HashMap;

use molex::entity::molecule::id::EntityId;
use molex::{MoleculeType, SSType};
use rustc_hash::FxHashMap;

use super::focus::Focus;
use super::surface::EntitySurface;
use super::VisoEngine;
use crate::animation::transition::Transition;
use crate::options::{DrawingMode, EntityAppearance};

/// Per-entity user-authored state that isn't derived from the
/// [`Assembly`](molex::Assembly).
#[derive(Default)]
pub(crate) struct EntityAnnotations {
    /// Focus state (session-wide or a specific entity).
    pub focus: Focus,
    /// Per-entity visibility (`true` = drawn). Missing = visible.
    pub visibility: FxHashMap<EntityId, bool>,
    /// Per-entity animation behavior overrides.
    pub behaviors: FxHashMap<EntityId, Transition>,
    /// Per-entity appearance overrides (None-fields inherit global).
    pub appearance: FxHashMap<EntityId, EntityAppearance>,
    /// Per-entity scores (for color-by-score visualization).
    pub scores: FxHashMap<EntityId, Vec<f64>>,
    /// Per-entity SS overrides (from puzzle annotations).
    pub ss_overrides: FxHashMap<EntityId, Vec<SSType>>,
    /// Per-entity molecular surfaces.
    pub surfaces: FxHashMap<EntityId, EntitySurface>,
}

impl EntityAnnotations {
    /// Whether `id` is currently visible. Entities with no explicit
    /// entry default to visible.
    #[must_use]
    pub fn is_visible(&self, id: EntityId) -> bool {
        self.visibility.get(&id).copied().unwrap_or(true)
    }

    /// Drop entries for entities no longer present in the assembly.
    pub fn retain_entities(&mut self, keep: impl Fn(EntityId) -> bool) {
        self.visibility.retain(|&id, _| keep(id));
        self.behaviors.retain(|&id, _| keep(id));
        self.appearance.retain(|&id, _| keep(id));
        self.scores.retain(|&id, _| keep(id));
        self.ss_overrides.retain(|&id, _| keep(id));
        self.surfaces.retain(|&id, _| keep(id));
    }

    /// Advance `focus` to the next entity in `focusable`, wrapping
    /// back to `Session` after the last. Returns the new focus.
    ///
    /// `focusable` is the engine-filtered list of entities eligible
    /// for focus (typically: visible, focusable molecule type). The
    /// annotation owns focus state but not the filter, so callers
    /// build the list themselves.
    pub fn cycle_focus(&mut self, focusable: &[EntityId]) -> Focus {
        self.focus = match self.focus {
            Focus::Session => focusable
                .first()
                .map_or(Focus::Session, |&id| Focus::Entity(id)),
            Focus::Entity(current) => {
                let idx = focusable.iter().position(|&id| id == current);
                match idx {
                    Some(i) if i + 1 < focusable.len() => {
                        Focus::Entity(focusable[i + 1])
                    }
                    _ => Focus::Session,
                }
            }
        };
        self.focus
    }

    /// Clear every annotation back to the default state (focus back
    /// to session-wide, every map emptied).
    pub fn reset(&mut self) {
        self.focus = Focus::default();
        self.visibility.clear();
        self.behaviors.clear();
        self.appearance.clear();
        self.scores.clear();
        self.ss_overrides.clear();
        self.surfaces.clear();
    }
}

// ── Engine-side annotation mutations (u32 API for VisoApp / bridge) ──
//
// Each entry point translates raw `u32` -> `EntityId` once, then
// operates on the annotation maps by opaque id. Mutations that
// invalidate rendered geometry also bump `scene.mesh_version`.

impl VisoEngine {
    /// Whether an entity is currently visible. Absent entries default
    /// to visible.
    pub(crate) fn is_entity_visible(&self, id: u32) -> bool {
        self.entity_id(id)
            .is_none_or(|eid| self.annotations.is_visible(eid))
    }

    /// Record the visibility of a single entity.
    pub(crate) fn set_entity_visible_internal(
        &mut self,
        id: u32,
        visible: bool,
    ) {
        let Some(eid) = self.entity_id(id) else {
            return;
        };
        let _ = self.annotations.visibility.insert(eid, visible);
        let v = self.scene.bump_mesh_version();
        if let Some(state) = self.scene.entity_state.get_mut(&eid) {
            state.mesh_version = v;
        }
    }

    /// Set visibility for every entity of a given molecule type.
    pub(crate) fn set_type_visibility_internal(
        &mut self,
        mol_type: MoleculeType,
        visible: bool,
    ) {
        let ids: Vec<u32> = self
            .scene
            .current
            .entities()
            .iter()
            .filter(|e| e.molecule_type() == mol_type)
            .map(|e| e.id().raw())
            .collect();
        for id in ids {
            self.set_entity_visible_internal(id, visible);
        }
    }

    /// Set per-entity scores (`None` clears).
    pub(crate) fn set_per_residue_scores_internal(
        &mut self,
        id: u32,
        scores: Option<Vec<f64>>,
    ) {
        let Some(eid) = self.entity_id(id) else {
            return;
        };
        match scores {
            Some(s) => {
                let _ = self.annotations.scores.insert(eid, s);
            }
            None => {
                let _ = self.annotations.scores.remove(&eid);
            }
        }
        let v = self.scene.bump_mesh_version();
        if let Some(state) = self.scene.entity_state.get_mut(&eid) {
            state.mesh_version = v;
        }
    }

    /// Set an SS override for an entity.
    pub(crate) fn set_ss_override_internal(
        &mut self,
        id: u32,
        ss: Vec<SSType>,
    ) {
        let Some(eid) = self.entity_id(id) else {
            return;
        };
        let _ = self.annotations.ss_overrides.insert(eid, ss);
        let v = self.scene.bump_mesh_version();
        if let Some(state) = self.scene.entity_state.get_mut(&eid) {
            state.mesh_version = v;
        }
    }

    /// Clear an entity's per-entity animation behavior override.
    pub(crate) fn clear_entity_behavior_internal(&mut self, id: u32) {
        if let Some(eid) = self.entity_id(id) {
            let _ = self.annotations.behaviors.remove(&eid);
        }
    }

    /// Read access to the per-entity behavior override for `id`.
    pub(crate) fn entity_behavior(&self, id: u32) -> Option<&Transition> {
        let eid = self.entity_id(id)?;
        self.annotations.behaviors.get(&eid)
    }

    /// Set the animation behavior for a specific entity.
    pub fn set_entity_behavior(
        &mut self,
        entity_id: u32,
        transition: Transition,
    ) {
        if let Some(eid) = self.entity_id(entity_id) {
            let _ = self.annotations.behaviors.insert(eid, transition);
        }
    }

    /// Clear a per-entity behavior override.
    pub fn clear_entity_behavior(&mut self, entity_id: u32) {
        self.clear_entity_behavior_internal(entity_id);
    }

    /// Set per-entity appearance overrides.
    pub fn set_entity_appearance(
        &mut self,
        entity_id: u32,
        overrides: EntityAppearance,
    ) {
        let Some(eid) = self.entity_id(entity_id) else {
            return;
        };
        let _ = self.annotations.appearance.insert(eid, overrides);
        self.apply_appearance_change(eid);
        self.sync_scene_to_renderers(HashMap::new());
    }

    /// Clear a per-entity appearance override.
    pub fn clear_entity_appearance(&mut self, entity_id: u32) {
        let Some(eid) = self.entity_id(entity_id) else {
            return;
        };
        let _ = self.annotations.appearance.remove(&eid);
        self.apply_appearance_change(eid);
        self.sync_scene_to_renderers(HashMap::new());
    }

    /// Look up a per-entity appearance override.
    #[must_use]
    pub fn entity_appearance(
        &self,
        entity_id: u32,
    ) -> Option<&EntityAppearance> {
        let eid = self.entity_id(entity_id)?;
        self.annotations.appearance.get(&eid)
    }

    /// Resolve the drawing mode for an entity: per-entity override wins,
    /// else global (with Cartoon falling back to type-default).
    pub(crate) fn resolved_drawing_mode(
        &self,
        eid: EntityId,
        mol_type: MoleculeType,
    ) -> DrawingMode {
        self.annotations
            .appearance
            .get(&eid)
            .and_then(|ovr| ovr.drawing_mode)
            .unwrap_or_else(|| {
                if self.options.display.drawing_mode == DrawingMode::Cartoon {
                    DrawingMode::default_for(mol_type)
                } else {
                    self.options.display.drawing_mode
                }
            })
    }

    /// Bump mesh version and re-resolve drawing mode for an entity
    /// whose appearance override just changed. Internal helper that
    /// sidesteps the borrow-check pitfall of reading `self` while
    /// holding a `&mut` into `entity_state`.
    fn apply_appearance_change(&mut self, eid: EntityId) {
        let mol_type = self
            .scene
            .entity_state
            .get(&eid)
            .map(|s| s.topology.molecule_type);
        let Some(mol_type) = mol_type else {
            return;
        };
        let drawing_mode = self.resolved_drawing_mode(eid, mol_type);
        let v = self.scene.bump_mesh_version();
        if let Some(state) = self.scene.entity_state.get_mut(&eid) {
            state.mesh_version = v;
            state.drawing_mode = drawing_mode;
        }
    }
}
