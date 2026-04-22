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
//!
//! [`AnnotationsScene`] is the disjoint-borrow write view: most
//! annotation mutations also bump per-entity `mesh_version`, so the
//! write path owns coordinated access to `annotations` plus the two
//! scene fields it stamps (`entity_state`, `next_mesh_version`).
//! `VisoEngine` exposes it through `annotations_mut()`; all annotation
//! mutators on `VisoEngine` become one-line dispatchers.

use std::collections::HashMap;

use molex::entity::molecule::id::EntityId;
use molex::{MoleculeType, SSType};
use rustc_hash::FxHashMap;

use super::entity_view::EntityView;
use super::focus::Focus;
use super::surface::EntitySurface;
use super::VisoEngine;
use crate::animation::transition::Transition;
use crate::options::{DrawingMode, EntityAppearance, VisoOptions};

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

    /// Per-entity animation behavior override, if set.
    #[must_use]
    pub fn behavior(&self, id: EntityId) -> Option<&Transition> {
        self.behaviors.get(&id)
    }

    /// Record a per-entity animation behavior override.
    pub fn set_behavior(&mut self, id: EntityId, transition: Transition) {
        let _ = self.behaviors.insert(id, transition);
    }

    /// Drop a per-entity animation behavior override.
    pub fn clear_behavior(&mut self, id: EntityId) {
        let _ = self.behaviors.remove(&id);
    }

    /// Per-entity appearance override, if set.
    #[must_use]
    pub fn appearance(&self, id: EntityId) -> Option<&EntityAppearance> {
        self.appearance.get(&id)
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

    /// Resolve the drawing mode for an entity: per-entity override wins,
    /// else global (with Cartoon falling back to type-default).
    #[must_use]
    pub fn resolved_drawing_mode(
        &self,
        options: &VisoOptions,
        eid: EntityId,
        mol_type: MoleculeType,
    ) -> DrawingMode {
        self.appearance(eid)
            .and_then(|ovr| ovr.drawing_mode)
            .unwrap_or_else(|| {
                if options.display.drawing_mode == DrawingMode::Cartoon {
                    DrawingMode::default_for(mol_type)
                } else {
                    options.display.drawing_mode
                }
            })
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

/// Disjoint-borrow write view over the annotation maps plus the scene
/// fields every mutation has to stamp.
///
/// Annotation writes almost always bump an entity's `mesh_version`, so
/// this struct bundles `annotations` together with `entity_state` and
/// the `next_mesh_version` dispenser — three disjoint `&mut`s — so the
/// write path is expressible without `&mut self` methods fighting
/// over `VisoEngine`. `VisoEngine::annotations_mut` is the constructor.
pub(crate) struct AnnotationsScene<'a> {
    annotations: &'a mut EntityAnnotations,
    entity_state: &'a mut FxHashMap<EntityId, EntityView>,
    next_mesh_version: &'a mut u64,
}

impl<'a> AnnotationsScene<'a> {
    /// Construct the view from the three disjoint fields it owns.
    pub(crate) fn new(
        annotations: &'a mut EntityAnnotations,
        entity_state: &'a mut FxHashMap<EntityId, EntityView>,
        next_mesh_version: &'a mut u64,
    ) -> Self {
        Self {
            annotations,
            entity_state,
            next_mesh_version,
        }
    }

    /// Pull a fresh `mesh_version` from the dispenser.
    fn bump(&mut self) -> u64 {
        let v = *self.next_mesh_version;
        *self.next_mesh_version = self.next_mesh_version.wrapping_add(1);
        v
    }

    /// Bump the dispenser and stamp the new version onto `eid`'s
    /// `EntityView` if one exists.
    fn bump_for(&mut self, eid: EntityId) {
        let v = self.bump();
        if let Some(state) = self.entity_state.get_mut(&eid) {
            state.mesh_version = v;
        }
    }

    /// Record visibility for `eid` and invalidate its mesh.
    pub(crate) fn set_visible(&mut self, eid: EntityId, visible: bool) {
        let _ = self.annotations.visibility.insert(eid, visible);
        self.bump_for(eid);
    }

    /// Record (or clear, with `None`) per-residue scores for `eid`.
    pub(crate) fn set_per_residue_scores(
        &mut self,
        eid: EntityId,
        scores: Option<Vec<f64>>,
    ) {
        match scores {
            Some(s) => {
                let _ = self.annotations.scores.insert(eid, s);
            }
            None => {
                let _ = self.annotations.scores.remove(&eid);
            }
        }
        self.bump_for(eid);
    }

    /// Record an SS override for `eid`.
    pub(crate) fn set_ss_override(&mut self, eid: EntityId, ss: Vec<SSType>) {
        let _ = self.annotations.ss_overrides.insert(eid, ss);
        self.bump_for(eid);
    }

    /// Record an appearance override for `eid` and restamp both
    /// `mesh_version` and the resolved `drawing_mode`.
    ///
    /// The resolved drawing mode is passed in because resolution reads
    /// `VisoOptions`, which lives outside the three fields this view
    /// owns; the caller resolves it before taking the view.
    pub(crate) fn set_appearance(
        &mut self,
        eid: EntityId,
        overrides: EntityAppearance,
        drawing_mode: DrawingMode,
    ) {
        let _ = self.annotations.appearance.insert(eid, overrides);
        let v = self.bump();
        if let Some(state) = self.entity_state.get_mut(&eid) {
            state.mesh_version = v;
            state.drawing_mode = drawing_mode;
        }
    }

    /// Drop an appearance override for `eid` and restamp `mesh_version`
    /// plus the resolved `drawing_mode` (see [`Self::set_appearance`]
    /// for why the caller resolves it).
    pub(crate) fn clear_appearance(
        &mut self,
        eid: EntityId,
        drawing_mode: DrawingMode,
    ) {
        let _ = self.annotations.appearance.remove(&eid);
        let v = self.bump();
        if let Some(state) = self.entity_state.get_mut(&eid) {
            state.mesh_version = v;
            state.drawing_mode = drawing_mode;
        }
    }
}

// ── Engine-side annotation dispatchers ─────────────────────────────
//
// Each entry point either (a) takes a raw `u32`, translates to
// `EntityId` once, and forwards to [`AnnotationsScene`] /
// [`EntityAnnotations`], or (b) takes an `EntityId` and forwards
// directly. Any follow-up work that needs more than the three
// annotation-scene fields (e.g. `sync_scene_to_renderers`, reading
// `VisoOptions`) stays in the dispatcher.

impl VisoEngine {
    /// Disjoint-borrow write view over the annotations + the scene
    /// fields every annotation mutation stamps.
    pub(crate) fn annotations_mut(&mut self) -> AnnotationsScene<'_> {
        AnnotationsScene::new(
            &mut self.annotations,
            &mut self.scene.entity_state,
            &mut self.scene.next_mesh_version,
        )
    }

    /// Whether an entity is currently visible. Absent entries default
    /// to visible.
    pub(crate) fn is_entity_visible(&self, id: u32) -> bool {
        self.entity_id(id)
            .is_none_or(|eid| self.annotations.is_visible(eid))
    }

    /// Record the visibility of a single entity.
    pub(crate) fn set_entity_visible(&mut self, id: u32, visible: bool) {
        if let Some(eid) = self.entity_id(id) {
            self.annotations_mut().set_visible(eid, visible);
        }
    }

    /// Set visibility for every entity of a given molecule type.
    pub(crate) fn set_type_visibility(
        &mut self,
        mol_type: MoleculeType,
        visible: bool,
    ) {
        let ids: Vec<EntityId> = self
            .scene
            .current
            .entities()
            .iter()
            .filter(|e| e.molecule_type() == mol_type)
            .map(molex::MoleculeEntity::id)
            .collect();
        let mut view = self.annotations_mut();
        for eid in ids {
            view.set_visible(eid, visible);
        }
    }

    /// Set per-entity scores (`None` clears).
    pub(crate) fn set_per_residue_scores(
        &mut self,
        id: u32,
        scores: Option<Vec<f64>>,
    ) {
        if let Some(eid) = self.entity_id(id) {
            self.annotations_mut().set_per_residue_scores(eid, scores);
        }
    }

    /// Set an SS override for an entity.
    pub(crate) fn set_ss_override(&mut self, id: u32, ss: Vec<SSType>) {
        if let Some(eid) = self.entity_id(id) {
            self.annotations_mut().set_ss_override(eid, ss);
        }
    }

    /// Read access to the per-entity behavior override for `id`.
    pub(crate) fn entity_behavior(&self, id: u32) -> Option<&Transition> {
        self.annotations.behavior(self.entity_id(id)?)
    }

    /// Set the animation behavior for a specific entity.
    pub fn set_entity_behavior(
        &mut self,
        entity_id: EntityId,
        transition: Transition,
    ) {
        self.annotations.set_behavior(entity_id, transition);
    }

    /// Clear a per-entity behavior override.
    pub fn clear_entity_behavior(&mut self, entity_id: EntityId) {
        self.annotations.clear_behavior(entity_id);
    }

    /// Set per-entity appearance overrides.
    pub fn set_entity_appearance(
        &mut self,
        entity_id: EntityId,
        overrides: EntityAppearance,
    ) {
        let mol_type = self
            .scene
            .entity_state
            .get(&entity_id)
            .map(|s| s.topology.molecule_type);
        if let Some(mol_type) = mol_type {
            let drawing_mode = self.resolved_drawing_mode(entity_id, mol_type);
            self.annotations_mut()
                .set_appearance(entity_id, overrides, drawing_mode);
        } else {
            let _ = self.annotations.appearance.insert(entity_id, overrides);
        }
        self.sync_scene_to_renderers(HashMap::new());
    }

    /// Clear a per-entity appearance override.
    pub fn clear_entity_appearance(&mut self, entity_id: EntityId) {
        let mol_type = self
            .scene
            .entity_state
            .get(&entity_id)
            .map(|s| s.topology.molecule_type);
        if let Some(mol_type) = mol_type {
            let drawing_mode = self.resolved_drawing_mode(entity_id, mol_type);
            self.annotations_mut()
                .clear_appearance(entity_id, drawing_mode);
        } else {
            let _ = self.annotations.appearance.remove(&entity_id);
        }
        self.sync_scene_to_renderers(HashMap::new());
    }

    /// Look up a per-entity appearance override.
    #[must_use]
    pub fn entity_appearance(
        &self,
        entity_id: EntityId,
    ) -> Option<&EntityAppearance> {
        self.annotations.appearance(entity_id)
    }

    /// Resolve the drawing mode for an entity: per-entity override wins,
    /// else global (with Cartoon falling back to type-default).
    pub(crate) fn resolved_drawing_mode(
        &self,
        eid: EntityId,
        mol_type: MoleculeType,
    ) -> DrawingMode {
        self.annotations
            .resolved_drawing_mode(&self.options, eid, mol_type)
    }
}
