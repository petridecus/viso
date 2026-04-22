//! Viso-side per-entity overlays.
//!
//! Captures user-authored opinions about entities that are *not*
//! derived from the [`Assembly`](molex::Assembly) snapshot — visibility
//! toggles, behavior transitions, appearance overrides, scores, SS
//! overrides, surfaces, and focus state.
//!
//! All maps are keyed on the opaque [`EntityId`] so lookups are O(1).
//! Callers arriving with a raw `u32` (UI commands, IPC) translate
//! *once* at the API boundary via
//! [`Scene::entity_id`](super::scene::Scene::entity_id) and then pass
//! the [`EntityId`] down.
//!
//! Reconciliation on [`Assembly`] sync happens via
//! [`EntityOverlays::retain_entities`], which drops entries for IDs
//! that have left the assembly.
//!
//! The `Focus` arm holds an [`EntityId`] for the same reason the maps
//! do; the UI-side `u32` is recovered via `EntityId::raw()` when
//! needed by external consumers (the bridge, the webview).

use molex::entity::molecule::id::EntityId;
use molex::SSType;
use rustc_hash::FxHashMap;

use super::focus::Focus;
use super::surface::EntitySurface;
use crate::animation::transition::Transition;
use crate::options::EntityAppearance;

/// Per-entity user-facing overlays.
#[derive(Default)]
pub(crate) struct EntityOverlays {
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

impl EntityOverlays {
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
    /// overlay owns focus state but not the filter, so callers build
    /// the list themselves.
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

    /// Clear every overlay back to the default state (focus back to
    /// session-wide, every map emptied).
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
