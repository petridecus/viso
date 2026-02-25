use std::time::{Duration, Instant};

use crate::renderer::picking::PickTarget;

const DOUBLE_CLICK_THRESHOLD: Duration = Duration::from_millis(400);

/// Result of processing a mouse-up event through the multi-click state machine.
pub enum ClickResult {
    /// No selection action (drag, mismatched up/down, etc.)
    NoAction,
    /// Single click on a target.
    SingleClick {
        /// The pick target that was clicked.
        target: PickTarget,
        /// Whether shift was held during the click.
        shift_held: bool,
    },
    /// Double-click on a target — select secondary structure segment.
    DoubleClick {
        /// The pick target that was double-clicked.
        target: PickTarget,
        /// Whether shift was held during the click.
        shift_held: bool,
    },
    /// Triple-click on a target — select entire chain.
    TripleClick {
        /// The pick target that was triple-clicked.
        target: PickTarget,
        /// Whether shift was held during the click.
        shift_held: bool,
    },
    /// Clicked on background — clear selection.
    ClearSelection,
}

/// Tracks mouse position, drag state, and the multi-click state machine.
pub struct InputState {
    /// Current cursor position in screen coordinates.
    pub mouse_pos: (f32, f32),
    /// Target under cursor at mouse-down.
    pub mouse_down_target: PickTarget,
    /// Whether a drag is in progress.
    pub is_dragging: bool,
    /// Last cursor position for computing deltas.
    last_cursor_pos: Option<(f32, f32)>,
    last_click_time: Instant,
    last_click_target: PickTarget,
    click_count: u32,
}

impl InputState {
    /// Create a new input state with no active click.
    pub fn new() -> Self {
        Self {
            mouse_pos: (0.0, 0.0),
            mouse_down_target: PickTarget::None,
            is_dragging: false,
            last_cursor_pos: None,
            last_click_time: Instant::now(),
            last_click_target: PickTarget::None,
            click_count: 0,
        }
    }

    /// Record what target (if any) is under the cursor at mouse-down.
    pub fn handle_mouse_down(&mut self, target: PickTarget) {
        self.mouse_down_target = target;
        self.is_dragging = false;
    }

    /// Mark that a drag occurred (significant mouse movement while pressed).
    pub fn mark_dragging(&mut self) {
        self.is_dragging = true;
    }

    /// Update cursor position for hover/picking and return the delta from the
    /// previous position.
    pub fn handle_mouse_position(&mut self, x: f32, y: f32) -> (f32, f32) {
        let delta = if let Some((lx, ly)) = self.last_cursor_pos {
            (x - lx, y - ly)
        } else {
            (0.0, 0.0)
        };
        self.last_cursor_pos = Some((x, y));
        self.mouse_pos = (x, y);
        delta
    }

    /// Process a mouse-up event and return what kind of click happened.
    ///
    /// `target` is the pick target under cursor at release time.
    /// `shift_held` is the current shift key state.
    pub fn process_mouse_up(
        &mut self,
        target: PickTarget,
        shift_held: bool,
    ) -> ClickResult {
        let mouse_up_target = target;
        let mouse_down_target = self.mouse_down_target;
        let now = Instant::now();

        // Reset state
        self.mouse_down_target = PickTarget::None;
        let was_dragging = self.is_dragging;
        self.is_dragging = false;

        // If we were dragging, don't do selection
        if was_dragging {
            self.last_click_time = now;
            self.last_click_target = PickTarget::None;
            self.click_count = 0;
            return ClickResult::NoAction;
        }

        // Click on a target — same target on down and up
        if !mouse_down_target.is_none()
            && mouse_down_target == mouse_up_target
        {
            // Check for multi-click on the same target
            if now.duration_since(self.last_click_time) < DOUBLE_CLICK_THRESHOLD
                && self.last_click_target == mouse_up_target
            {
                self.click_count = (self.click_count + 1).min(3);
            } else {
                self.click_count = 1;
            }

            self.last_click_time = now;
            self.last_click_target = mouse_up_target;

            match self.click_count {
                3 => ClickResult::TripleClick {
                    target: mouse_up_target,
                    shift_held,
                },
                2 => ClickResult::DoubleClick {
                    target: mouse_up_target,
                    shift_held,
                },
                _ => ClickResult::SingleClick {
                    target: mouse_up_target,
                    shift_held,
                },
            }
        } else if mouse_down_target.is_none() && mouse_up_target.is_none() {
            // Clicked on background
            self.last_click_time = now;
            self.last_click_target = PickTarget::None;
            self.click_count = 0;
            ClickResult::ClearSelection
        } else {
            // Mouse down and up on different things — no action
            self.last_click_time = now;
            self.last_click_target = PickTarget::None;
            self.click_count = 0;
            ClickResult::NoAction
        }
    }
}
