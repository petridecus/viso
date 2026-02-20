use std::time::{Duration, Instant};

const DOUBLE_CLICK_THRESHOLD: Duration = Duration::from_millis(400);

/// Result of processing a mouse-up event through the multi-click state machine.
pub enum ClickResult {
    /// No selection action (drag, mismatched up/down, etc.)
    NoAction,
    /// Single click on a residue.
    SingleClick { shift_held: bool },
    /// Double-click on a residue — select secondary structure segment.
    DoubleClick { residue: i32, shift_held: bool },
    /// Triple-click on a residue — select entire chain.
    TripleClick { residue: i32, shift_held: bool },
    /// Clicked on background — clear selection.
    ClearSelection,
}

/// Tracks mouse position, drag state, and the multi-click state machine.
pub struct InputState {
    pub mouse_pos: (f32, f32),
    pub mouse_down_residue: i32,
    pub is_dragging: bool,
    last_click_time: Instant,
    last_click_residue: i32,
    click_count: u32,
}

impl InputState {
    /// Create a new input state with no active click.
    pub fn new() -> Self {
        Self {
            mouse_pos: (0.0, 0.0),
            mouse_down_residue: -1,
            is_dragging: false,
            last_click_time: Instant::now(),
            last_click_residue: -1,
            click_count: 0,
        }
    }

    /// Record what residue (if any) is under the cursor at mouse-down.
    pub fn handle_mouse_down(&mut self, hovered_residue: i32) {
        self.mouse_down_residue = hovered_residue;
        self.is_dragging = false;
    }

    /// Mark that a drag occurred (significant mouse movement while pressed).
    pub fn mark_dragging(&mut self) {
        self.is_dragging = true;
    }

    /// Update cursor position for hover/picking.
    pub fn handle_mouse_position(&mut self, x: f32, y: f32) {
        self.mouse_pos = (x, y);
    }

    /// Process a mouse-up event and return what kind of click happened.
    ///
    /// `hovered_residue` is the residue under cursor at release time.
    /// `shift_held` is the current shift key state.
    pub fn process_mouse_up(
        &mut self,
        hovered_residue: i32,
        shift_held: bool,
    ) -> ClickResult {
        let mouse_up_residue = hovered_residue;
        let mouse_down_residue = self.mouse_down_residue;
        let now = Instant::now();

        // Reset state
        self.mouse_down_residue = -1;
        let was_dragging = self.is_dragging;
        self.is_dragging = false;

        // If we were dragging, don't do selection
        if was_dragging {
            self.last_click_time = now;
            self.last_click_residue = -1;
            self.click_count = 0;
            return ClickResult::NoAction;
        }

        // Click on a residue — same residue on down and up
        if mouse_down_residue >= 0 && mouse_down_residue == mouse_up_residue {
            // Check for multi-click on the same residue
            if now.duration_since(self.last_click_time) < DOUBLE_CLICK_THRESHOLD
                && self.last_click_residue == mouse_up_residue
            {
                self.click_count = (self.click_count + 1).min(3);
            } else {
                self.click_count = 1;
            }

            self.last_click_time = now;
            self.last_click_residue = mouse_up_residue;

            match self.click_count {
                3 => ClickResult::TripleClick {
                    residue: mouse_up_residue,
                    shift_held,
                },
                2 => ClickResult::DoubleClick {
                    residue: mouse_up_residue,
                    shift_held,
                },
                _ => ClickResult::SingleClick { shift_held },
            }
        } else if mouse_down_residue < 0 && mouse_up_residue < 0 {
            // Clicked on background
            self.last_click_time = now;
            self.last_click_residue = -1;
            self.click_count = 0;
            ClickResult::ClearSelection
        } else {
            // Mouse down and up on different things — no action
            self.last_click_time = now;
            self.last_click_residue = -1;
            self.click_count = 0;
            ClickResult::NoAction
        }
    }
}
