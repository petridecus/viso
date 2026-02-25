use web_time::{Duration, Instant};

/// Frame timing with FPS calculation and optional frame limiting
pub struct FrameTiming {
    /// Target FPS (0 = unlimited)
    target_fps: u32,
    /// Minimum frame duration based on target FPS
    min_frame_duration: Duration,
    /// Last frame timestamp
    last_frame: Instant,
    /// Smoothed FPS using exponential moving average
    smoothed_fps: f32,
    /// Smoothing factor (lower = smoother, 0.0-1.0)
    smoothing: f32,
}

impl FrameTiming {
    /// Create a new frame timer with the given FPS target (0 = unlimited).
    pub fn new(target_fps: u32) -> Self {
        let min_frame_duration = if target_fps > 0 {
            Duration::from_secs_f64(1.0 / target_fps as f64)
        } else {
            Duration::ZERO
        };

        Self {
            target_fps,
            min_frame_duration,
            last_frame: Instant::now(),
            smoothed_fps: 60.0, // Start with reasonable default
            smoothing: 0.05,    /* 5% new value, 95% old value for smooth
                                 * display */
        }
    }

    /// Call at the start of each frame. Returns true if enough time has passed
    /// to render.
    pub fn should_render(&self) -> bool {
        if self.target_fps == 0 {
            return true;
        }
        self.last_frame.elapsed() >= self.min_frame_duration
    }

    /// Call after rendering to update timing.
    pub fn end_frame(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame);
        self.last_frame = now;

        // Calculate instantaneous FPS
        let frame_time = elapsed.as_secs_f32();
        if frame_time > 0.0 {
            let instant_fps = 1.0 / frame_time;
            // Exponential moving average for smooth display
            self.smoothed_fps = self.smoothed_fps * (1.0 - self.smoothing)
                + instant_fps * self.smoothing;
        }
    }

    /// Get the current FPS (smoothed)
    pub fn fps(&self) -> f32 {
        self.smoothed_fps
    }
}
