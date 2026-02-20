use std::fmt;

/// Errors that can occur during GPU context initialization.
#[derive(Debug)]
pub enum RenderContextError {
    SurfaceCreation(wgpu::CreateSurfaceError),
    AdapterRequest(wgpu::RequestAdapterError),
    DeviceRequest(wgpu::RequestDeviceError),
    UnsupportedSurface,
}

impl fmt::Display for RenderContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SurfaceCreation(e) => write!(f, "surface creation failed: {e}"),
            Self::AdapterRequest(e) => {
                write!(f, "no compatible GPU adapter found: {e}")
            }
            Self::DeviceRequest(e) => write!(f, "device request failed: {e}"),
            Self::UnsupportedSurface => {
                write!(f, "surface configuration not supported by adapter")
            }
        }
    }
}

impl std::error::Error for RenderContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SurfaceCreation(e) => Some(e),
            Self::AdapterRequest(e) => Some(e),
            Self::DeviceRequest(e) => Some(e),
            _ => None,
        }
    }
}

/// Owns the core wgpu resources: device, queue, surface, and configuration.
pub struct RenderContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    /// Supersampling scale factor (1 = native, 2 = 2x SSAA).
    pub render_scale: u32,
}

impl RenderContext {
    /// Create a new render context from the given window surface target and initial size.
    pub async fn new(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        initial_size: (u32, u32),
    ) -> Result<Self, RenderContextError> {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .map_err(RenderContextError::SurfaceCreation)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(RenderContextError::AdapterRequest)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Primary Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .map_err(RenderContextError::DeviceRequest)?;

        let mut config = surface
            .get_default_config(&adapter, initial_size.0, initial_size.1)
            .ok_or(RenderContextError::UnsupportedSurface)?;
        config.width = initial_size.0;
        config.height = initial_size.1;
        config.present_mode = wgpu::PresentMode::Fifo;

        surface.configure(&device, &config);

        Ok(Self {
            device,
            queue,
            surface,
            config,
            render_scale: 1,
        })
    }

    /// Internal render width (swapchain width * render_scale).
    pub fn render_width(&self) -> u32 {
        self.config.width * self.render_scale
    }

    /// Internal render height (swapchain height * render_scale).
    pub fn render_height(&self) -> u32 {
        self.config.height * self.render_scale
    }

    /// Reconfigure the surface for the new window size. Ignores zero-sized dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Set the surface DPI scale factor (currently a no-op placeholder).
    pub fn set_surface_scale(&self, _scale: f64) {}

    /// Acquire the next swapchain texture for rendering.
    pub fn get_next_frame(
        &self,
    ) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }

    /// Create a new command encoder for recording GPU commands.
    pub fn create_encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            })
    }

    /// Finish the encoder and submit its command buffer to the GPU queue.
    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(std::iter::once(encoder.finish()));
    }
}
