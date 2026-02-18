pub struct RenderContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    /// Supersampling scale factor (1 = native, 2 = 2x SSAA).
    pub render_scale: u32,
}

impl RenderContext {
    pub async fn new(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        initial_size: (u32, u32),
    ) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Primary Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .unwrap();

        let mut config = surface
            .get_default_config(&adapter, initial_size.0, initial_size.1)
            .unwrap();
        config.width = initial_size.0;
        config.height = initial_size.1;
        config.present_mode = wgpu::PresentMode::Fifo;

        surface.configure(&device, &config);

        Self {
            device,
            queue,
            surface,
            config,
            render_scale: 1,
        }
    }

    /// Internal render width (swapchain width * render_scale).
    pub fn render_width(&self) -> u32 {
        self.config.width * self.render_scale
    }

    /// Internal render height (swapchain height * render_scale).
    pub fn render_height(&self) -> u32 {
        self.config.height * self.render_scale
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn set_surface_scale(&self, _scale: f64) {}

    pub fn get_next_frame(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }

    pub fn create_encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            })
    }

    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(std::iter::once(encoder.finish()));
    }
}
