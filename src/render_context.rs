/// Round down to nearest multiple of 8 to prevent GPU stride errors.
fn align_down(n: u32) -> u32 {
    n & !7
}

pub struct RenderContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
}

impl RenderContext {
    /// Create a new render context from any window-like object.
    ///
    /// The `window` must implement `Into<wgpu::SurfaceTarget<'static>>` — this is
    /// satisfied by `Arc<winit::window::Window>` (standalone) and Tauri's `WebviewWindow`.
    /// `initial_size` is `(width, height)` in physical pixels.
    pub async fn new(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        initial_size: (u32, u32),
    ) -> Self {
        eprintln!("[RenderContext::new] initial_size={}x{}", initial_size.0, initial_size.1);
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
        eprintln!("[RenderContext::new] adapter: {:?}, backend: {:?}", adapter.get_info().name, adapter.get_info().backend);

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
        config.width = align_down(initial_size.0);
        config.height = align_down(initial_size.1);

        // Use Fifo (vsync) — widely supported across all backends
        config.present_mode = wgpu::PresentMode::Fifo;

        eprintln!("[RenderContext::new] surface config: {}x{} format={:?} present_mode={:?}",
            config.width, config.height, config.format, config.present_mode);

        surface.configure(&device, &config);

        Self {
            device,
            queue,
            surface,
            config,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let w = align_down(width);
        let h = align_down(height);
        if w > 0 && h > 0 {
            eprintln!("[RenderContext::resize] {}x{} -> {}x{}", self.config.width, self.config.height, w, h);
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
    }

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
