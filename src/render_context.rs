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

    /// Set the surface layer's scale factor for correct HiDPI compositing.
    ///
    /// On macOS the CAMetalLayer's `contentsScale` defaults to 1.0 — without
    /// updating it the compositor downsamples the full-res drawable, producing
    /// blurry output on Retina displays. On Vulkan/DX12 the surface dimensions
    /// from `window.inner_size()` are sufficient; this is a no-op there for now.
    pub fn set_surface_scale(&self, scale_factor: f64) {
        log::info!(
            "set_surface_scale called: scale_factor={}, surface config={}x{}",
            scale_factor, self.config.width, self.config.height,
        );
        #[cfg(target_os = "macos")]
        unsafe {
            if let Some(hal_surface) = self.surface.as_hal::<wgpu::hal::api::Metal>() {
                let layer = hal_surface.render_layer().lock();
                let old_scale = layer.contents_scale();
                layer.set_contents_scale(scale_factor);
                log::info!(
                    "Metal layer contentsScale: {} -> {}",
                    old_scale, scale_factor,
                );
            } else {
                log::warn!("as_hal::<Metal>() returned None — surface scale NOT set");
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = scale_factor;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            eprintln!("[RenderContext::resize] {}x{} -> {}x{}", self.config.width, self.config.height, width, height);
            self.config.width = width;
            self.config.height = height;
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
