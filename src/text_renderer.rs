use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer as GlyphonRenderer, Viewport,
};
use wgpu::MultisampleState;

use crate::render_context::RenderContext;

pub struct TextRenderer {
    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    text_renderer: GlyphonRenderer,
    viewport: Viewport,
    buffer: Buffer,
}

impl TextRenderer {
    pub fn new(context: &RenderContext) -> Self {
        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(&context.device);
        let mut atlas = TextAtlas::new(&context.device, &context.queue, &cache, context.config.format);
        let text_renderer = GlyphonRenderer::new(
            &mut atlas,
            &context.device,
            MultisampleState::default(),
            None,
        );

        let viewport = Viewport::new(&context.device, &cache);

        // Create a buffer for FPS text (larger font for visibility)
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(32.0, 40.0));
        buffer.set_size(
            &mut font_system,
            Some(300.0),
            Some(50.0),
        );

        Self {
            font_system,
            swash_cache,
            atlas,
            text_renderer,
            viewport,
            buffer,
        }
    }

    pub fn update_fps(&mut self, fps: f32) {
        let text = format!("FPS: {:.0}", fps);
        let attrs = Attrs::new().family(Family::Monospace).color(Color::rgb(200, 200, 200));
        self.buffer.set_text(
            &mut self.font_system,
            &text,
            &attrs,
            Shaping::Basic,
            None, // No alignment override
        );
        self.buffer.shape_until_scroll(&mut self.font_system, false);
    }

    pub fn resize(&mut self, _width: u32, _height: u32) {
        self.buffer.set_size(
            &mut self.font_system,
            Some(300.0),
            Some(50.0),
        );
    }

    pub fn prepare(&mut self, context: &RenderContext) {
        log::debug!(
            "text_renderer.prepare: viewport={}x{}",
            context.config.width, context.config.height,
        );
        self.viewport.update(
            &context.queue,
            Resolution {
                width: context.config.width,
                height: context.config.height,
            },
        );

        self.text_renderer
            .prepare(
                &context.device,
                &context.queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                [TextArea {
                    buffer: &self.buffer,
                    left: 10.0,
                    top: 10.0,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: context.config.width as i32,
                        bottom: context.config.height as i32,
                    },
                    default_color: Color::rgb(200, 200, 200),
                    custom_glyphs: &[],
                }],
                &mut self.swash_cache,
            )
            .unwrap();
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.text_renderer
            .render(&self.atlas, &self.viewport, render_pass)
            .unwrap();
    }
}
