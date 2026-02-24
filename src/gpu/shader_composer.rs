use std::{borrow::Cow, collections::HashMap};

use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderLanguage,
    ShaderType,
};

/// Wraps `naga_oil::compose::Composer` to provide shader composition with
/// `#import` support.
///
/// All WGSL shader sources are loaded at construction time from `src/shaders/`.
/// Consuming code references shaders by their path relative to that directory
/// (e.g. `"screen/fxaa.wgsl"`), eliminating duplicated file-system paths
/// across the codebase.
pub struct ShaderComposer {
    composer: Composer,
    sources: HashMap<&'static str, &'static str>,
}

/// Paths of shared modules that are registered with naga_oil for `#import`
/// support. Order matters — modules with no dependencies first.
const MODULE_PATHS: &[&str] = &[
    "modules/fullscreen.wgsl",
    "modules/camera.wgsl",
    "modules/lighting.wgsl",
    "modules/sdf.wgsl",
    "modules/raymarch.wgsl",
    "modules/volume.wgsl",
];

impl Default for ShaderComposer {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderComposer {
    /// Create a new composer with all shader sources loaded and shared modules
    /// registered.
    pub fn new() -> Self {
        let sources: HashMap<&'static str, &'static str> = HashMap::from([
            // Shared modules
            (
                "modules/fullscreen.wgsl",
                include_str!("../shaders/modules/fullscreen.wgsl"),
            ),
            (
                "modules/camera.wgsl",
                include_str!("../shaders/modules/camera.wgsl"),
            ),
            (
                "modules/lighting.wgsl",
                include_str!("../shaders/modules/lighting.wgsl"),
            ),
            (
                "modules/sdf.wgsl",
                include_str!("../shaders/modules/sdf.wgsl"),
            ),
            (
                "modules/raymarch.wgsl",
                include_str!("../shaders/modules/raymarch.wgsl"),
            ),
            (
                "modules/volume.wgsl",
                include_str!("../shaders/modules/volume.wgsl"),
            ),
            // Screen-space / post-processing
            (
                "screen/bloom_threshold.wgsl",
                include_str!("../shaders/screen/bloom_threshold.wgsl"),
            ),
            (
                "screen/bloom_blur.wgsl",
                include_str!("../shaders/screen/bloom_blur.wgsl"),
            ),
            (
                "screen/bloom_upsample.wgsl",
                include_str!("../shaders/screen/bloom_upsample.wgsl"),
            ),
            (
                "screen/fxaa.wgsl",
                include_str!("../shaders/screen/fxaa.wgsl"),
            ),
            (
                "screen/composite.wgsl",
                include_str!("../shaders/screen/composite.wgsl"),
            ),
            (
                "screen/ssao.wgsl",
                include_str!("../shaders/screen/ssao.wgsl"),
            ),
            (
                "screen/ssao_blur.wgsl",
                include_str!("../shaders/screen/ssao_blur.wgsl"),
            ),
            // Raster geometry
            (
                "raster/mesh/backbone_tube.wgsl",
                include_str!("../shaders/raster/mesh/backbone_tube.wgsl"),
            ),
            (
                "raster/mesh/backbone_na.wgsl",
                include_str!("../shaders/raster/mesh/backbone_na.wgsl"),
            ),
            (
                "raster/impostor/capsule.wgsl",
                include_str!("../shaders/raster/impostor/capsule.wgsl"),
            ),
            (
                "raster/impostor/sphere.wgsl",
                include_str!("../shaders/raster/impostor/sphere.wgsl"),
            ),
            (
                "raster/impostor/cone.wgsl",
                include_str!("../shaders/raster/impostor/cone.wgsl"),
            ),
            (
                "raster/impostor/polygon.wgsl",
                include_str!("../shaders/raster/impostor/polygon.wgsl"),
            ),
            // Utility
            (
                "utility/picking_mesh.wgsl",
                include_str!("../shaders/utility/picking_mesh.wgsl"),
            ),
            (
                "utility/picking_capsule.wgsl",
                include_str!("../shaders/utility/picking_capsule.wgsl"),
            ),
        ]);

        let mut composer = Composer::default();

        // Register shared modules in dependency order.
        for path in MODULE_PATHS {
            let source = sources[path];
            let _ = composer
                .add_composable_module(ComposableModuleDescriptor {
                    source,
                    file_path: path,
                    language: ShaderLanguage::Wgsl,
                    ..Default::default()
                })
                .unwrap_or_else(|e| {
                    panic!("Failed to register shader module '{path}': {e:?}")
                });
        }

        Self { composer, sources }
    }

    /// Compose a shader into a `wgpu::ShaderModule` ready for pipeline
    /// creation.
    ///
    /// `path` is relative to the shaders directory, e.g.
    /// `"screen/fxaa.wgsl"`.
    pub fn compose(
        &mut self,
        device: &wgpu::Device,
        label: &str,
        path: &str,
    ) -> wgpu::ShaderModule {
        let source = self
            .sources
            .get(path)
            .unwrap_or_else(|| panic!("Unknown shader path: {path}"));

        let naga_module = self
            .composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path: path,
                shader_type: ShaderType::Wgsl,
                ..Default::default()
            })
            .unwrap_or_else(|e| {
                panic!("Failed to compose shader '{path}': {e}")
            });

        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Naga(Cow::Owned(naga_module)),
        })
    }

    /// Compose a shader source into a `naga::Module` without creating a wgpu
    /// shader module. Useful for testing shader composition without a GPU
    /// device.
    pub fn compose_naga(
        &mut self,
        path: &str,
    ) -> Result<naga::Module, Box<naga_oil::compose::ComposerError>> {
        let source = self
            .sources
            .get(path)
            .unwrap_or_else(|| panic!("Unknown shader path: {path}"));

        self.composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path: path,
                shader_type: ShaderType::Wgsl,
                ..Default::default()
            })
            .map_err(Box::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All composable (non-module) shader paths.
    fn composable_shader_paths() -> Vec<&'static str> {
        vec![
            "screen/bloom_threshold.wgsl",
            "screen/bloom_blur.wgsl",
            "screen/bloom_upsample.wgsl",
            "screen/fxaa.wgsl",
            "screen/composite.wgsl",
            "screen/ssao.wgsl",
            "screen/ssao_blur.wgsl",
            "raster/mesh/backbone_tube.wgsl",
            "raster/impostor/capsule.wgsl",
            "raster/impostor/sphere.wgsl",
            "raster/impostor/cone.wgsl",
            "utility/picking_mesh.wgsl",
            "utility/picking_capsule.wgsl",
        ]
    }

    #[test]
    fn test_all_shaders_compose() {
        let mut composer = ShaderComposer::new();
        for path in composable_shader_paths() {
            let _ = composer.compose_naga(path).unwrap_or_else(|e| {
                panic!("Shader '{path}' failed to compose: {e}")
            });
        }
    }
}
