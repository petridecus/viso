use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderLanguage, ShaderType,
};
use std::borrow::Cow;

/// Wraps `naga_oil::compose::Composer` to provide shader composition with `#import` support.
///
/// Pre-loads all shared WGSL modules at construction time. Consuming shaders use
/// `#import viso::module_name` to pull in shared code. The composer produces
/// `naga::Module` IR directly, skipping WGSL re-parse at runtime.
pub struct ShaderComposer {
    composer: Composer,
}

/// Shared module definition: (source, file_path, import_path)
struct ModuleDef {
    source: &'static str,
    file_path: &'static str,
}

impl Default for ShaderComposer {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderComposer {
    pub fn new() -> Self {
        let mut composer = Composer::default();

        // Register shared modules in dependency order.
        // Modules with no dependencies first, then modules that depend on earlier ones.
        let modules: &[ModuleDef] = &[
            ModuleDef {
                source: include_str!("../../assets/shaders/modules/fullscreen.wgsl"),
                file_path: "modules/fullscreen.wgsl",
            },
            ModuleDef {
                source: include_str!("../../assets/shaders/modules/camera.wgsl"),
                file_path: "modules/camera.wgsl",
            },
            ModuleDef {
                source: include_str!("../../assets/shaders/modules/lighting.wgsl"),
                file_path: "modules/lighting.wgsl",
            },
            ModuleDef {
                source: include_str!("../../assets/shaders/modules/sdf.wgsl"),
                file_path: "modules/sdf.wgsl",
            },
            ModuleDef {
                source: include_str!("../../assets/shaders/modules/raymarch.wgsl"),
                file_path: "modules/raymarch.wgsl",
            },
            ModuleDef {
                source: include_str!("../../assets/shaders/modules/volume.wgsl"),
                file_path: "modules/volume.wgsl",
            },
        ];

        for m in modules {
            composer
                .add_composable_module(ComposableModuleDescriptor {
                    source: m.source,
                    file_path: m.file_path,
                    language: ShaderLanguage::Wgsl,
                    ..Default::default()
                })
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to register shader module '{}': {:?}",
                        m.file_path, e
                    )
                });
        }

        Self { composer }
    }

    /// Compose a shader source string (which may contain `#import` directives)
    /// into a `wgpu::ShaderModule` ready for pipeline creation.
    pub fn compose(
        &mut self,
        device: &wgpu::Device,
        label: &str,
        source: &str,
        file_path: &str,
    ) -> wgpu::ShaderModule {
        let naga_module = self
            .composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path,
                shader_type: ShaderType::Wgsl,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("Failed to compose shader '{}': {}", file_path, e));

        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Naga(Cow::Owned(naga_module)),
        })
    }

    /// Compose a shader source into a `naga::Module` without creating a wgpu shader module.
    /// Useful for testing shader composition without a GPU device.
    pub fn compose_naga(
        &mut self,
        source: &str,
        file_path: &str,
    ) -> Result<naga::Module, Box<naga_oil::compose::ComposerError>> {
        self.composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path,
                shader_type: ShaderType::Wgsl,
                ..Default::default()
            })
            .map_err(Box::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shader source definitions for all composable shaders in the project.
    /// Each entry is (source, file_path).
    fn all_shader_sources() -> Vec<(&'static str, &'static str)> {
        vec![
            (
                include_str!("../../assets/shaders/screen/bloom_threshold.wgsl"),
                "bloom_threshold.wgsl",
            ),
            (
                include_str!("../../assets/shaders/screen/bloom_blur.wgsl"),
                "bloom_blur.wgsl",
            ),
            (
                include_str!("../../assets/shaders/screen/bloom_upsample.wgsl"),
                "bloom_upsample.wgsl",
            ),
            (
                include_str!("../../assets/shaders/screen/fxaa.wgsl"),
                "fxaa.wgsl",
            ),
            (
                include_str!("../../assets/shaders/screen/composite.wgsl"),
                "composite.wgsl",
            ),
            (
                include_str!("../../assets/shaders/screen/ssao.wgsl"),
                "ssao.wgsl",
            ),
            (
                include_str!("../../assets/shaders/screen/ssao_blur.wgsl"),
                "ssao_blur.wgsl",
            ),
            (
                include_str!("../../assets/shaders/raster/mesh/backbone_tube.wgsl"),
                "backbone_tube.wgsl",
            ),
            (
                include_str!("../../assets/shaders/raster/impostor/capsule.wgsl"),
                "capsule_impostor.wgsl",
            ),
            (
                include_str!("../../assets/shaders/raster/impostor/sphere.wgsl"),
                "sphere_impostor.wgsl",
            ),
            (
                include_str!("../../assets/shaders/raster/impostor/cone.wgsl"),
                "cone_impostor.wgsl",
            ),
            (
                include_str!("../../assets/shaders/utility/picking_mesh.wgsl"),
                "picking.wgsl",
            ),
            (
                include_str!("../../assets/shaders/utility/picking_capsule.wgsl"),
                "picking_capsule.wgsl",
            ),
        ]
    }

    #[test]
    fn test_all_shaders_compose() {
        let mut composer = ShaderComposer::new();
        for (source, file_path) in all_shader_sources() {
            composer
                .compose_naga(source, file_path)
                .unwrap_or_else(|e| panic!("Shader '{}' failed to compose: {}", file_path, e));
        }
    }
}
