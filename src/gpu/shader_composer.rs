use std::borrow::Cow;

use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderLanguage,
    ShaderType,
};

use crate::error::VisoError;

/// All composable (non-module) shaders in the engine.
#[derive(Debug, Clone, Copy)]
pub enum Shader {
    BloomThreshold,
    BloomBlur,
    BloomUpsample,
    Fxaa,
    Composite,
    Ssao,
    SsaoBlur,
    BackboneTube,
    Capsule,
    Sphere,
    Cone,
    Polygon,
    PickingMesh,
    PickingCapsule,
    PickingSphere,
}

impl Shader {
    /// Pipeline label for wgpu debug markers.
    fn label(self) -> &'static str {
        match self {
            Self::BloomThreshold => "Bloom Threshold Shader",
            Self::BloomBlur => "Bloom Blur Shader",
            Self::BloomUpsample => "Bloom Upsample Shader",
            Self::Fxaa => "FXAA Shader",
            Self::Composite => "Composite Shader",
            Self::Ssao => "SSAO Shader",
            Self::SsaoBlur => "SSAO Blur Shader",
            Self::BackboneTube => "Backbone Tube",
            Self::Capsule => "Capsule Impostor",
            Self::Sphere => "Sphere Impostor",
            Self::Cone => "Cone Impostor",
            Self::Polygon => "Polygon Impostor",
            Self::PickingMesh => "Picking Tube Shader",
            Self::PickingCapsule => "Picking Capsule Shader",
            Self::PickingSphere => "Picking Sphere Shader",
        }
    }

    /// Path relative to the shaders directory (internal).
    fn path(self) -> &'static str {
        match self {
            Self::BloomThreshold => "screen/bloom_threshold.wgsl",
            Self::BloomBlur => "screen/bloom_blur.wgsl",
            Self::BloomUpsample => "screen/bloom_upsample.wgsl",
            Self::Fxaa => "screen/fxaa.wgsl",
            Self::Composite => "screen/composite.wgsl",
            Self::Ssao => "screen/ssao.wgsl",
            Self::SsaoBlur => "screen/ssao_blur.wgsl",
            Self::BackboneTube => "raster/mesh/backbone_tube.wgsl",
            Self::Capsule => "raster/impostor/capsule.wgsl",
            Self::Sphere => "raster/impostor/sphere.wgsl",
            Self::Cone => "raster/impostor/cone.wgsl",
            Self::Polygon => "raster/impostor/polygon.wgsl",
            Self::PickingMesh => "utility/picking_mesh.wgsl",
            Self::PickingCapsule => "utility/picking_capsule.wgsl",
            Self::PickingSphere => "utility/picking_sphere.wgsl",
        }
    }

    /// Source code via `include_str!`.
    fn source(self) -> &'static str {
        match self {
            Self::BloomThreshold => {
                include_str!("../shaders/screen/bloom_threshold.wgsl")
            }
            Self::BloomBlur => {
                include_str!("../shaders/screen/bloom_blur.wgsl")
            }
            Self::BloomUpsample => {
                include_str!("../shaders/screen/bloom_upsample.wgsl")
            }
            Self::Fxaa => include_str!("../shaders/screen/fxaa.wgsl"),
            Self::Composite => {
                include_str!("../shaders/screen/composite.wgsl")
            }
            Self::Ssao => include_str!("../shaders/screen/ssao.wgsl"),
            Self::SsaoBlur => {
                include_str!("../shaders/screen/ssao_blur.wgsl")
            }
            Self::BackboneTube => {
                include_str!("../shaders/raster/mesh/backbone_tube.wgsl")
            }
            Self::Capsule => {
                include_str!("../shaders/raster/impostor/capsule.wgsl")
            }
            Self::Sphere => {
                include_str!("../shaders/raster/impostor/sphere.wgsl")
            }
            Self::Cone => {
                include_str!("../shaders/raster/impostor/cone.wgsl")
            }
            Self::Polygon => {
                include_str!("../shaders/raster/impostor/polygon.wgsl")
            }
            Self::PickingMesh => {
                include_str!("../shaders/utility/picking_mesh.wgsl")
            }
            Self::PickingCapsule => {
                include_str!("../shaders/utility/picking_capsule.wgsl")
            }
            Self::PickingSphere => {
                include_str!("../shaders/utility/picking_sphere.wgsl")
            }
        }
    }
}

/// Paths of shared modules that are registered with naga_oil for `#import`
/// support. Order matters — modules with no dependencies first.
const MODULE_PATHS: &[(&str, &str)] = &[
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
];

/// Wraps `naga_oil::compose::Composer` to provide shader composition with
/// `#import` support.
///
/// All WGSL shader sources are compiled in at build time via `include_str!`.
/// Consuming code references shaders by their [`Shader`] enum variant,
/// eliminating fragile string paths across the codebase.
pub struct ShaderComposer {
    composer: Composer,
}

impl ShaderComposer {
    /// Create a new composer with shared modules registered.
    pub fn new() -> Result<Self, VisoError> {
        let mut composer = Composer::default();

        for (path, source) in MODULE_PATHS {
            let _ = composer
                .add_composable_module(ComposableModuleDescriptor {
                    source,
                    file_path: path,
                    language: ShaderLanguage::Wgsl,
                    ..Default::default()
                })
                .map_err(|e| {
                    VisoError::Shader(format!(
                        "failed to register shader module '{path}': {e:?}"
                    ))
                })?;
        }

        Ok(Self { composer })
    }

    /// Compose a shader into a `wgpu::ShaderModule` ready for pipeline
    /// creation.
    pub fn compose(
        &mut self,
        device: &wgpu::Device,
        shader: Shader,
    ) -> Result<wgpu::ShaderModule, VisoError> {
        let label = shader.label();
        let path = shader.path();
        let source = shader.source();

        let naga_module = self
            .composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path: path,
                shader_type: ShaderType::Wgsl,
                ..Default::default()
            })
            .map_err(|e| {
                VisoError::Shader(format!(
                    "failed to compose shader '{path}': {e}"
                ))
            })?;

        Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Naga(Cow::Owned(naga_module)),
        }))
    }

    /// Compose a shader source into a `naga::Module` without creating a wgpu
    /// shader module. Useful for testing shader composition without a GPU
    /// device.
    #[cfg(test)]
    pub fn compose_naga(
        &mut self,
        shader: Shader,
    ) -> Result<naga::Module, VisoError> {
        let path = shader.path();
        let source = shader.source();

        self.composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path: path,
                shader_type: ShaderType::Wgsl,
                ..Default::default()
            })
            .map_err(|e| {
                VisoError::Shader(format!(
                    "failed to compose shader '{path}': {e}"
                ))
            })
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// All composable (non-module) shaders.
    fn composable_shaders() -> Vec<Shader> {
        vec![
            Shader::BloomThreshold,
            Shader::BloomBlur,
            Shader::BloomUpsample,
            Shader::Fxaa,
            Shader::Composite,
            Shader::Ssao,
            Shader::SsaoBlur,
            Shader::BackboneTube,
            Shader::Capsule,
            Shader::Sphere,
            Shader::Cone,
            Shader::PickingMesh,
            Shader::PickingCapsule,
            Shader::PickingSphere,
        ]
    }

    #[test]
    fn test_all_shaders_compose() {
        let mut composer = ShaderComposer::new().expect("ShaderComposer::new");
        for shader in composable_shaders() {
            let _ = composer.compose_naga(shader).unwrap_or_else(|e| {
                panic!("Shader '{shader:?}' failed to compose: {e}")
            });
        }
    }
}
