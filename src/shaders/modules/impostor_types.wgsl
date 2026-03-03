// Shared instance struct definitions for impostor shaders.
//
// Used by both render and picking variants to keep struct layouts in sync.

#define_import_path viso::impostor_types

struct SphereInstance {
    center: vec4<f32>,   // xyz=position, w=radius
    color: vec4<f32>,    // xyz=RGB, w=entity_id (packed as float)
};

struct CapsuleInstance {
    endpoint_a: vec4<f32>,  // xyz=position, w=radius
    endpoint_b: vec4<f32>,  // xyz=position, w=residue_idx (packed as float)
    color_a: vec4<f32>,     // xyz=RGB at endpoint A, w=unused
    color_b: vec4<f32>,     // xyz=RGB at endpoint B, w=unused
};
