// MMAST simulator — instanced render shader.
//
// Compute stack mapping:
//   Primitive #40 Scatter/Gather (L₀) — per-instance model matrix
//   Primitive #41 Embedding Lookup (L₀) — per-instance color from buffer
//   Primitive #1 FMA (L₁) — matrix × vertex in hardware
//   Primitive #5 Comparison (L₂max) — depth test in fixed-function
//
// Total draw calls: 1 per object TYPE (not per object).
// N trees = 1 draw call. N rocks = 1 draw call.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    sun_dir: vec4<f32>,       // xyz = direction, w = intensity
    ambient: vec4<f32>,       // rgb = ambient color, a = intensity
    fog_color: vec4<f32>,     // rgb = fog color, a = density
    camera_pos: vec4<f32>,    // xyz = eye position
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct InstanceInput {
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(7) inst_color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(vert: VertexInput, inst: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(inst.model_0, inst.model_1, inst.model_2, inst.model_3);
    let world_pos = model * vec4<f32>(vert.position, 1.0);

    // Normal transform: upper-left 3x3 of model matrix (assumes uniform scale).
    let normal_mat = mat3x3<f32>(inst.model_0.xyz, inst.model_1.xyz, inst.model_2.xyz);
    let world_normal = normalize(normal_mat * vert.normal);

    // Instance color overrides vertex color if non-zero alpha; otherwise use vertex color.
    let color = select(vert.color, inst.inst_color, inst.inst_color.a > 0.01);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos;
    out.world_normal = world_normal;
    out.color = color;
    out.world_pos = world_pos.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let sun = normalize(camera.sun_dir.xyz);
    let sun_intensity = camera.sun_dir.w;

    // Diffuse lighting.
    let ndotl = max(dot(n, sun), 0.0);
    let diffuse = ndotl * sun_intensity;

    // Hemisphere ambient: lerp between ground color and sky color by normal.y.
    let sky_blend = n.y * 0.5 + 0.5;
    let ambient = camera.ambient.rgb * camera.ambient.a;
    let hemi = mix(ambient * 0.5, ambient, sky_blend);

    var lit = in.color.rgb * (hemi + diffuse);

    // Exponential fog.
    let dist = distance(camera.camera_pos.xyz, in.world_pos);
    let fog_factor = 1.0 - exp(-dist * camera.fog_color.a);
    lit = mix(lit, camera.fog_color.rgb, clamp(fog_factor, 0.0, 1.0));

    // Simple Reinhard tonemap.
    lit = lit / (lit + vec3<f32>(1.0));

    return vec4<f32>(lit, in.color.a);
}
