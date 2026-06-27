// cull.wgsl — GPU frustum culling with indirect draw compaction.
//
// Compute stack mapping:
//   #5  Comparison/Predicate (L₂max) — plane-sphere distance tests
//   #40 Scatter/Gather (L₀)          — compaction of visible instances
//   D₄  Embarrassingly parallel      — each instance independent
//
// For each source instance:
//   1. Test bounding sphere against 6 frustum planes
//   2. If visible, atomicAdd(indirect.instance_count, 1) → out_idx
//   3. Copy source[idx] → visible[out_idx]
// The render pass then uses draw_indexed_indirect with the populated
// indirect buffer — no CPU involvement between cull and draw.

struct CullUniform {
    frustum_planes: array<vec4<f32>, 6>,
    source_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct DrawIndirect {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: u32,
    first_instance: u32,
};

struct InstanceData {
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    m3: vec4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: CullUniform;
@group(0) @binding(1) var<storage, read> source: array<InstanceData>;
@group(0) @binding(2) var<storage, read> bounds: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> visible: array<InstanceData>;
@group(0) @binding(4) var<storage, read_write> indirect: DrawIndirect;

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.source_count {
        return;
    }

    let b = bounds[idx];
    let center = b.xyz;
    let radius = b.w;

    // Frustum sphere test: for each plane, signed distance must be >= -radius.
    for (var i = 0u; i < 6u; i++) {
        let plane = params.frustum_planes[i];
        let dist = dot(plane.xyz, center) + plane.w;
        if dist < -radius {
            return;
        }
    }

    // Visible — atomic compaction into the visible buffer.
    let out_idx = atomicAdd(&indirect.instance_count, 1u);
    visible[out_idx] = source[idx];
}
