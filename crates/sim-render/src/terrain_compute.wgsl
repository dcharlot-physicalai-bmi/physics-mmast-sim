// Terrain compute shader — GPU-side FBM noise generation.
//
// Compute stack mapping:
//   Primitive #23 FFT (B₂) — frequency-domain noise octaves
//   Primitive #24 Convolution (B₂) — FBM octave accumulation
//   Primitive #2 Bitwise Logic (L₀) — hash function for noise seed
//   Concurrency D₄ — embarrassingly parallel (each vertex independent)
//
// Replaces CPU sin/cos terrain generation with GPU-parallel FBM.
// One dispatch generates the entire terrain heightfield.

struct TerrainParams {
    grid_size: u32,        // vertices per side
    world_size: f32,       // meters
    seed: u32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
    amplitude: f32,
    frequency: f32,
};

@group(0) @binding(0) var<uniform> params: TerrainParams;
@group(0) @binding(1) var<storage, read_write> vertices: array<f32>;
// Layout: [px, py, pz, nx, ny, nz, cr, cg, cb, ca] × grid_size²

// Hash — Primitive #2 Bitwise Logic (L₀, thermodynamically free).
fn hash2d(x: f32, y: f32) -> f32 {
    var p = vec2<f32>(x, y);
    p = fract(p * vec2<f32>(443.8975, 397.2973));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

// Value noise with smooth interpolation.
fn noise2d(x: f32, y: f32) -> f32 {
    let ix = floor(x);
    let iy = floor(y);
    let fx = fract(x);
    let fy = fract(y);

    // Smooth step (cubic hermite) — avoids grid artifacts.
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);

    let a = hash2d(ix, iy);
    let b = hash2d(ix + 1.0, iy);
    let c = hash2d(ix, iy + 1.0);
    let d = hash2d(ix + 1.0, iy + 1.0);

    return mix(mix(a, b, ux), mix(c, d, ux), uy);
}

// FBM — Primitive #24 Convolution: accumulate octaves.
fn fbm(x: f32, y: f32) -> f32 {
    var value = 0.0;
    var amp = params.amplitude;
    var freq = params.frequency;
    let s = f32(params.seed) * 0.001;

    for (var i = 0u; i < params.octaves; i++) {
        value += amp * noise2d(x * freq + s, y * freq + s * 1.7);
        freq *= params.lacunarity;
        amp *= params.persistence;
    }
    return value;
}

@compute @workgroup_size(8, 8)
fn terrain_gen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.grid_size;
    if gid.x >= n || gid.y >= n {
        return;
    }

    let step = params.world_size / f32(n - 1u);
    let half = params.world_size / 2.0;
    let px = f32(gid.x) * step - half;
    let pz = f32(gid.y) * step - half;

    // Height from FBM.
    let py = fbm(px, pz);

    // Flatten a runway corridor at z ≈ 0, x ∈ [-45, 45].
    var height = py;
    let runway_blend_x = smoothstep(35.0, 50.0, abs(px));
    let runway_blend_z = smoothstep(6.0, 12.0, abs(pz));
    let runway_blend = max(runway_blend_x, runway_blend_z);
    height = mix(0.0, height, runway_blend);

    // Normal via finite differences (sample neighbors).
    let eps = step;
    let hx1 = fbm(px + eps, pz);
    let hx0 = fbm(px - eps, pz);
    let hz1 = fbm(px, pz + eps);
    let hz0 = fbm(px, pz - eps);
    let nx = (hx0 - hx1) / (2.0 * eps);
    let nz = (hz0 - hz1) / (2.0 * eps);
    let normal = normalize(vec3<f32>(nx, 1.0, nz));

    // Color from height — Primitive #41 Embedding Lookup (L₀).
    // Simple biome LUT: low = dark water, mid = green, high = grey rock.
    var color: vec3<f32>;
    if height < -0.3 {
        color = vec3<f32>(0.04, 0.08, 0.18); // deep water
    } else if height < 0.0 {
        color = vec3<f32>(0.06, 0.12, 0.15); // shallow
    } else if height < 0.8 {
        let t = height / 0.8;
        color = mix(vec3<f32>(0.08, 0.16, 0.08), vec3<f32>(0.12, 0.18, 0.06), t); // grass to dry
    } else {
        color = vec3<f32>(0.15, 0.14, 0.13); // rock
    }

    // Write to storage buffer (10 floats per vertex).
    let idx = (gid.y * n + gid.x) * 10u;
    vertices[idx + 0u] = px;
    vertices[idx + 1u] = height;
    vertices[idx + 2u] = pz;
    vertices[idx + 3u] = normal.x;
    vertices[idx + 4u] = normal.y;
    vertices[idx + 5u] = normal.z;
    vertices[idx + 6u] = color.x;
    vertices[idx + 7u] = color.y;
    vertices[idx + 8u] = color.z;
    vertices[idx + 9u] = 1.0; // alpha
}
