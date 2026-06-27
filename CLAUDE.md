# MMAST Physics Simulator

Physics-informed, real-time multi-vehicle performance simulator for the MMAST
(Multi-Material Adaptive Surface Technology) energy and signature stack.

## Stack

- Rust 1.94.1, edition 2024
- openie-cad crates as dependencies (cad-kernel, cad-sim, cad-math, cad-types, cad-spice, cad-format)
- WebGPU via wgpu (renderer)
- WASM via wasm-bindgen (browser boundary)

## Build

```bash
cargo build                      # native (CLI + tests)
cargo test                       # all workspace tests
cargo run -p sim-cli -- --help   # run CLI
```

## Architecture

Nine crates in dependency order:

1. **sim-core** — foundation types, units, time/solar model, state bus, energy balance
2. **sim-vehicle** — parameterized vehicle archetypes (HALE, quad, strato, AUV, airship, rover)
3. **sim-environment** — queryable environment models (atmosphere ISA, ocean, space, terrain/Mars)
4. **sim-mmast** — MMAST module library (PV, metasurface ARC, radiative cooling, VO₂, spectral TEG, soaring, RF, TENG, EAD ion)
5. **sim-dynamics** — solver: 6-DOF integration + MMAST evaluation + energy balance
6. **sim-render** — WebGPU renderer (consumes sim state, never modifies it)
7. **sim-report** — analytical reporting (energy summary, per-module breakdown, feasibility, sensitivity)
8. **sim-wasm** — WASM boundary for browser (exposes solver + reporter as JS-callable functions)
9. **sim-cli** — headless CLI for batch runs and parameter sweeps

## Key Design Principles

- **The energy-balance inequality is the universal abstraction.** Every vehicle in every medium must satisfy ∫harvest ≥ ∫demand + losses. The vehicle, environment, and MMAST modules are the three orthogonal parameterization axes.
- **Sim core is headless.** The renderer and reporter are separate consumers of the state stream. You can run the solver without rendering (CLI, batch, Monte Carlo).
- **MMAST modules are composable.** Each module implements `MmastModule` trait: `applicable()`, `power_w()`, `mass_kg()`. The solver evaluates all applicable modules every time step.
- **Every module pays back twice.** The dual-use property (stealth + sustainment) is encoded by making modules contribute both energy and signature effects.
