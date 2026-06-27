<div align="center">

# MMAST Physics Simulator

**The surface that pays twice — persistence is an energy-balance problem.**

*A physics-informed, real-time multi-vehicle energy + signature simulator for the MMAST (Multi-Material Adaptive Surface Technology) stack. Every vehicle, in every medium, must satisfy one inequality: ∫harvest ≥ ∫demand + losses.*

[![Rust](https://img.shields.io/badge/Rust-edition_2024-cfaa5b?style=flat-square&logo=rust)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-46e0c0?style=flat-square)](LICENSE)
[![Charlot Lab](https://img.shields.io/badge/Charlot_Lab-Physical_AI_%40_BMI-cfaa5b?style=flat-square)](https://labs.physicalai-bmi.org/charlot)
[![Built on CadFuture](https://img.shields.io/badge/built_on-CadFuture-46e0c0?style=flat-square)](https://github.com/dcharlot-physicalai-bmi/cad-future)

</div>

---

## Why

An embodied system that has to *persist* — fly for months, run for a Martian year, stay dark — is governed by a single inequality: **harvest must meet demand, integrated over the mission.** MMAST treats the vehicle's **skin as the power plant** and reads every gram against that ledger.

And every layer pays back **twice**: the radiative-cooling film that cools the cells is the IR-stealth skin; the metasurface that lifts photovoltaic gain is the radar-absorbing coat; the TENG skin that scavenges vibration is the distributed structural-health sensor. Each module is added for one purpose and contributes to at least two — energy **and** signature.

## The abstraction

> **The energy-balance inequality is the universal abstraction.** Every vehicle in every medium must satisfy ∫harvest ≥ ∫demand + losses.

One physics-informed solver over **three orthogonal axes**:

- **Vehicle archetype** — HALE solar UAV, recon quad, stratospheric glider, AUV, station-keeping airship, cloud-carrier, planetary rover.
- **Medium** — atmosphere (ISA), ocean, space, terrain / Mars.
- **Surface module** — PV, metasurface ARC, radiative cooling, VO₂ battery jacket, spectral-split TEG, dynamic soaring, ambient-RF rectenna, skin TENG, EAD ion.

Each module implements the `MmastModule` trait — `applicable()`, `power_w()`, `mass_kg()` — and the solver evaluates every applicable module each time step, accumulating the energy ledger and the signature contribution.

## Built on CadFuture

Vehicle geometry, tessellation, and thermal FEA come from [**CadFuture**](https://github.com/dcharlot-physicalai-bmi/cad-future), the Charlot Lab's computable-world-model engine:

| CadFuture crate | Used for |
|-----------------|----------|
| `physical-brep` | B-Rep vehicle solids (`make_box`, …) |
| `physical-tessellation` | render-ready triangle meshes |
| `physical-fea` | steady-state thermal FEA (VO₂ jacket) |

Where OmniSense, CadFuture, and Graph of the World let an embodied system **perceive and model** the world, MMAST is the **act** side of the same loop: designing a body that can sustain itself in it.

## Workspace

A Rust workspace (`edition = 2024`). Twelve crates:

| Crate | Role |
|-------|------|
| `sim-core` | foundation types, units, time/solar model, state bus, energy balance |
| `sim-geometry` | parametric vehicle geometry on CadFuture's B-Rep kernel |
| `sim-vehicle` | parameterized vehicle archetypes + aero/mobility models |
| `sim-environment` | queryable environment models (atmosphere, ocean, space, Mars) |
| `sim-mmast` | the MMAST surface-module library |
| `sim-thermal` | VO₂ battery-jacket thermal model on CadFuture's FEA |
| `sim-dynamics` | solver — 6-DOF integration + MMAST evaluation + energy balance |
| `sim-render` | WebGPU renderer (consumes sim state, never mutates it) |
| `sim-report` | analytical reporting (energy summary, feasibility, sensitivity) |
| `sim-wasm` | WASM boundary for the browser |
| `sim-cli` | headless CLI for batch runs and parameter sweeps |
| `sim-viewer` | native real-time viewer |

## Build

```bash
cargo build                      # native (CLI + tests)
cargo test                       # all workspace tests
cargo run -p sim-cli -- --help   # run CLI
```

> **Note:** on a network filesystem without file locking, incremental compilation fails to acquire its session lock. Set `CARGO_INCREMENTAL=0`, or add `[build]\nincremental = false` to `.cargo/config.toml`.

## Links

- **Research topic** — https://physicalai-bmi.org/research/charlot-lab#topic-mmast
- **The Charlot Lab** — https://labs.physicalai-bmi.org/charlot
- **CadFuture** — https://github.com/dcharlot-physicalai-bmi/cad-future
- **Institute for Physical AI** — https://physicalai-bmi.org

---

<div align="center">
<sub>The Charlot Lab · Institute for Physical AI · Bailey Military Institute</sub>
</div>
