//! `sim-environment` — Queryable environment models for all operating media.
//!
//! The environment is a function of (position, time) → physical conditions.
//! Each medium (atmosphere, ocean, space, ground) provides a different set
//! of ambient properties. The MMAST modules and the dynamics solver query
//! these properties every time step.

pub mod atmosphere;
pub mod atmosphere_lut;
pub mod ocean;
pub mod space;
pub mod terrain;

use serde::{Deserialize, Serialize};
use sim_core::time::SimClock;

/// Environmental conditions at a point in space and time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvConditions {
    /// Ambient temperature (K).
    pub temperature_k: f64,
    /// Ambient pressure (Pa).
    pub pressure_pa: f64,
    /// Medium density (kg/m³).
    pub density: f64,
    /// Dynamic viscosity (Pa·s).
    pub viscosity: f64,
    /// Speed of sound in the medium (m/s).
    pub speed_of_sound: f64,
    /// Gravitational acceleration (m/s²).
    pub gravity: f64,
    /// Solar irradiance reaching the vehicle (W/m²), after atmospheric
    /// extinction, cloud cover, and depth attenuation.
    pub solar_irradiance: f64,
    /// Sky temperature for radiative cooling calculations (K).
    /// In atmosphere: ~3 K effective in the 8–13 µm window at altitude.
    /// In space: 2.7 K CMB.
    /// In ocean: N/A (set to ambient).
    pub sky_temperature_k: f64,
    /// Wind / current velocity at this position (m/s), world frame.
    pub wind_velocity: [f64; 3],
    /// Ambient RF power density (W/m²) — for RF harvesting.
    pub ambient_rf_density: f64,
    /// Cloud cover fraction (0.0–1.0).
    pub cloud_cover: f64,
}

/// Trait implemented by each environment model.
pub trait Environment: Send + Sync {
    /// Query the full conditions at a position and time.
    fn conditions(&self, position: [f64; 3], clock: &SimClock) -> EnvConditions;

    /// Fast-path density query. Default implementation calls `conditions`,
    /// but environments with internal LUTs should override to skip the
    /// full struct construction.
    fn density_at(&self, position: [f64; 3], clock: &SimClock) -> f64 {
        self.conditions(position, clock).density
    }

    /// Fast-path temperature query.
    fn temperature_at(&self, position: [f64; 3], clock: &SimClock) -> f64 {
        self.conditions(position, clock).temperature_k
    }

    /// Terrain / seabed / ground height at (x, z) coordinates (m).
    /// Returns `None` if not applicable (e.g., space).
    fn ground_height(&self, x: f64, z: f64) -> Option<f64>;
}
