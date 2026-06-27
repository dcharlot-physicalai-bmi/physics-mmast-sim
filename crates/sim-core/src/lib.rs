//! `sim-core` — Foundation types for the MMAST physics simulator.
//!
//! Everything in the simulator flows through this crate: physical units,
//! time representation, the state bus that connects the solver to consumers
//! (renderer, reporter, analytics), and the trait interfaces that define
//! what a Vehicle, Environment, and Module are.

pub mod units;
pub mod state;
pub mod time;
pub mod energy;

use serde::{Deserialize, Serialize};

// Re-export nalgebra and glam so downstream crates don't need direct deps.
pub use glam;
pub use nalgebra;

/// Unique identifier for any entity in the simulation.
pub type EntityId = uuid::Uuid;

/// A medium the vehicle operates in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Medium {
    /// Atmospheric flight (troposphere, stratosphere, mesosphere).
    Atmosphere,
    /// Surface or subsurface aquatic operation.
    Aquatic,
    /// Orbital or deep-space vacuum.
    Space,
    /// Ground contact (wheeled, tracked, legged).
    Ground,
}

/// Top-level simulation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    /// Solver time step (seconds).
    pub dt: f64,
    /// Mission duration (seconds).
    pub duration: f64,
    /// Operating medium.
    pub medium: Medium,
    /// Geographic latitude (degrees, for solar/atmospheric models).
    pub latitude: f64,
    /// Day of year (1–366, for solar declination).
    pub day_of_year: u16,
    /// Start hour (0.0–24.0, local solar time).
    pub start_hour: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,            // 100 Hz
            duration: 86_400.0,  // 24 hours
            medium: Medium::Atmosphere,
            latitude: 35.0,      // mid-latitude
            day_of_year: 172,    // summer solstice
            start_hour: 6.0,
        }
    }
}
