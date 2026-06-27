//! Simulation state bus — the single source of truth that the solver writes
//! and consumers (renderer, reporter, dashboard) read.

use crate::time::SimClock;
use crate::EntityId;
use serde::{Deserialize, Serialize};

/// Per-module harvest/cost record for the current time step.
///
/// Uses `Cow<'static, str>` for the name so static names (the common case
/// for MMAST modules) avoid heap allocation on the hot path.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModulePowerRecord {
    /// Module name (e.g. "PV baseline", "VO₂ jacket").
    pub name: std::borrow::Cow<'static, str>,
    /// Instantaneous power contribution (positive = harvest, negative = cost).
    pub power_w: f64,
    /// Accumulated energy contribution over the mission so far (Wh).
    pub accumulated_wh: f64,
    /// Whether this module is currently active.
    pub active: bool,
}

/// The complete simulation state snapshot at one instant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimState {
    /// Unique run ID.
    pub run_id: EntityId,
    /// Clock state.
    pub clock: SimClock,

    // ---- Kinematics ----
    /// World-space position (m).
    pub position: [f64; 3],
    /// World-space velocity (m/s).
    pub velocity: [f64; 3],
    /// Attitude quaternion [x, y, z, w].
    pub attitude: [f64; 4],
    /// Body-frame angular velocity (rad/s).
    pub angular_velocity: [f64; 3],

    // ---- Energy ----
    /// Battery state of charge (Wh remaining).
    pub battery_soc_wh: f64,
    /// Battery capacity (Wh).
    pub battery_capacity_wh: f64,
    /// Battery temperature (K).
    pub battery_temp_k: f64,
    /// Total instantaneous harvest power (W).
    pub total_harvest_w: f64,
    /// Total instantaneous demand power (W).
    pub total_demand_w: f64,
    /// Net instantaneous power (harvest - demand) (W).
    pub net_power_w: f64,

    // ---- Per-module breakdown ----
    /// Ordered list of active MMAST modules and their contributions.
    pub modules: Vec<ModulePowerRecord>,

    // ---- Vehicle ----
    /// Propulsion mode (e.g. "brushless_prop", "ead_ion", "idle").
    pub propulsion_mode: String,
    /// Airspeed / waterflow speed / ground speed (m/s).
    pub speed_mps: f64,
    /// Altitude above reference (m) or depth below surface (negative).
    pub altitude_m: f64,

    // ---- Environment ----
    /// Ambient temperature at vehicle position (K).
    pub ambient_temp_k: f64,
    /// Ambient pressure (Pa).
    pub ambient_pressure_pa: f64,
    /// Medium density at vehicle position (kg/m³).
    pub medium_density: f64,
    /// Solar irradiance at vehicle surface (W/m²), accounting for clouds/depth.
    pub solar_irradiance_wm2: f64,
    /// Wind / current vector at vehicle position (m/s).
    pub wind_vector: [f64; 3],
}

impl SimState {
    /// Battery state of charge as a fraction (0.0–1.0).
    pub fn soc_fraction(&self) -> f64 {
        if self.battery_capacity_wh <= 0.0 {
            return 0.0;
        }
        (self.battery_soc_wh / self.battery_capacity_wh).clamp(0.0, 1.0)
    }

    /// Daily energy margin projected from current instantaneous rates (Wh).
    pub fn projected_daily_margin_wh(&self) -> f64 {
        self.net_power_w * 24.0
    }
}

/// Time-series of state snapshots — the full mission record.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimTimeSeries {
    pub snapshots: Vec<SimState>,
}

impl SimTimeSeries {
    pub fn push(&mut self, state: SimState) {
        self.snapshots.push(state);
    }

    pub fn duration_hours(&self) -> f64 {
        self.snapshots
            .last()
            .map(|s| s.clock.elapsed / 3600.0)
            .unwrap_or(0.0)
    }
}
