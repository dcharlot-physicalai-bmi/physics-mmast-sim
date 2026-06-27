//! VO₂ adaptive-emissivity battery jacket — the night-survival enabler.
//!
//! Above 68°C metal-insulator transition: ε ≈ 0.2 (low emissivity).
//! Below: ε ≈ 0.8 (high emissivity).
//! Wired as a thermal diode: at night, surface stays low-ε, radiative loss
//! to cold sky drops ~4×. Battery I²R self-heating holds 5–15°C with zero
//! parasitic heater draw. Saves ~250 Wh/night vs. resistive heating.

use crate::MmastModule;
use sim_core::time::SimClock;
use sim_environment::EnvConditions;
use sim_vehicle::VehicleParams;

/// VO₂ adaptive-emissivity battery enclosure.
#[derive(Debug, Clone)]
pub struct Vo2BatteryJacket {
    /// Heater power that would otherwise be required (W).
    pub avoided_heater_w: f64,
    /// Mass of the VO₂ film + enclosure (kg).
    pub mass_kg_val: f64,
}

impl Default for Vo2BatteryJacket {
    fn default() -> Self {
        Self {
            avoided_heater_w: 18.0, // ~18 W continuous at night ≈ 250 Wh over 14h
            mass_kg_val: 0.15,
        }
    }
}

impl MmastModule for Vo2BatteryJacket {
    fn name(&self) -> &'static str {
        "VO₂ battery jacket"
    }

    fn applicable(&self, _vehicle: &VehicleParams, env: &EnvConditions) -> bool {
        // Active when it's cold enough that a heater would be needed.
        // Ambient below ~5°C (278 K) would normally require heating.
        env.temperature_k < 278.0
    }

    fn power_w(
        &self,
        _vehicle: &VehicleParams,
        env: &EnvConditions,
        _clock: &SimClock,
        battery_temp_k: f64,
    ) -> f64 {
        // This module doesn't harvest — it avoids a cost.
        // Modeled as positive power (savings) equal to the heater that isn't running.
        // Scale by how cold it is: full savings below 263 K (-10°C),
        // partial savings between 263–278 K.
        // Also factor in battery temperature: if already warm, less savings needed.
        let cold_factor = ((278.0 - env.temperature_k) / 15.0).clamp(0.0, 1.0);
        let batt_factor = ((288.0 - battery_temp_k) / 25.0).clamp(0.0, 1.0);
        self.avoided_heater_w * cold_factor * batt_factor
    }

    fn mass_kg(&self) -> f64 {
        self.mass_kg_val
    }

    fn description(&self) -> &'static str {
        "VO₂ thermal diode: eliminates resistive battery heater at night, saves ~250 Wh"
    }
}
