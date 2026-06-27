//! Ambient RF energy harvesting — HaiLa BSC2000, ONiO.zero class.
//!
//! Microwatt-class in flight; primary value is powering the avionics
//! watchdog bus independently of the main propulsion battery.

use crate::MmastModule;
use sim_core::time::SimClock;
use sim_environment::EnvConditions;
use sim_vehicle::VehicleParams;

/// Ambient RF rectenna harvester.
#[derive(Debug, Clone)]
pub struct RfHarvest {
    /// Effective rectenna area (m²).
    pub rectenna_area_m2: f64,
    /// Power conversion efficiency at typical input levels.
    pub pce: f64,
    /// Mass of the rectenna patch (kg).
    pub mass_kg_val: f64,
}

impl Default for RfHarvest {
    fn default() -> Self {
        Self {
            rectenna_area_m2: 0.01,
            pce: 0.30,
            mass_kg_val: 0.005,
        }
    }
}

impl MmastModule for RfHarvest {
    fn name(&self) -> &'static str {
        "Ambient RF harvest"
    }

    fn applicable(&self, _vehicle: &VehicleParams, env: &EnvConditions) -> bool {
        env.ambient_rf_density > 0.0
    }

    fn power_w(
        &self,
        _vehicle: &VehicleParams,
        env: &EnvConditions,
        _clock: &SimClock,
        _battery_temp_k: f64,
    ) -> f64 {
        env.ambient_rf_density * self.rectenna_area_m2 * self.pce
    }

    fn mass_kg(&self) -> f64 {
        self.mass_kg_val
    }

    fn description(&self) -> &'static str {
        "Rectenna patch harvesting ambient RF; powers avionics watchdog bus"
    }
}
