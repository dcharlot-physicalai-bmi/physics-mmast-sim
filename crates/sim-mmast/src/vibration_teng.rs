//! Vibration / TENG skin harvest — boundary-layer flutter and prop BPF
//! piezo cantilever array. Microwatt-class; powers SHM sensor nodes.

use crate::MmastModule;
use sim_core::time::SimClock;
use sim_environment::EnvConditions;
use sim_vehicle::VehicleParams;

/// Vibration / triboelectric nanogenerator skin.
#[derive(Debug, Clone)]
pub struct VibrationTeng {
    /// Per-element power output (W) at nominal conditions.
    pub per_element_w: f64,
    /// Number of harvester elements.
    pub element_count: u32,
    /// Mass per element (kg).
    pub mass_per_element: f64,
}

impl Default for VibrationTeng {
    fn default() -> Self {
        Self {
            per_element_w: 0.003,   // 3 mW per piezo cantilever
            element_count: 20,
            mass_per_element: 0.002,
        }
    }
}

impl MmastModule for VibrationTeng {
    fn name(&self) -> &'static str {
        "Skin TENG"
    }

    fn applicable(&self, _vehicle: &VehicleParams, env: &EnvConditions) -> bool {
        // Needs to be moving through a medium (not in vacuum).
        env.density > 0.001
    }

    fn power_w(
        &self,
        _vehicle: &VehicleParams,
        env: &EnvConditions,
        _clock: &SimClock,
        _battery_temp_k: f64,
    ) -> f64 {
        // Scale with medium density relative to sea-level air.
        let density_ratio = (env.density / 1.225).min(2.0);
        self.per_element_w * self.element_count as f64 * density_ratio
    }

    fn mass_kg(&self) -> f64 {
        self.mass_per_element * self.element_count as f64
    }

    fn description(&self) -> &'static str {
        "Piezo cantilever + TENG skin; powers distributed SHM sensor nodes"
    }
}
