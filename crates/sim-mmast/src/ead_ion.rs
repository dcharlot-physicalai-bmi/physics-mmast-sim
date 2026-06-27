//! Electroaerodynamic (EAD) ion propulsion — silent, no-moving-parts thrust.
//!
//! MIT Barrett group: multistaged ducted EAD at 10–20 N/kW.
//! Used as a terminal stealth burst, not primary cruise.

use crate::MmastModule;
use sim_core::time::SimClock;
use sim_environment::EnvConditions;
use sim_vehicle::VehicleParams;

/// EAD ion propulsion module (cost, not harvest).
#[derive(Debug, Clone)]
pub struct EadIonPropulsion {
    /// Thrust-to-power ratio (N/kW).
    pub thrust_to_power_nkw: f64,
    /// Required thrust for the stealth segment (N).
    pub stealth_thrust_n: f64,
    /// Stealth burst duration per day (seconds).
    pub burst_duration_s: f64,
    /// Mass of the HV converter + electrodes (kg).
    pub mass_kg_val: f64,
}

impl Default for EadIonPropulsion {
    fn default() -> Self {
        Self {
            thrust_to_power_nkw: 15.0,
            stealth_thrust_n: 2.75, // same as cruise thrust for 7 kg aircraft
            burst_duration_s: 600.0, // 10 minutes per day
            mass_kg_val: 0.25,
        }
    }
}

impl MmastModule for EadIonPropulsion {
    fn name(&self) -> &'static str {
        "EAD ion burst"
    }

    fn applicable(&self, vehicle: &VehicleParams, _env: &EnvConditions) -> bool {
        vehicle.secondary_propulsion == Some(sim_vehicle::PropulsionType::EadIon)
    }

    fn power_w(
        &self,
        _vehicle: &VehicleParams,
        _env: &EnvConditions,
        clock: &SimClock,
        _battery_temp_k: f64,
    ) -> f64 {
        // Model: one burst per day at a fixed time (02:00–02:10 local for
        // peak stealth). Reports as a negative power (cost).
        let hour = clock.solar_hour;
        let burst_active = hour >= 2.0 && hour < 2.0 + self.burst_duration_s / 3600.0;

        if burst_active {
            // Power = thrust / (thrust-to-power ratio in N/kW) × 1000.
            let power_w = self.stealth_thrust_n / self.thrust_to_power_nkw * 1000.0;
            -power_w // cost
        } else {
            0.0 // not active
        }
    }

    fn mass_kg(&self) -> f64 {
        self.mass_kg_val
    }

    fn description(&self) -> &'static str {
        "Multistaged ducted EAD: zero acoustic/thermal signature for 10 min terminal burst"
    }
}
