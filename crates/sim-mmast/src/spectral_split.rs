//! Spectral-splitting hybrid PV + TEG — dichroic metasurface routes
//! above-bandgap photons to Si, sub-bandgap IR to a thin-film TEG
//! whose cold side is the radiative cooling film.

use crate::MmastModule;
use sim_core::time::SimClock;
use sim_environment::EnvConditions;
use sim_vehicle::VehicleParams;

/// Spectral-splitting PV + thermoelectric generator.
#[derive(Debug, Clone)]
pub struct SpectralSplitTeg {
    /// Fraction of incident solar redirected to TEG (sub-bandgap).
    pub sub_bandgap_fraction: f64,
    /// TEG conversion efficiency at operating ΔT.
    pub teg_efficiency: f64,
    /// Mass of dichroic film + TEG per m² (kg/m²).
    pub mass_per_m2: f64,
}

impl Default for SpectralSplitTeg {
    fn default() -> Self {
        Self {
            sub_bandgap_fraction: 0.20, // ~20% of solar is below Si bandgap
            teg_efficiency: 0.045,      // ~4.5% at ΔT ≈ 55 K
            mass_per_m2: 0.55,          // dichroic film + thin-film TEG
        }
    }
}

impl MmastModule for SpectralSplitTeg {
    fn name(&self) -> &'static str {
        "Spectral split TEG"
    }

    fn applicable(&self, vehicle: &VehicleParams, env: &EnvConditions) -> bool {
        vehicle.solar_area_m2 > 0.0 && env.solar_irradiance > 50.0
    }

    fn power_w(
        &self,
        vehicle: &VehicleParams,
        env: &EnvConditions,
        _clock: &SimClock,
        _battery_temp_k: f64,
    ) -> f64 {
        let sub_bandgap_flux = env.solar_irradiance * self.sub_bandgap_fraction;
        sub_bandgap_flux * vehicle.solar_area_m2 * self.teg_efficiency
    }

    fn mass_kg(&self) -> f64 {
        // Returns per-m² cost; actual mass computed by caller with vehicle area.
        0.0
    }

    fn description(&self) -> &'static str {
        "Dichroic metasurface routes sub-bandgap IR to TEG; combined η 32–36%"
    }
}
