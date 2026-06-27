//! `sim-mmast` — Multi-Material Adaptive Surface Technology module library.
//!
//! Each module implements a single physics contribution: an instantaneous
//! power (positive = harvest, negative = cost), a mass penalty, and
//! conditions under which it is applicable. The simulator attaches a set
//! of modules to a vehicle and the solver queries them every time step.
//!
//! The unifying insight from drone-notes.md: every module added for one
//! purpose pays back in at least two. The metasurface that does radar
//! cloak is the same one that does ARC; the radiative cooling film that
//! does IR stealth is the same one that cools the PV. This library
//! encodes that dual-use property by making modules composable and
//! by exposing both the energy and the signature effect.

pub mod pv;
pub mod metasurface_arc;
pub mod radiative_cooling;
pub mod vo2_jacket;
pub mod spectral_split;
pub mod soaring;
pub mod rf_harvest;
pub mod vibration_teng;
pub mod ead_ion;
pub mod lut;

use sim_core::state::ModulePowerRecord;
use sim_core::time::SimClock;
use sim_environment::EnvConditions;
use sim_vehicle::VehicleParams;

/// Trait that every MMAST module implements.
pub trait MmastModule: Send + Sync {
    /// Human-readable name for dashboards and reports. Must be `'static`
    /// so consumers can zero-copy into `ModulePowerRecord::name`.
    fn name(&self) -> &'static str;

    /// Whether this module is physically applicable in the current conditions.
    /// E.g., soaring requires atmosphere with shear; PV requires solar flux.
    fn applicable(&self, vehicle: &VehicleParams, env: &EnvConditions) -> bool;

    /// Instantaneous power contribution (W). Positive = harvest, negative = cost.
    fn power_w(
        &self,
        vehicle: &VehicleParams,
        env: &EnvConditions,
        clock: &SimClock,
        battery_temp_k: f64,
    ) -> f64;

    /// Mass added to the vehicle (kg).
    fn mass_kg(&self) -> f64;

    /// One-line description for UI tooltips.
    fn description(&self) -> &'static str;
}

/// Evaluate all modules for one time step, returning per-module records.
pub fn evaluate_modules(
    modules: &[Box<dyn MmastModule>],
    vehicle: &VehicleParams,
    env: &EnvConditions,
    clock: &SimClock,
    battery_temp_k: f64,
    dt_hours: f64,
    accumulators: &mut Vec<f64>,
) -> Vec<ModulePowerRecord> {
    if accumulators.len() < modules.len() {
        accumulators.resize(modules.len(), 0.0);
    }

    modules
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let active = m.applicable(vehicle, env);
            let power = if active {
                m.power_w(vehicle, env, clock, battery_temp_k)
            } else {
                0.0
            };
            accumulators[i] += power * dt_hours;

            ModulePowerRecord {
                name: std::borrow::Cow::Borrowed(m.name()),
                power_w: power,
                accumulated_wh: accumulators[i],
                active,
            }
        })
        .collect()
}

/// Build the full MMAST stack — all modules included. Each module's
/// `applicable()` method gates itself per vehicle + environment, so it's
/// safe to include everything. Modules that don't apply simply report
/// `active: false` and `power_w: 0.0`.
pub fn full_stack() -> Vec<Box<dyn MmastModule>> {
    vec![
        Box::new(pv::PhotovoltaicModule::default()),
        Box::new(metasurface_arc::MetasurfaceArc::default()),
        Box::new(radiative_cooling::RadiativeCooling::default()),
        Box::new(vo2_jacket::Vo2BatteryJacket::default()),
        Box::new(spectral_split::SpectralSplitTeg::default()),
        Box::new(soaring::DynamicSoaring::default()),
        Box::new(rf_harvest::RfHarvest::default()),
        Box::new(vibration_teng::VibrationTeng::default()),
        Box::new(ead_ion::EadIonPropulsion::default()),
    ]
}

/// Alias — backwards compat.
pub fn default_hale_stack() -> Vec<Box<dyn MmastModule>> {
    full_stack()
}
