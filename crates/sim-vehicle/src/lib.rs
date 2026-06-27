//! `sim-vehicle` — Parameterized vehicle models for multi-domain simulation.
//!
//! Each vehicle archetype defines its geometry, mass properties, propulsive
//! efficiency, aerodynamic/hydrodynamic model, and energy storage. The
//! simulator instantiates one of these and the MMAST module library attaches
//! physics modifiers to it based on what the environment permits.

pub mod aero;
pub mod archetype;

use serde::{Deserialize, Serialize};

/// Propulsion system type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropulsionType {
    /// Brushless electric motor + propeller.
    BrushlessProp,
    /// Electroaerodynamic (ion) thruster.
    EadIon,
    /// Jet / turbine.
    Turbine,
    /// Electric ducted fan.
    Edf,
    /// Marine propeller or waterjet.
    MarineProp,
    /// Ion thruster (space).
    SpaceIon,
    /// Reaction wheels (attitude only).
    ReactionWheels,
    /// Wheeled drive (ground).
    WheeledDrive,
    /// Tracked drive (ground).
    TrackedDrive,
    /// Legged locomotion.
    Legged,
    /// Passive (glider, drifter, satellite in coast).
    Passive,
}

/// Energy storage model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyStorage {
    /// Total capacity (Wh).
    pub capacity_wh: f64,
    /// Current state of charge (Wh).
    pub soc_wh: f64,
    /// Round-trip efficiency (0.0–1.0, typically ~0.90 for Li-ion).
    pub round_trip_efficiency: f64,
    /// Mass of the storage system (kg).
    pub mass_kg: f64,
    /// Specific energy (Wh/kg).
    pub specific_energy: f64,
    /// Current temperature (K).
    pub temperature_k: f64,
    /// Temperature coefficient — capacity loss fraction per K below nominal.
    pub temp_coeff_per_k: f64,
    /// Nominal operating temperature (K).
    pub nominal_temp_k: f64,
}

impl EnergyStorage {
    /// Usable capacity at current temperature.
    pub fn usable_capacity_wh(&self) -> f64 {
        let dt = (self.nominal_temp_k - self.temperature_k).max(0.0);
        let derating = 1.0 - self.temp_coeff_per_k * dt;
        self.capacity_wh * derating.max(0.1)
    }

    /// Li-ion 250 Wh/kg default.
    pub fn li_ion(capacity_wh: f64) -> Self {
        let mass = capacity_wh / 250.0;
        Self {
            capacity_wh,
            soc_wh: capacity_wh,
            round_trip_efficiency: 0.90,
            mass_kg: mass,
            specific_energy: 250.0,
            temperature_k: 293.15,
            temp_coeff_per_k: 0.005,
            nominal_temp_k: 298.15,
        }
    }
}

/// Full vehicle parameter set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleParams {
    /// Human-readable name.
    pub name: String,
    /// Total dry mass (kg), excluding battery.
    pub dry_mass_kg: f64,
    /// Energy storage.
    pub storage: EnergyStorage,
    /// Operating medium.
    pub medium: sim_core::Medium,
    /// Primary propulsion type.
    pub propulsion: PropulsionType,
    /// Secondary propulsion (e.g., EAD for stealth burst).
    pub secondary_propulsion: Option<PropulsionType>,
    /// Overall drivetrain efficiency (prop × motor × ESC).
    pub drivetrain_efficiency: f64,
    /// Mobility model (fixed-wing, rotorcraft, marine, ground, space).
    pub mobility: aero::MobilityModel,
    /// Solar-facing surface area (m²) — available for PV modules.
    pub solar_area_m2: f64,
    /// Avionics continuous power draw (W).
    pub avionics_power_w: f64,
    /// Payload continuous power draw (W).
    pub payload_power_w: f64,
}

impl VehicleParams {
    /// Total mass including battery (kg).
    pub fn total_mass_kg(&self) -> f64 {
        self.dry_mass_kg + self.storage.mass_kg
    }

    /// Required mechanical cruise power (W) in the given medium density.
    pub fn cruise_power_mechanical(&self, medium_density: f64) -> f64 {
        self.mobility.cruise_power_mechanical(self.total_mass_kg(), medium_density)
    }

    /// Required electrical cruise power (W) including drivetrain losses.
    pub fn cruise_power_electrical(&self, medium_density: f64) -> f64 {
        self.cruise_power_mechanical(medium_density) / self.drivetrain_efficiency
    }

    /// Total continuous electrical demand (W).
    pub fn total_demand_w(&self, medium_density: f64) -> f64 {
        self.cruise_power_electrical(medium_density)
            + self.avionics_power_w
            + self.payload_power_w
    }
}
