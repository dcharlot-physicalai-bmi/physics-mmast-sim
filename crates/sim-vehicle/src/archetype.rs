//! Preset vehicle archetypes — geometry + mass + PV area derived from
//! openie-cad's B-Rep kernel.

use crate::aero::MobilityModel;
use crate::{EnergyStorage, PropulsionType, VehicleParams};
use sim_core::Medium;
use sim_geometry::VehicleGeometry;

/// Helper: build VehicleParams from a VehicleGeometry + configuration.
fn from_geometry(
    geom: VehicleGeometry,
    battery: EnergyStorage,
    medium: Medium,
    propulsion: PropulsionType,
    secondary: Option<PropulsionType>,
    drivetrain_eff: f64,
    mobility: MobilityModel,
    avionics_w: f64,
    payload_w: f64,
) -> VehicleParams {
    let dry_mass = (geom.derived.mass_kg - battery.mass_kg).max(0.5);
    let solar_area = geom.derived.upward_projected_area_m2;
    VehicleParams {
        name: geom.name,
        dry_mass_kg: dry_mass,
        storage: battery,
        medium,
        propulsion,
        secondary_propulsion: secondary,
        drivetrain_efficiency: drivetrain_eff,
        mobility,
        solar_area_m2: solar_area,
        avionics_power_w: avionics_w,
        payload_power_w: payload_w,
    }
}

impl VehicleParams {
    /// HALE solar UAV — geometry from cad-kernel.
    pub fn hale_solar_uav() -> Self {
        let geom = VehicleGeometry::hale_solar_uav()
            .expect("HALE geometry must build");
        from_geometry(
            geom,
            EnergyStorage::li_ion(400.0),
            Medium::Atmosphere,
            PropulsionType::BrushlessProp,
            Some(PropulsionType::EadIon),
            0.60,
            MobilityModel::hale_solar_uav(),
            5.0, 5.0,
        )
    }

    /// Small reconnaissance quadcopter — geometry from cad-kernel.
    pub fn recon_quadcopter() -> Self {
        let geom = VehicleGeometry::recon_quadcopter()
            .expect("Quad geometry must build");
        from_geometry(
            geom,
            EnergyStorage::li_ion(90.0),
            Medium::Atmosphere,
            PropulsionType::BrushlessProp,
            None,
            0.55,
            MobilityModel::quadcopter(),
            3.0, 2.0,
        )
    }

    /// Stratospheric glider / pseudo-satellite — geometry from cad-kernel.
    pub fn stratospheric_glider() -> Self {
        let geom = VehicleGeometry::stratospheric_glider()
            .expect("Strato geometry must build");
        from_geometry(
            geom,
            EnergyStorage::li_ion(3000.0),
            Medium::Atmosphere,
            PropulsionType::BrushlessProp,
            Some(PropulsionType::EadIon),
            0.62,
            MobilityModel::stratospheric_glider(),
            15.0, 25.0,
        )
    }

    /// Autonomous underwater vehicle — geometry from cad-kernel.
    pub fn auv() -> Self {
        let geom = VehicleGeometry::auv()
            .expect("AUV geometry must build");
        from_geometry(
            geom,
            EnergyStorage::li_ion(800.0),
            Medium::Aquatic,
            PropulsionType::MarineProp,
            Some(PropulsionType::Passive),
            0.50,
            MobilityModel::auv_torpedo(),
            2.0, 5.0,
        )
    }

    /// High-altitude station-keeping airship — geometry from cad-kernel.
    pub fn airship() -> Self {
        let geom = VehicleGeometry::airship()
            .expect("Airship geometry must build");
        from_geometry(
            geom,
            EnergyStorage::li_ion(6000.0),
            Medium::Atmosphere,
            PropulsionType::Edf,
            Some(PropulsionType::EadIon),
            0.55,
            MobilityModel::airship(),
            20.0, 40.0,
        )
    }

    /// Cloud Carrier v2 — 170m autonomous stratospheric platform.
    ///
    /// 170 m × 170 m × 30 m lenticular, 10 tons, operates at 30 km.
    /// ~22,700 m² solar skin. No life support. Fully autonomous/robotic.
    ///
    /// Buoyancy: ~580,000 m³ envelope × 0.0167 kg/m³ net lift = ~9,700 kg.
    /// 10-ton mass budget fits within single-atmosphere H₂ buoyancy.
    pub fn cloud_carrier() -> Self {
        let geom = VehicleGeometry::cloud_carrier()
            .expect("Cloud carrier geometry must build");
        from_geometry(
            geom,
            EnergyStorage::li_ion(500_000.0),   // 500 kWh battery (2 tons at 250 Wh/kg)
            Medium::Atmosphere,
            PropulsionType::Edf,
            Some(PropulsionType::EadIon),
            0.60,
            MobilityModel::cloud_carrier(),
            2000.0,                             // 2 kW avionics
            8000.0,                             // 8 kW payload (fab + repair + launch idle)
        )
    }

    /// Small planetary rover — geometry from cad-kernel.
    pub fn planetary_rover() -> Self {
        let geom = VehicleGeometry::planetary_rover()
            .expect("Rover geometry must build");
        from_geometry(
            geom,
            EnergyStorage::li_ion(1200.0),
            Medium::Ground,
            PropulsionType::WheeledDrive,
            None,
            0.70,
            MobilityModel::mars_rover(),
            10.0, 30.0,
        )
    }
}
