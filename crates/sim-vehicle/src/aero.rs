//! Mobility model — parameterized power-to-move for all vehicle types.
//!
//! Each regime (fixed-wing, rotorcraft, marine, ground, space) has a
//! different physics model for computing cruise power from vehicle
//! parameters and medium conditions.

use serde::{Deserialize, Serialize};

/// Mobility regime — determines which power model to use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MobilityModel {
    /// Fixed-wing: parabolic drag polar, P = W·v / (L/D).
    FixedWing(FixedWingParams),
    /// Rotorcraft: disk loading model, P = T^1.5 / √(2·ρ·A).
    Rotorcraft(RotorcraftParams),
    /// Marine: drag = 0.5·ρ·v²·Cd·A (no lift requirement).
    Marine(MarineParams),
    /// Ground: rolling resistance + aero drag + grade.
    Ground(GroundParams),
    /// Space: no drag, power only for ΔV maneuvers.
    Space(SpaceParams),
}

/// Fixed-wing aerodynamic parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedWingParams {
    /// Wing area (m²).
    pub wing_area_m2: f64,
    /// Wingspan (m).
    pub span_m: f64,
    /// Aspect ratio.
    pub aspect_ratio: f64,
    /// Maximum L/D.
    pub ld_max: f64,
    /// Cruise speed (m/s).
    pub cruise_speed_mps: f64,
    /// Zero-lift drag coefficient.
    pub cd0: f64,
    /// Oswald efficiency factor.
    pub oswald_e: f64,
}

impl FixedWingParams {
    pub fn cruise_cl(&self, mass_kg: f64, density: f64) -> f64 {
        let w = mass_kg * 9.81;
        let q = 0.5 * density * self.cruise_speed_mps.powi(2);
        w / (q * self.wing_area_m2)
    }

    pub fn cd_at_cl(&self, cl: f64) -> f64 {
        self.cd0 + cl.powi(2) / (std::f64::consts::PI * self.oswald_e * self.aspect_ratio)
    }

    pub fn ld_at_cl(&self, cl: f64) -> f64 {
        cl / self.cd_at_cl(cl)
    }

    pub fn cruise_power_w(&self, mass_kg: f64, density: f64) -> f64 {
        let w = mass_kg * 9.81;
        let cl = self.cruise_cl(mass_kg, density);
        let ld = self.ld_at_cl(cl);
        w * self.cruise_speed_mps / ld
    }
}

/// Rotorcraft parameters (quad, hex, octo, coaxial).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotorcraftParams {
    /// Total rotor disk area (m²) — sum of all rotors.
    pub disk_area_m2: f64,
    /// Figure of merit (hover efficiency, 0.5–0.8).
    pub figure_of_merit: f64,
    /// Parasite drag area f = Cd·S (m²) for forward flight.
    pub parasite_drag_area_m2: f64,
    /// Cruise speed (m/s).
    pub cruise_speed_mps: f64,
}

impl RotorcraftParams {
    /// Hover power: P_hover = T^1.5 / (FM · √(2·ρ·A)).
    pub fn hover_power_w(&self, mass_kg: f64, density: f64) -> f64 {
        let thrust = mass_kg * 9.81;
        thrust.powf(1.5) / (self.figure_of_merit * (2.0 * density * self.disk_area_m2).sqrt())
    }

    /// Forward flight power (simplified: hover + parasite drag).
    pub fn cruise_power_w(&self, mass_kg: f64, density: f64) -> f64 {
        // In forward flight, induced power decreases but parasite drag increases.
        // Simplified: P_cruise ≈ 0.7·P_hover + 0.5·ρ·v³·f
        let p_hover = self.hover_power_w(mass_kg, density);
        let p_parasite = 0.5 * density * self.cruise_speed_mps.powi(3) * self.parasite_drag_area_m2;
        0.7 * p_hover + p_parasite
    }
}

/// Marine / AUV parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarineParams {
    /// Wetted reference area (m²).
    pub wetted_area_m2: f64,
    /// Drag coefficient.
    pub cd: f64,
    /// Cruise speed (m/s).
    pub cruise_speed_mps: f64,
}

impl MarineParams {
    pub fn cruise_power_w(&self, _mass_kg: f64, density: f64) -> f64 {
        // P = F·v = 0.5·ρ·v²·Cd·A · v = 0.5·ρ·v³·Cd·A
        0.5 * density * self.cruise_speed_mps.powi(3) * self.cd * self.wetted_area_m2
    }
}

/// Ground vehicle parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundParams {
    /// Rolling resistance coefficient (0.01–0.05 paved, 0.1–0.35 loose soil/regolith).
    pub crr: f64,
    /// Frontal area for aerodynamic drag (m²).
    pub frontal_area_m2: f64,
    /// Aerodynamic drag coefficient.
    pub cd_aero: f64,
    /// Cruise speed (m/s).
    pub cruise_speed_mps: f64,
    /// Grade (fraction, 0.0 = flat, 0.05 = 5% slope).
    pub grade: f64,
}

impl GroundParams {
    /// Cruise power: P = v · (F_rolling + F_aero + F_grade).
    pub fn cruise_power_w(&self, mass_kg: f64, air_density: f64) -> f64 {
        let w = mass_kg * 9.81;
        let v = self.cruise_speed_mps;

        // Rolling resistance: F = Crr · W
        let f_rolling = self.crr * w;

        // Aerodynamic drag: F = 0.5 · ρ · v² · Cd · A
        let f_aero = 0.5 * air_density * v.powi(2) * self.cd_aero * self.frontal_area_m2;

        // Grade resistance: F = W · sin(atan(grade)) ≈ W · grade for small grades
        let f_grade = w * self.grade;

        v * (f_rolling + f_aero + f_grade)
    }
}

/// Space vehicle parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceParams {
    /// Station-keeping ΔV budget per day (m/s/day). LEO ≈ 0, GEO ≈ 0.05.
    pub stationkeep_dv_per_day: f64,
    /// Specific impulse of propulsion (s).
    pub isp_s: f64,
    /// Reaction wheel power draw for attitude control (W).
    pub attitude_power_w: f64,
}

impl SpaceParams {
    pub fn cruise_power_w(&self, _mass_kg: f64, _density: f64) -> f64 {
        // In space, "cruise" power is just attitude control.
        // Propulsive ΔV is impulsive, not continuous power.
        self.attitude_power_w
    }
}

// ---- Unified interface ----

impl MobilityModel {
    /// Compute mechanical cruise power (W) for any vehicle type.
    pub fn cruise_power_mechanical(&self, mass_kg: f64, medium_density: f64) -> f64 {
        match self {
            Self::FixedWing(p)  => p.cruise_power_w(mass_kg, medium_density),
            Self::Rotorcraft(p) => p.cruise_power_w(mass_kg, medium_density),
            Self::Marine(p)     => p.cruise_power_w(mass_kg, medium_density),
            Self::Ground(p)     => p.cruise_power_w(mass_kg, medium_density),
            Self::Space(p)      => p.cruise_power_w(mass_kg, medium_density),
        }
    }

    /// Cruise speed (m/s).
    pub fn cruise_speed(&self) -> f64 {
        match self {
            Self::FixedWing(p)  => p.cruise_speed_mps,
            Self::Rotorcraft(p) => p.cruise_speed_mps,
            Self::Marine(p)     => p.cruise_speed_mps,
            Self::Ground(p)     => p.cruise_speed_mps,
            Self::Space(_)      => 0.0, // orbital, not ground-relative
        }
    }

    /// L/D at cruise (only meaningful for fixed-wing; returns 1.0 for others).
    pub fn cruise_ld(&self, mass_kg: f64, density: f64) -> f64 {
        match self {
            Self::FixedWing(p) => {
                let cl = p.cruise_cl(mass_kg, density);
                p.ld_at_cl(cl)
            }
            _ => 1.0,
        }
    }

    /// Maximum L/D (only meaningful for fixed-wing; returns 1.0 for others).
    pub fn ld_max(&self) -> f64 {
        match self {
            Self::FixedWing(p) => p.ld_max,
            _ => 1.0,
        }
    }
}

// ---- Presets ----

impl MobilityModel {
    /// HALE solar UAV — the drone-notes.md 7 kg reference design.
    pub fn hale_solar_uav() -> Self {
        Self::FixedWing(FixedWingParams {
            wing_area_m2: 3.0,
            span_m: 6.0,
            aspect_ratio: 12.0,
            ld_max: 25.0,
            cruise_speed_mps: 12.0,
            cd0: 0.018,
            oswald_e: 0.85,
        })
    }

    /// Small reconnaissance quadcopter.
    pub fn quadcopter() -> Self {
        Self::Rotorcraft(RotorcraftParams {
            disk_area_m2: 4.0 * std::f64::consts::PI * 0.127_f64.powi(2), // 4× 5" props
            figure_of_merit: 0.55,
            parasite_drag_area_m2: 0.008,
            cruise_speed_mps: 8.0,
        })
    }

    /// AUV torpedo-body glider.
    pub fn auv_torpedo() -> Self {
        Self::Marine(MarineParams {
            wetted_area_m2: 0.5,
            cd: 0.08,
            cruise_speed_mps: 1.5,
        })
    }

    /// High-altitude station-keeping airship.
    pub fn airship() -> Self {
        Self::FixedWing(FixedWingParams {
            wing_area_m2: 40.0,
            span_m: 8.0,
            aspect_ratio: 1.6,
            ld_max: 4.0,
            cruise_speed_mps: 5.0,
            cd0: 0.025,
            oswald_e: 0.75,
        })
    }

    /// Cloud Carrier — 50m autonomous stratospheric platform.
    ///
    /// Primary lift is buoyancy from lift gas (H2/He) — NOT aerodynamic.
    /// The platform floats; station-keeping only needs to counter wind drift,
    /// not hold the mass up. We model this as pure drag (Marine-style): the
    /// station-keeping propellers only overcome the pressure drag against
    /// stratospheric winds when drifting off station.
    ///
    /// At 30 km: ρ_air ≈ 0.018 kg/m³, typical wind 15 m/s, the carrier
    /// operates with ~5 m/s ground-relative speed during station-keeping.
    /// Frontal area (side view) ≈ 500 m² for a 50×10 m lenticular.
    pub fn cloud_carrier() -> Self {
        Self::Marine(MarineParams {
            wetted_area_m2: 5100.0,     // side profile (170m × 30m)
            cd: 0.3,                    // lenticular is more streamlined at scale
            cruise_speed_mps: 5.0,      // ground-relative station-keeping speed
        })
    }

    /// Stratospheric glider (Zephyr-class).
    pub fn stratospheric_glider() -> Self {
        Self::FixedWing(FixedWingParams {
            wing_area_m2: 18.0,
            span_m: 25.0,
            aspect_ratio: 35.0,
            ld_max: 35.0,
            cruise_speed_mps: 18.0,
            cd0: 0.012,
            oswald_e: 0.90,
        })
    }

    /// Earth ground rover.
    pub fn earth_rover() -> Self {
        Self::Ground(GroundParams {
            crr: 0.015,           // paved / hard surface
            frontal_area_m2: 0.8,
            cd_aero: 0.4,
            cruise_speed_mps: 2.0,
            grade: 0.0,
        })
    }

    /// Mars rover (loose regolith, thin atmosphere).
    pub fn mars_rover() -> Self {
        Self::Ground(GroundParams {
            crr: 0.25,            // loose regolith — high rolling resistance
            frontal_area_m2: 0.8,
            cd_aero: 0.4,
            cruise_speed_mps: 0.04, // ~150 m/hr
            grade: 0.0,
        })
    }

    /// LEO satellite / cubesat.
    pub fn leo_satellite() -> Self {
        Self::Space(SpaceParams {
            stationkeep_dv_per_day: 0.0,
            isp_s: 300.0,
            attitude_power_w: 5.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hale_cruise_power_sanity() {
        let model = MobilityModel::hale_solar_uav();
        let p = model.cruise_power_mechanical(7.0, 1.225);
        // drone-notes.md estimated ~33 W at best L/D; actual drag polar
        // gives higher because cruise CL is off the L/D-max point.
        assert!(p > 20.0 && p < 80.0, "HALE cruise power was {p} W");
    }

    #[test]
    fn quad_hover_power_sanity() {
        let model = MobilityModel::quadcopter();
        let p = model.cruise_power_mechanical(1.5, 1.225);
        // A 1.5 kg quad typically draws ~100–200 W mechanical in cruise
        assert!(p > 50.0 && p < 300.0, "Quad cruise power was {p} W");
    }

    #[test]
    fn mars_rover_power_sanity() {
        let model = MobilityModel::mars_rover();
        // Mars: 80 kg, ρ ≈ 0.020, v = 0.04 m/s
        let p = model.cruise_power_mechanical(80.0, 0.020);
        // Rolling resistance: 0.25 × 80 × 3.72 × 0.04 ≈ 2.98 W
        // Aero is negligible at 0.04 m/s in thin atmosphere
        assert!(p > 1.0 && p < 10.0, "Mars rover cruise power was {p} W");
    }

    #[test]
    fn auv_power_sanity() {
        let model = MobilityModel::auv_torpedo();
        let p = model.cruise_power_mechanical(30.0, 1025.0);
        // 0.5 × 1025 × 1.5³ × 0.08 × 0.5 ≈ 69 W
        assert!(p > 30.0 && p < 150.0, "AUV cruise power was {p} W");
    }

    #[test]
    fn space_cruise_power_is_attitude_only() {
        let model = MobilityModel::leo_satellite();
        let p = model.cruise_power_mechanical(100.0, 0.0);
        assert!((p - 5.0).abs() < 0.01, "Space cruise power was {p} W");
    }
}
