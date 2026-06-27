//! Mission envelope LUT — pre-computed 4D feasibility grid keyed on
//! (latitude × cloud cover × altitude × day of year) for a specific vehicle.
//!
//! This is the analytical reporting layer's LUT: instead of running a
//! live simulation every time the user drags a slider, the envelope
//! pre-computes the full parameter space once and serves every subsequent
//! query as an array read.
//!
//! The atmosphere LUT is rebaked once per day_of_year (cheap — ~4 ms)
//! and shared across all (lat × cloud × alt) cells for that day via
//! `Arc::clone`. Within each day, cells are computed in parallel via
//! rayon.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use sim_core::SimConfig;
use sim_dynamics::Solver;
use sim_environment::atmosphere::StandardAtmosphere;
use sim_environment::atmosphere_lut::AtmosphereLut;
use sim_vehicle::VehicleParams;

/// One cell in the mission envelope.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EnvelopeCell {
    pub harvested_wh: f64,
    pub consumed_wh: f64,
    pub margin_wh: f64,
}

impl EnvelopeCell {
    pub fn feasible(&self) -> bool {
        self.margin_wh > 0.0
    }
    pub fn margin_ratio(&self) -> f64 {
        if self.consumed_wh <= 0.0 {
            return f64::INFINITY;
        }
        self.margin_wh / self.consumed_wh
    }
}

/// Pre-computed 4D feasibility envelope.
///
/// Cell layout is row-major across (lat, cloud, alt, doy) with doy as
/// the innermost (fastest-varying) axis:
///   index = ((lat_i * cloud_bins + cloud_i) * alt_bins + alt_i) * doy_bins + doy_i
///
/// Per-module contributions are stored in a flat `module_contributions`
/// array indexed by `cell_index * module_names.len() + module_idx`.
/// This lets the UI toggle modules on/off at query time by re-summing
/// only the active ones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionEnvelope {
    pub vehicle: String,

    pub lat_min: f64,
    pub lat_max: f64,
    pub lat_bins: usize,

    pub cloud_min: f64,
    pub cloud_max: f64,
    pub cloud_bins: usize,

    pub alt_min: f64,
    pub alt_max: f64,
    pub alt_bins: usize,

    pub doy_min: u16,
    pub doy_max: u16,
    pub doy_bins: usize,

    /// Names of every tracked module (9 MMAST + 3 implicit costs).
    pub module_names: Vec<String>,

    /// Aggregate cells — harvested/consumed/margin totals.
    pub cells: Vec<EnvelopeCell>,

    /// Flat per-module breakdown: `cells.len() * module_names.len()` f32s.
    /// Positive = harvest contribution, negative = cost.
    pub module_contributions: Vec<f32>,
}

impl MissionEnvelope {
    /// Bake the 4D envelope. One 24h simulation per cell, parallelized
    /// within each day of year (atmosphere LUT shared across cells).
    #[allow(clippy::too_many_arguments)]
    pub fn bake(
        vehicle: VehicleParams,
        lat_min: f64,
        lat_max: f64,
        lat_bins: usize,
        cloud_min: f64,
        cloud_max: f64,
        cloud_bins: usize,
        alt_min: f64,
        alt_max: f64,
        alt_bins: usize,
        doy_min: u16,
        doy_max: u16,
        doy_bins: usize,
    ) -> Self {
        let total = lat_bins * cloud_bins * alt_bins * doy_bins;
        let mut cells = vec![EnvelopeCell::default(); total];

        let lat_step = if lat_bins > 1 { (lat_max - lat_min) / (lat_bins - 1) as f64 } else { 0.0 };
        let cloud_step = if cloud_bins > 1 { (cloud_max - cloud_min) / (cloud_bins - 1) as f64 } else { 0.0 };
        let alt_step = if alt_bins > 1 { (alt_max - alt_min) / (alt_bins - 1) as f64 } else { 0.0 };
        let doy_step = if doy_bins > 1 {
            (doy_max - doy_min) as f64 / (doy_bins - 1) as f64
        } else {
            0.0
        };

        let vehicle_name = vehicle.name.clone();

        // Seed the module name list from one run of the stack.
        let seed_modules = sim_mmast::full_stack();
        let seed_names: Vec<&'static str> = seed_modules.iter().map(|m| m.name()).collect();
        let mut module_names: Vec<String> = seed_names.iter().map(|s| s.to_string()).collect();
        // Implicit cost entries injected by the solver.
        module_names.push("Propulsion (cruise)".into());
        module_names.push("Avionics + payload".into());
        module_names.push("Battery losses".into());
        let num_modules = module_names.len();
        let mut module_contributions = vec![0.0_f32; total * num_modules];

        // Outer loop: day of year (rebakes the atmosphere LUT each iteration).
        // Inner parallel loop: (lat, cloud, alt) grid for this day.
        for doy_i in 0..doy_bins {
            let doy = (doy_min as f64 + doy_i as f64 * doy_step).round() as u16;
            let atmo_lut = Arc::new(AtmosphereLut::bake(doy));

            let tasks: Vec<(usize, usize, usize)> = (0..lat_bins)
                .flat_map(|li| {
                    (0..cloud_bins).flat_map(move |ci| {
                        (0..alt_bins).map(move |ai| (li, ci, ai))
                    })
                })
                .collect();

            let module_names_slice: Vec<String> = module_names.clone();
            let day_cells: Vec<(usize, EnvelopeCell, Vec<f32>)> = tasks
                .par_iter()
                .map(|&(li, ci, ai)| {
                    let latitude = lat_min + li as f64 * lat_step;
                    let cloud = cloud_min + ci as f64 * cloud_step;
                    let altitude = alt_min + ai as f64 * alt_step;

                    let config = SimConfig {
                        dt: 1.0,
                        duration: 86_400.0,
                        latitude,
                        day_of_year: doy,
                        ..Default::default()
                    };
                    let modules = sim_mmast::full_stack();
                    let mut solver = Solver::new(config, vehicle.clone(), modules);
                    solver.set_position([0.0, altitude, 0.0]);

                    let atm = StandardAtmosphere {
                        cloud_cover: cloud,
                        latitude_deg: latitude,
                        ..Default::default()
                    }
                    .with_shared_lut(atmo_lut.clone());

                    solver.bake_mmast_lut(&atm);
                    let series = solver.run(&atm, 86_400.0);
                    let last = series.snapshots.last().expect("at least one snapshot");

                    // Pull per-module accumulated_wh in the canonical module order.
                    let mut per_module = vec![0.0_f32; module_names_slice.len()];
                    for (j, name) in module_names_slice.iter().enumerate() {
                        if let Some(m) = last.modules.iter().find(|m| m.name.as_ref() == name) {
                            per_module[j] = m.accumulated_wh as f32;
                        }
                    }

                    let harvested: f64 = per_module.iter()
                        .filter(|&&v| v > 0.0)
                        .map(|&v| v as f64)
                        .sum();
                    let consumed: f64 = per_module.iter()
                        .filter(|&&v| v < 0.0)
                        .map(|&v| (-v) as f64)
                        .sum();

                    let flat_idx = ((li * cloud_bins + ci) * alt_bins + ai) * doy_bins + doy_i;
                    (
                        flat_idx,
                        EnvelopeCell {
                            harvested_wh: harvested,
                            consumed_wh: consumed,
                            margin_wh: harvested - consumed,
                        },
                        per_module,
                    )
                })
                .collect();

            for (idx, cell, per_module) in day_cells {
                cells[idx] = cell;
                let base = idx * num_modules;
                for (j, v) in per_module.into_iter().enumerate() {
                    module_contributions[base + j] = v;
                }
            }
        }

        Self {
            vehicle: vehicle_name,
            lat_min,
            lat_max,
            lat_bins,
            cloud_min,
            cloud_max,
            cloud_bins,
            alt_min,
            alt_max,
            alt_bins,
            doy_min,
            doy_max,
            doy_bins,
            module_names,
            cells,
            module_contributions,
        }
    }

    #[inline]
    pub fn cell_index(&self, li: usize, ci: usize, ai: usize, di: usize) -> usize {
        ((li * self.cloud_bins + ci) * self.alt_bins + ai) * self.doy_bins + di
    }

    #[inline]
    pub fn cell_at(&self, li: usize, ci: usize, ai: usize, di: usize) -> EnvelopeCell {
        self.cells[self.cell_index(li, ci, ai, di)]
    }

    /// Sample with quadrilinear interpolation.
    pub fn sample(&self, latitude: f64, cloud: f64, altitude_m: f64, day_of_year: u16) -> EnvelopeCell {
        let lat_f = if self.lat_bins > 1 {
            ((latitude - self.lat_min) / (self.lat_max - self.lat_min) * (self.lat_bins - 1) as f64)
                .clamp(0.0, (self.lat_bins - 1) as f64 - 1e-6)
        } else { 0.0 };
        let cloud_f = if self.cloud_bins > 1 {
            ((cloud - self.cloud_min) / (self.cloud_max - self.cloud_min) * (self.cloud_bins - 1) as f64)
                .clamp(0.0, (self.cloud_bins - 1) as f64 - 1e-6)
        } else { 0.0 };
        let alt_f = if self.alt_bins > 1 {
            ((altitude_m - self.alt_min) / (self.alt_max - self.alt_min) * (self.alt_bins - 1) as f64)
                .clamp(0.0, (self.alt_bins - 1) as f64 - 1e-6)
        } else { 0.0 };
        let doy_f = if self.doy_bins > 1 {
            ((day_of_year as f64 - self.doy_min as f64) / (self.doy_max - self.doy_min) as f64
                * (self.doy_bins - 1) as f64)
                .clamp(0.0, (self.doy_bins - 1) as f64 - 1e-6)
        } else { 0.0 };

        let l0 = lat_f.floor() as usize;
        let c0 = cloud_f.floor() as usize;
        let a0 = alt_f.floor() as usize;
        let d0 = doy_f.floor() as usize;
        let fl = lat_f - l0 as f64;
        let fc = cloud_f - c0 as f64;
        let fa = alt_f - a0 as f64;
        let fd = doy_f - d0 as f64;

        // 16 corner cells, then quad-linear interpolation.
        let mut corners = [EnvelopeCell::default(); 16];
        for (n, (dl, dc, da, dd)) in [
            (0,0,0,0),(1,0,0,0),(0,1,0,0),(1,1,0,0),
            (0,0,1,0),(1,0,1,0),(0,1,1,0),(1,1,1,0),
            (0,0,0,1),(1,0,0,1),(0,1,0,1),(1,1,0,1),
            (0,0,1,1),(1,0,1,1),(0,1,1,1),(1,1,1,1),
        ].iter().enumerate() {
            corners[n] = self.cell_at(l0 + dl, c0 + dc, a0 + da, d0 + dd);
        }

        let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
        let blend = |field: fn(&EnvelopeCell) -> f64| -> f64 {
            // Interpolate across lat, then cloud, then alt, then doy.
            let mut tmp = [0.0_f64; 8];
            for i in 0..8 {
                tmp[i] = lerp(field(&corners[2 * i]), field(&corners[2 * i + 1]), fl);
            }
            let mut tmp2 = [0.0_f64; 4];
            for i in 0..4 {
                tmp2[i] = lerp(tmp[2 * i], tmp[2 * i + 1], fc);
            }
            let x0 = lerp(tmp2[0], tmp2[1], fa);
            let x1 = lerp(tmp2[2], tmp2[3], fa);
            lerp(x0, x1, fd)
        };

        EnvelopeCell {
            harvested_wh: blend(|c| c.harvested_wh),
            consumed_wh: blend(|c| c.consumed_wh),
            margin_wh: blend(|c| c.margin_wh),
        }
    }

    /// Sample a 2D lat×cloud slice at a fixed altitude bin and day-of-year bin.
    pub fn slice(&self, ai: usize, di: usize) -> Vec<EnvelopeCell> {
        let mut out = Vec::with_capacity(self.lat_bins * self.cloud_bins);
        for li in 0..self.lat_bins {
            for ci in 0..self.cloud_bins {
                out.push(self.cell_at(li, ci, ai, di));
            }
        }
        out
    }

    pub fn to_csv(&self) -> String {
        let mut out = String::from("latitude,cloud_cover,altitude_m,day_of_year,harvested_wh,consumed_wh,margin_wh,feasible\n");
        let axis_val = |i: usize, min: f64, max: f64, bins: usize| {
            if bins > 1 { min + i as f64 * (max - min) / (bins - 1) as f64 } else { min }
        };
        let doy_at = |i: usize| {
            if self.doy_bins > 1 {
                (self.doy_min as f64
                    + i as f64 * (self.doy_max - self.doy_min) as f64 / (self.doy_bins - 1) as f64)
                    .round() as u16
            } else {
                self.doy_min
            }
        };
        for li in 0..self.lat_bins {
            for ci in 0..self.cloud_bins {
                for ai in 0..self.alt_bins {
                    for di in 0..self.doy_bins {
                        let lat = axis_val(li, self.lat_min, self.lat_max, self.lat_bins);
                        let cloud = axis_val(ci, self.cloud_min, self.cloud_max, self.cloud_bins);
                        let alt = axis_val(ai, self.alt_min, self.alt_max, self.alt_bins);
                        let doy = doy_at(di);
                        let cell = self.cell_at(li, ci, ai, di);
                        out.push_str(&format!(
                            "{:.2},{:.3},{:.0},{},{:.1},{:.1},{:.1},{}\n",
                            lat, cloud, alt, doy,
                            cell.harvested_wh, cell.consumed_wh, cell.margin_wh, cell.feasible()
                        ));
                    }
                }
            }
        }
        out
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    pub fn to_html(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_default();
        let template = include_str!("envelope_template.html");
        template.replace("__ENVELOPE_JSON__", &json)
    }

    pub fn feasibility_fraction(&self) -> f64 {
        let n = self.cells.len();
        if n == 0 { return 0.0; }
        let feasible = self.cells.iter().filter(|c| c.feasible()).count();
        feasible as f64 / n as f64
    }

    /// Max cloud tolerance at a given latitude, at a specific altitude + doy slice.
    pub fn max_cloud_at_latitude(&self, latitude: f64, altitude_m: f64, day_of_year: u16) -> Option<f64> {
        let mut best: Option<f64> = None;
        for ci in 0..self.cloud_bins {
            let cloud = if self.cloud_bins > 1 {
                self.cloud_min + ci as f64 * (self.cloud_max - self.cloud_min) / (self.cloud_bins - 1) as f64
            } else {
                self.cloud_min
            };
            let cell = self.sample(latitude, cloud, altitude_m, day_of_year);
            if cell.feasible() {
                best = Some(cloud);
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_4d_bake_and_sample() {
        let env = MissionEnvelope::bake(
            VehicleParams::hale_solar_uav(),
            0.0, 60.0, 4,       // lat
            0.0, 0.9, 4,        // cloud
            500.0, 10_000.0, 3, // altitude
            80, 260, 3,         // day of year (Mar-Sep)
        );
        assert_eq!(env.cells.len(), 4 * 4 * 3 * 3);

        // Clear sky, summer solstice (~172), low altitude, equator — feasible.
        let clear = env.sample(0.0, 0.0, 500.0, 172);
        assert!(clear.feasible(), "HALE clear equator summer should be feasible, got {clear:?}");

        // Overcast, winter (~80), high latitude — harder.
        let hard = env.sample(60.0, 0.9, 500.0, 80);
        assert!(hard.margin_wh < clear.margin_wh);
    }

    #[test]
    fn envelope_slice_shape() {
        let env = MissionEnvelope::bake(
            VehicleParams::hale_solar_uav(),
            0.0, 60.0, 4,
            0.0, 0.9, 4,
            500.0, 10_000.0, 2,
            172, 172, 1,
        );
        let slice = env.slice(0, 0);
        assert_eq!(slice.len(), 4 * 4);
    }
}
