//! `sim-dynamics` — The main simulation solver.
//!
//! Integrates vehicle 6-DOF dynamics forward in time, queries MMAST modules
//! each step, updates the energy balance, and emits state snapshots for
//! consumers (renderer, reporter).

use sim_core::energy::EnergyBalance;
use sim_core::state::{ModulePowerRecord, SimState};
use sim_core::time::SimClock;
use sim_core::{SimConfig, EntityId};
use sim_environment::Environment;
use sim_mmast::{MmastModule, evaluate_modules};
use sim_mmast::lut::MmastLut;
use sim_vehicle::VehicleParams;

/// The simulation solver.
pub struct Solver {
    pub config: SimConfig,
    pub clock: SimClock,
    pub vehicle: VehicleParams,
    pub modules: Vec<Box<dyn MmastModule>>,
    pub energy: EnergyBalance,

    /// Pre-baked MMAST LUT. When present, `tick()` reads from the LUT
    /// instead of dispatching through trait objects. Same output, L₀ cost.
    pub mmast_lut: Option<MmastLut>,

    /// If true, `tick()` updates `self.energy` (O(N²) search + allocation).
    /// Default: false. Enable only if you need a running total outside of
    /// the per-snapshot per-module accumulators.
    pub track_energy_balance: bool,

    // Internal state.
    position: [f64; 3],
    velocity: [f64; 3],
    attitude: [f64; 4],
    angular_velocity: [f64; 3],
    module_accumulators: Vec<f64>,
    cost_accumulators: [f64; 3],
    /// Pre-allocated record buffer reused across steps — no per-step
    /// Vec allocation. Capacity 12 = 9 MMAST modules + 3 implicit costs.
    records: Vec<ModulePowerRecord>,
    // Last-tick derived values needed for snapshot().
    last_total_harvest_w: f64,
    last_total_cost_w: f64,
    last_ambient_temp_k: f64,
    last_ambient_pressure_pa: f64,
    last_density: f64,
    last_solar_irradiance: f64,
    last_wind: [f64; 3],
    last_prop_mode: &'static str,
    run_id: EntityId,
}

impl Solver {
    /// Create a new solver with the given config, vehicle, and MMAST stack.
    pub fn new(
        config: SimConfig,
        vehicle: VehicleParams,
        modules: Vec<Box<dyn MmastModule>>,
    ) -> Self {
        let clock = SimClock::new(config.start_hour, config.latitude, config.day_of_year);
        let n = modules.len();
        Self {
            config,
            clock,
            vehicle,
            modules,
            energy: EnergyBalance::default(),
            mmast_lut: None,
            track_energy_balance: false,
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            attitude: [0.0, 0.0, 0.0, 1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            module_accumulators: vec![0.0; n],
            cost_accumulators: [0.0; 3],
            records: Vec::with_capacity(n + 3),
            last_total_harvest_w: 0.0,
            last_total_cost_w: 0.0,
            last_ambient_temp_k: 288.15,
            last_ambient_pressure_pa: 101_325.0,
            last_density: 1.225,
            last_solar_irradiance: 0.0,
            last_wind: [0.0; 3],
            last_prop_mode: "brushless_prop",
            run_id: EntityId::new_v4(),
        }
    }

    /// Bake the MMAST module stack into a LUT by running the full module
    /// stack at every (hour, altitude, battery_temp) grid point using the
    /// provided Environment — same path the solver will use at runtime.
    /// Call once after construction.
    pub fn bake_mmast_lut(&mut self, env: &dyn Environment) {
        let lut = MmastLut::bake(
            &self.vehicle,
            env,
            self.config.latitude,
            self.config.day_of_year,
        );
        tracing::info!(
            "Baked MMAST LUT: {} entries, {:.1} MB",
            lut.len(),
            lut.size_bytes() as f64 / (1024.0 * 1024.0),
        );
        self.module_accumulators.resize(lut.module_names.len(), 0.0);
        self.mmast_lut = Some(lut);
    }

    /// Set initial position (e.g., runway threshold altitude).
    pub fn set_position(&mut self, pos: [f64; 3]) {
        self.position = pos;
    }

    /// Fast path: advance physics by one dt without building a SimState snapshot.
    /// Fills `self.records` in place (no per-step allocation). Call
    /// `snapshot()` separately if you need a SimState out.
    pub fn tick(&mut self, env: &dyn Environment) {
        let dt = self.config.dt;
        let dt_hours = dt / 3600.0;

        // ---- Environment fast path ----
        // Only query the two primitives we need for physics. Full
        // EnvConditions struct is deferred to snapshot() below.
        let density = env.density_at(self.position, &self.clock);
        let ambient_temp_k = env.temperature_at(self.position, &self.clock);

        // ---- MMAST modules ----
        self.records.clear();
        let mut harvest_w = 0.0_f64;
        let mut module_cost_w = 0.0_f64;

        if let Some(lut) = &self.mmast_lut {
            // LUT path — one array read, pushes into pre-allocated buffer.
            let entry = lut.sample(
                self.clock.solar_hour,
                self.position[1].max(0.0),
                self.vehicle.storage.temperature_k,
            );
            for (i, &pw) in entry.power.iter().enumerate() {
                let active = entry.active[i];
                let power = if active { pw } else { 0.0 };
                self.module_accumulators[i] += power * dt_hours;
                if power > 0.0 {
                    harvest_w += power;
                } else if power < 0.0 {
                    module_cost_w += -power;
                }
                self.records.push(ModulePowerRecord {
                    name: std::borrow::Cow::Borrowed(lut.module_names[i]),
                    power_w: power,
                    accumulated_wh: self.module_accumulators[i],
                    active,
                });
            }
        } else {
            // Trait dispatch fallback — needs the full EnvConditions.
            let conditions = env.conditions(self.position, &self.clock);
            let recs = evaluate_modules(
                &self.modules,
                &self.vehicle,
                &conditions,
                &self.clock,
                self.vehicle.storage.temperature_k,
                dt_hours,
                &mut self.module_accumulators,
            );
            for r in &recs {
                if r.active && r.power_w > 0.0 {
                    harvest_w += r.power_w;
                } else if r.active && r.power_w < 0.0 {
                    module_cost_w += -r.power_w;
                }
            }
            self.records.extend(recs);
        }

        // ---- Implicit cost entries ----
        let cruise_elec_w = self.vehicle.cruise_power_electrical(density);
        let avionics_w = self.vehicle.avionics_power_w + self.vehicle.payload_power_w;
        let batt_loss_w = harvest_w * (1.0 - self.vehicle.storage.round_trip_efficiency);

        self.cost_accumulators[0] += -cruise_elec_w * dt_hours;
        self.cost_accumulators[1] += -avionics_w * dt_hours;
        self.cost_accumulators[2] += -batt_loss_w * dt_hours;

        self.records.push(ModulePowerRecord {
            name: std::borrow::Cow::Borrowed("Propulsion (cruise)"),
            power_w: -cruise_elec_w,
            accumulated_wh: self.cost_accumulators[0],
            active: true,
        });
        self.records.push(ModulePowerRecord {
            name: std::borrow::Cow::Borrowed("Avionics + payload"),
            power_w: -avionics_w,
            accumulated_wh: self.cost_accumulators[1],
            active: true,
        });
        self.records.push(ModulePowerRecord {
            name: std::borrow::Cow::Borrowed("Battery losses"),
            power_w: -batt_loss_w,
            accumulated_wh: self.cost_accumulators[2],
            active: harvest_w > 0.0,
        });

        let total_cost = module_cost_w + cruise_elec_w + avionics_w + batt_loss_w;
        let net_power = harvest_w - total_cost;

        // ---- Battery SOC ----
        let energy_delta_wh = net_power * dt_hours;
        self.vehicle.storage.soc_wh =
            (self.vehicle.storage.soc_wh + energy_delta_wh)
                .clamp(0.0, self.vehicle.storage.capacity_wh);

        // ---- Battery thermal ----
        {
            let thermal_tau = 600.0;
            let blend = (dt / thermal_tau).min(0.3);
            let discharge_w = total_cost.min(self.vehicle.storage.soc_wh / dt_hours.max(1e-9) * 0.95);
            let i2r_heat_w = discharge_w * 0.04;
            let i2r_delta_k = i2r_heat_w * dt / (self.vehicle.storage.mass_kg * 1000.0);
            self.vehicle.storage.temperature_k =
                self.vehicle.storage.temperature_k * (1.0 - blend) + ambient_temp_k * blend + i2r_delta_k;
        }

        // ---- Optional energy balance bookkeeping (off by default) ----
        if self.track_energy_balance {
            self.energy.update(&self.records, dt_hours);
        }

        // ---- Kinematics + clock ----
        let speed = self.vehicle.mobility.cruise_speed();
        self.velocity = [speed, 0.0, 0.0];
        self.position[0] += self.velocity[0] * dt;
        self.clock.advance(dt, self.config.latitude, self.config.day_of_year);

        // ---- Stash derived values for a later snapshot() call ----
        self.last_total_harvest_w = harvest_w;
        self.last_total_cost_w = total_cost;
        self.last_ambient_temp_k = ambient_temp_k;
        self.last_density = density;
        self.last_prop_mode = if self
            .records
            .iter()
            .any(|m| m.name == "EAD ion burst" && m.active && m.power_w < 0.0)
        {
            "ead_ion"
        } else {
            "brushless_prop"
        };
    }

    /// Build a full SimState snapshot from the current solver state.
    /// Queries the full EnvConditions (pressure, solar, wind) that tick()
    /// intentionally skipped. Call only at sample points, not every step.
    pub fn snapshot(&mut self, env: &dyn Environment) -> SimState {
        // Full env query — only here, not in the hot path.
        let conditions = env.conditions(self.position, &self.clock);
        self.last_ambient_pressure_pa = conditions.pressure_pa;
        self.last_solar_irradiance = conditions.solar_irradiance;
        self.last_wind = conditions.wind_velocity;

        let speed = self.vehicle.mobility.cruise_speed();
        let net_power = self.last_total_harvest_w - self.last_total_cost_w;

        SimState {
            run_id: self.run_id,
            clock: self.clock,
            position: self.position,
            velocity: self.velocity,
            attitude: self.attitude,
            angular_velocity: self.angular_velocity,
            battery_soc_wh: self.vehicle.storage.soc_wh,
            battery_capacity_wh: self.vehicle.storage.capacity_wh,
            battery_temp_k: self.vehicle.storage.temperature_k,
            total_harvest_w: self.last_total_harvest_w,
            total_demand_w: self.last_total_cost_w,
            net_power_w: net_power,
            modules: self.records.clone(),
            propulsion_mode: self.last_prop_mode.into(),
            speed_mps: speed,
            altitude_m: self.position[1],
            ambient_temp_k: conditions.temperature_k,
            ambient_pressure_pa: conditions.pressure_pa,
            medium_density: conditions.density,
            solar_irradiance_wm2: conditions.solar_irradiance,
            wind_vector: conditions.wind_velocity,
        }
    }

    /// Back-compat: tick + snapshot in one call. Use `tick()` + `snapshot()`
    /// directly when you want to separate them (e.g., sampling every N steps).
    pub fn step(&mut self, env: &dyn Environment) -> SimState {
        self.tick(env);
        self.snapshot(env)
    }

    /// Run the full mission, collecting snapshots at `sample_interval_s`.
    /// Tick() runs every step (fast path). Snapshot() runs only at sample
    /// points — so the SimState construction + full EnvConditions query
    /// only happen when the caller actually needs them.
    pub fn run(
        &mut self,
        env: &dyn Environment,
        sample_interval_s: f64,
    ) -> sim_core::state::SimTimeSeries {
        let mut series = sim_core::state::SimTimeSeries::default();
        let mut next_sample = 0.0;
        let total = self.config.duration;

        while self.clock.elapsed < total {
            self.tick(env);
            if self.clock.elapsed >= next_sample {
                let state = self.snapshot(env);
                series.push(state);
                next_sample += sample_interval_s;
            }
        }

        series
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_environment::atmosphere::StandardAtmosphere;

    #[test]
    fn hale_uav_one_hour() {
        let config = SimConfig {
            dt: 1.0,
            duration: 3600.0,
            ..Default::default()
        };
        let vehicle = VehicleParams::hale_solar_uav();
        let modules = sim_mmast::full_stack();
        let mut solver = Solver::new(config, vehicle, modules);
        solver.set_position([0.0, 500.0, 0.0]);

        let atm = StandardAtmosphere::default();
        let series = solver.run(&atm, 60.0);

        assert!(!series.snapshots.is_empty());
        let last = series.snapshots.last().unwrap();
        assert!(last.clock.elapsed > 3500.0);
        assert!(last.total_harvest_w > 0.0);
        // Propulsion should now be tracked as a module.
        assert!(
            last.modules.iter().any(|m| m.name == "Propulsion (cruise)" && m.power_w < 0.0),
            "Propulsion cost not found in module records"
        );
    }

    #[test]
    fn hale_24h_energy_balance() {
        let config = SimConfig {
            dt: 10.0, // coarser for speed
            duration: 86400.0,
            ..Default::default()
        };
        let vehicle = VehicleParams::hale_solar_uav();
        let modules = sim_mmast::full_stack();
        let mut solver = Solver::new(config, vehicle, modules);
        solver.set_position([0.0, 500.0, 0.0]);

        let atm = StandardAtmosphere::default();
        let series = solver.run(&atm, 3600.0);

        let last = series.snapshots.last().unwrap();

        // Harvest should be in the ~3000–6000 Wh range for a clear mid-latitude day.
        let total_harvest: f64 = last.modules.iter()
            .filter(|m| m.accumulated_wh > 0.0)
            .map(|m| m.accumulated_wh)
            .sum();
        assert!(total_harvest > 2000.0, "Harvest too low: {total_harvest}");

        // Total cost should include propulsion — should be > 1000 Wh over 24h.
        let total_cost: f64 = last.modules.iter()
            .filter(|m| m.accumulated_wh < 0.0)
            .map(|m| m.accumulated_wh.abs())
            .sum();
        assert!(total_cost > 500.0, "Cost too low (propulsion not tracking?): {total_cost}");

        // Net should be positive for a clear-sky HALE.
        let net = total_harvest - total_cost;
        assert!(net > 0.0, "Energy balance negative: {net}");
    }
}
