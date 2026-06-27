//! `sim-report` — Analytical reporting for MMAST simulations.
//!
//! Consumes `SimTimeSeries` and produces:
//! - Energy balance summary (harvest / demand / margin over 24h)
//! - Per-module contribution breakdown
//! - Sensitivity analysis (sweep one parameter, report margin vs. parameter)
//! - Mission feasibility verdict (can it fly indefinitely? how many overcast days?)
//! - Mission envelope LUTs (2D feasibility grids for dashboards)
//! - Exportable JSON/CSV for external dashboards

pub mod envelope;
pub mod fleet;

use serde::{Deserialize, Serialize};
use sim_core::state::SimTimeSeries;

/// 24-hour energy summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergySummary {
    pub total_harvested_wh: f64,
    pub total_consumed_wh: f64,
    pub net_margin_wh: f64,
    pub margin_ratio: f64,
    pub peak_harvest_w: f64,
    pub peak_demand_w: f64,
    pub min_battery_soc_wh: f64,
    pub max_battery_soc_wh: f64,
    pub per_module: Vec<ModuleContribution>,
    pub feasible: bool,
    pub max_overcast_days: u32,
}

/// Per-module contribution in a summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleContribution {
    pub name: String,
    pub total_wh: f64,
    pub fraction_of_harvest: f64,
    pub peak_w: f64,
    pub active_hours: f64,
}

/// Generate an energy summary from a completed simulation.
pub fn summarize(series: &SimTimeSeries) -> EnergySummary {
    let snapshots = &series.snapshots;
    if snapshots.is_empty() {
        return EnergySummary {
            total_harvested_wh: 0.0,
            total_consumed_wh: 0.0,
            net_margin_wh: 0.0,
            margin_ratio: 0.0,
            peak_harvest_w: 0.0,
            peak_demand_w: 0.0,
            min_battery_soc_wh: 0.0,
            max_battery_soc_wh: 0.0,
            per_module: vec![],
            feasible: false,
            max_overcast_days: 0,
        };
    }

    let last = snapshots.last().unwrap();
    let peak_harvest = snapshots.iter().map(|s| s.total_harvest_w).fold(0.0_f64, f64::max);
    let peak_demand = snapshots.iter().map(|s| s.total_demand_w).fold(0.0_f64, f64::max);
    let min_soc = snapshots.iter().map(|s| s.battery_soc_wh).fold(f64::MAX, f64::min);
    let max_soc = snapshots.iter().map(|s| s.battery_soc_wh).fold(0.0_f64, f64::max);

    // Per-module from last snapshot's accumulated values.
    let total_harvest: f64 = last
        .modules
        .iter()
        .filter(|m| m.accumulated_wh > 0.0)
        .map(|m| m.accumulated_wh)
        .sum();

    let per_module: Vec<ModuleContribution> = last
        .modules
        .iter()
        .map(|m| ModuleContribution {
            name: m.name.to_string(),
            total_wh: m.accumulated_wh,
            fraction_of_harvest: if total_harvest > 0.0 {
                m.accumulated_wh.max(0.0) / total_harvest
            } else {
                0.0
            },
            peak_w: snapshots
                .iter()
                .flat_map(|s| s.modules.iter())
                .filter(|sm| sm.name == m.name)
                .map(|sm| sm.power_w.abs())
                .fold(0.0_f64, f64::max),
            active_hours: 0.0, // TODO: compute from time-series
        })
        .collect();

    let total_consumed: f64 = last
        .modules
        .iter()
        .filter(|m| m.accumulated_wh < 0.0)
        .map(|m| m.accumulated_wh.abs())
        .sum();

    let net_margin = total_harvest - total_consumed;
    let margin_ratio = if total_consumed > 0.0 {
        net_margin / total_consumed
    } else {
        f64::INFINITY
    };

    // How many consecutive overcast days (75% solar reduction) the margin covers.
    let daily_demand = total_consumed; // from the sim duration
    let daily_harvest_overcast = total_harvest * 0.25;
    let max_overcast = if daily_harvest_overcast >= daily_demand {
        u32::MAX // can fly indefinitely even overcast
    } else if net_margin > 0.0 {
        let deficit_per_day = daily_demand - daily_harvest_overcast;
        let surplus = net_margin; // accumulated surplus from clear day
        (surplus / deficit_per_day).floor() as u32
    } else {
        0
    };

    EnergySummary {
        total_harvested_wh: total_harvest,
        total_consumed_wh: total_consumed,
        net_margin_wh: net_margin,
        margin_ratio,
        peak_harvest_w: peak_harvest,
        peak_demand_w: peak_demand,
        min_battery_soc_wh: min_soc,
        max_battery_soc_wh: max_soc,
        per_module,
        feasible: net_margin > 0.0,
        max_overcast_days: max_overcast,
    }
}

/// Export the time series to JSON.
pub fn to_json(series: &SimTimeSeries) -> String {
    serde_json::to_string_pretty(series).unwrap_or_default()
}
