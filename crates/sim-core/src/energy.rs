//! Energy balance — the central inequality that every vehicle must satisfy.
//!
//! ∫ harvest(t) dt  ≥  ∫ demand(t) dt  +  losses
//!
//! This module provides the accumulator and feasibility check.

use crate::state::ModulePowerRecord;
use serde::{Deserialize, Serialize};

/// Running energy balance accumulator.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnergyBalance {
    /// Total energy harvested so far (Wh).
    pub harvested_wh: f64,
    /// Total energy consumed so far (Wh).
    pub consumed_wh: f64,
    /// Total storage losses (Wh).
    pub storage_losses_wh: f64,
    /// Per-module accumulators keyed by module name.
    pub per_module: Vec<(String, f64)>,
}

impl EnergyBalance {
    /// Update from a set of module records over a time step `dt_hours`.
    pub fn update(&mut self, modules: &[ModulePowerRecord], dt_hours: f64) {
        for m in modules {
            if !m.active {
                continue;
            }
            let energy = m.power_w * dt_hours;
            if energy >= 0.0 {
                self.harvested_wh += energy;
            } else {
                self.consumed_wh += energy.abs();
            }
            // Update per-module accumulator.
            if let Some(entry) = self.per_module.iter_mut().find(|(n, _)| n.as_str() == m.name.as_ref()) {
                entry.1 += energy;
            } else {
                self.per_module.push((m.name.as_ref().to_string(), energy));
            }
        }
    }

    /// Net margin (positive = surplus).
    pub fn net_margin_wh(&self) -> f64 {
        self.harvested_wh - self.consumed_wh - self.storage_losses_wh
    }

    /// Is the energy balance feasible (net positive)?
    pub fn is_feasible(&self) -> bool {
        self.net_margin_wh() > 0.0
    }

    /// Margin as a fraction of consumed energy.
    pub fn margin_ratio(&self) -> f64 {
        if self.consumed_wh <= 0.0 {
            return f64::INFINITY;
        }
        self.net_margin_wh() / self.consumed_wh
    }
}
