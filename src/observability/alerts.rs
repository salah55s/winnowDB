//! Alerting rules for WinnowDB
//!
//! Provides configurable threshold-based alerts for monitoring.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::Mutex;
use wasm_bindgen::prelude::*;

/// Alert severity levels
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Alert condition types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AlertCondition {
    /// Memory usage exceeds threshold (percentage)
    MemoryAbove { percent: f64 },
    /// Error rate exceeds threshold (percentage)
    ErrorRateAbove { percent: f64 },
    /// Latency exceeds threshold (ms)
    LatencyAbove { ms: f64 },
    /// QPS exceeds threshold
    QpsAbove { qps: f64 },
    /// Circuit breaker is open
    CircuitBreakerOpen,
    /// Vector count exceeds threshold
    VectorCountAbove { count: u64 },
}

/// An alerting rule
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub enabled: bool,
    /// Cooldown in ms (to prevent alert storms)
    pub cooldown_ms: u64,
}

/// A fired alert
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FiredAlert {
    pub rule_id: String,
    pub rule_name: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: f64,
    pub value: f64,
}

/// Alert state for cooldown tracking
struct AlertState {
    last_fired: f64,
}

/// Alert engine
#[derive(Clone)]
pub struct AlertEngine {
    rules: Arc<Mutex<Vec<AlertRule>>>,
    states: Arc<Mutex<std::collections::HashMap<String, AlertState>>>,
    fired: Arc<Mutex<Vec<FiredAlert>>>,
    max_alerts: usize,
}

impl AlertEngine {
    pub fn new(max_alerts: usize) -> Self {
        Self {
            rules: Arc::new(Mutex::new(Vec::new())),
            states: Arc::new(Mutex::new(std::collections::HashMap::new())),
            fired: Arc::new(Mutex::new(Vec::new())),
            max_alerts,
        }
    }
    
    /// Add or update a rule
    pub fn add_rule(&self, rule: AlertRule) {
        let mut rules = self.rules.lock();
        if let Some(existing) = rules.iter_mut().find(|r| r.id == rule.id) {
            *existing = rule;
        } else {
            rules.push(rule);
        }
    }
    
    /// Remove a rule
    pub fn remove_rule(&self, id: &str) -> bool {
        let mut rules = self.rules.lock();
        let len_before = rules.len();
        rules.retain(|r| r.id != id);
        rules.len() < len_before
    }
    
    /// Check all rules against current metrics
    pub fn check(&self, metrics: &AlertMetrics) -> Vec<FiredAlert> {
        let rules = self.rules.lock();
        let mut states = self.states.lock();
        let mut fired = self.fired.lock();
        let now = js_sys::Date::now();
        let mut new_alerts = Vec::new();
        
        for rule in rules.iter() {
            if !rule.enabled {
                continue;
            }
            
            // Check cooldown
            if let Some(state) = states.get(&rule.id) {
                if now - state.last_fired < rule.cooldown_ms as f64 {
                    continue;
                }
            }
            
            // Evaluate condition
            let (triggered, value, message) = match &rule.condition {
                AlertCondition::MemoryAbove { percent } => {
                    let v = metrics.memory_percent;
                    (v > *percent, v, format!("Memory at {:.1}%", v))
                },
                AlertCondition::ErrorRateAbove { percent } => {
                    let v = metrics.error_rate * 100.0;
                    (v > *percent, v, format!("Error rate at {:.2}%", v))
                },
                AlertCondition::LatencyAbove { ms } => {
                    let v = metrics.avg_latency_ms;
                    (v > *ms, v, format!("Avg latency at {:.1}ms", v))
                },
                AlertCondition::QpsAbove { qps } => {
                    let v = metrics.qps;
                    (v > *qps, v, format!("QPS at {:.1}", v))
                },
                AlertCondition::CircuitBreakerOpen => {
                    (metrics.circuit_open, 1.0, "Circuit breaker is OPEN".to_string())
                },
                AlertCondition::VectorCountAbove { count } => {
                    let v = metrics.vector_count as f64;
                    (metrics.vector_count > *count, v, format!("Vector count at {}", metrics.vector_count))
                },
            };
            
            if triggered {
                let alert = FiredAlert {
                    rule_id: rule.id.clone(),
                    rule_name: rule.name.clone(),
                    severity: rule.severity,
                    message,
                    timestamp: now,
                    value,
                };
                
                states.insert(rule.id.clone(), AlertState { last_fired: now });
                
                if fired.len() >= self.max_alerts {
                    fired.remove(0);
                }
                fired.push(alert.clone());
                new_alerts.push(alert);
            }
        }
        
        new_alerts
    }
    
    /// Get recent alerts
    pub fn recent(&self, count: usize) -> Vec<FiredAlert> {
        let fired = self.fired.lock();
        fired.iter().rev().take(count).cloned().collect()
    }
    
    /// Get alerts by severity
    pub fn by_severity(&self, severity: AlertSeverity, count: usize) -> Vec<FiredAlert> {
        let fired = self.fired.lock();
        fired.iter()
            .filter(|a| a.severity == severity)
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
    
    /// Clear all alerts
    pub fn clear(&self) {
        self.fired.lock().clear();
    }
    
    /// List all rules
    pub fn list_rules(&self) -> Vec<AlertRule> {
        self.rules.lock().clone()
    }
}

impl Default for AlertEngine {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Metrics snapshot for alert evaluation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlertMetrics {
    pub memory_percent: f64,
    pub error_rate: f64,
    pub avg_latency_ms: f64,
    pub qps: f64,
    pub circuit_open: bool,
    pub vector_count: u64,
}

/// WASM-exposed alert manager
#[wasm_bindgen]
pub struct AlertManager {
    engine: AlertEngine,
}

#[wasm_bindgen]
impl AlertManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let engine = AlertEngine::default();
        
        // Add default rules
        engine.add_rule(AlertRule {
            id: "memory_warn".to_string(),
            name: "High Memory Usage".to_string(),
            condition: AlertCondition::MemoryAbove { percent: 80.0 },
            severity: AlertSeverity::Warning,
            enabled: true,
            cooldown_ms: 60_000,
        });
        
        engine.add_rule(AlertRule {
            id: "memory_crit".to_string(),
            name: "Critical Memory Usage".to_string(),
            condition: AlertCondition::MemoryAbove { percent: 95.0 },
            severity: AlertSeverity::Critical,
            enabled: true,
            cooldown_ms: 30_000,
        });
        
        engine.add_rule(AlertRule {
            id: "error_rate".to_string(),
            name: "High Error Rate".to_string(),
            condition: AlertCondition::ErrorRateAbove { percent: 5.0 },
            severity: AlertSeverity::Warning,
            enabled: true,
            cooldown_ms: 60_000,
        });
        
        engine.add_rule(AlertRule {
            id: "circuit_open".to_string(),
            name: "Circuit Breaker Open".to_string(),
            condition: AlertCondition::CircuitBreakerOpen,
            severity: AlertSeverity::Critical,
            enabled: true,
            cooldown_ms: 30_000,
        });
        
        Self { engine }
    }
    
    /// Add a custom rule
    pub fn add_rule(&self, rule: JsValue) -> Result<(), JsValue> {
        let r: AlertRule = serde_wasm_bindgen::from_value(rule)?;
        self.engine.add_rule(r);
        Ok(())
    }
    
    /// Remove a rule
    pub fn remove_rule(&self, id: &str) -> bool {
        self.engine.remove_rule(id)
    }
    
    /// Check alerts with current metrics
    pub fn check(&self, metrics: JsValue) -> Result<JsValue, JsValue> {
        let m: AlertMetrics = serde_wasm_bindgen::from_value(metrics)?;
        let alerts = self.engine.check(&m);
        Ok(serde_wasm_bindgen::to_value(&alerts)?)
    }
    
    /// Get recent alerts
    pub fn recent(&self, count: usize) -> JsValue {
        let alerts = self.engine.recent(count);
        serde_wasm_bindgen::to_value(&alerts).unwrap_or(JsValue::NULL)
    }
    
    /// Get critical alerts
    pub fn critical(&self, count: usize) -> JsValue {
        let alerts = self.engine.by_severity(AlertSeverity::Critical, count);
        serde_wasm_bindgen::to_value(&alerts).unwrap_or(JsValue::NULL)
    }
    
    /// List all rules
    pub fn list_rules(&self) -> JsValue {
        let rules = self.engine.list_rules();
        serde_wasm_bindgen::to_value(&rules).unwrap_or(JsValue::NULL)
    }
    
    /// Clear alerts
    pub fn clear(&self) {
        self.engine.clear();
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}
