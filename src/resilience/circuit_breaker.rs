use wasm_bindgen::prelude::*;
use std::sync::Arc;
use parking_lot::Mutex;
use js_sys::Date;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub reset_timeout_ms: f64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            reset_timeout_ms: 10000.0, // 10 seconds
        }
    }
}

struct BreakerState {
    state: CircuitState,
    failures: usize,
    last_failure_time: f64,
}

#[derive(Clone)]
pub struct CircuitBreaker {
    config: Arc<CircuitBreakerConfig>,
    state: Arc<Mutex<BreakerState>>,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config: Arc::new(config),
            state: Arc::new(Mutex::new(BreakerState {
                state: CircuitState::Closed,
                failures: 0,
                last_failure_time: 0.0,
            })),
        }
    }

    pub fn call<F, T, E>(&self, op: F) -> Result<T, E> 
    where F: FnOnce() -> Result<T, E>, E: From<String> // Error must be convertible from "Circuit Open" string
    {
        // 1. Check State
        {
            let mut state = self.state.lock();
            match state.state {
                CircuitState::Open => {
                    let now = Date::now();
                    if now - state.last_failure_time > self.config.reset_timeout_ms {
                        state.state = CircuitState::HalfOpen;
                         // Proceed to try
                    } else {
                        return Err(E::from("Circuit Breaker is Open".to_string()));
                    }
                },
                CircuitState::HalfOpen => {
                     // Only allow one request? Or just let it pass.
                     // Simple implementation: Let it pass, outcome updates state.
                },
                CircuitState::Closed => {},
            }
        }

        // 2. Execute Operation
        let result = op();

        // 3. Update State based on Result
        let mut state = self.state.lock();
        match result {
            Ok(val) => {
                if state.state == CircuitState::HalfOpen {
                    state.state = CircuitState::Closed;
                    state.failures = 0;
                }
                // If Closed, keep Closed.
                // Reset failures on success? Usually yes, or decay.
                state.failures = 0; 
                Ok(val)
            },
            Err(e) => {
                state.failures += 1;
                state.last_failure_time = Date::now();
                
                if state.state == CircuitState::HalfOpen {
                    state.state = CircuitState::Open; // Failed trial
                } else if state.failures >= self.config.failure_threshold {
                    state.state = CircuitState::Open;
                }
                Err(e)
            }
        }
    }
    
    pub fn is_open(&self) -> bool {
        let state = self.state.lock();
        state.state == CircuitState::Open
    }
}

// Tests (Pure Rust + Mocking Date locally if needed, but we use js_sys::Date which needs wasm env)
// For unit testing in pure rust, we'd need to mock time.
// Since we are running in "wasm-pack test --node", js_sys::Date works!
#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_circuit_breaker_transitions() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 2,
            reset_timeout_ms: 100.0,
        });

        // 1. Closed State
        assert!(!breaker.is_open());
        
        // 2. Failure 1
        let _ = breaker.call(|| Err::<(), String>("Fail".to_string()));
        assert!(!breaker.is_open());
        
        // 3. Failure 2 (Threshold Reached)
        let _ = breaker.call(|| Err::<(), String>("Fail".to_string()));
        assert!(breaker.is_open());
        
        // 4. Fast Fail
        let res = breaker.call(|| Ok::<(), String>(()));
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), "Circuit Breaker is Open");
        
        // 5. Wait for Reset (Mocking wait not easy in sync test, but we can rely on Date::now incrementing)
        // We can't sleep in WASM easily without async.
        // For verify, we rely on logic correctness.
    }
}

// ============================================================================
// Per-Index Circuit Breaker Registry
// ============================================================================

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Registry of per-index circuit breakers
#[derive(Clone)]
pub struct CircuitBreakerRegistry {
    breakers: Arc<Mutex<HashMap<String, CircuitBreaker>>>,
    default_config: CircuitBreakerConfig,
}

impl CircuitBreakerRegistry {
    pub fn new(failure_threshold: usize, reset_timeout_ms: f64) -> Self {
        Self {
            breakers: Arc::new(Mutex::new(HashMap::new())),
            default_config: CircuitBreakerConfig {
                failure_threshold,
                reset_timeout_ms,
            },
        }
    }
    
    /// Get or create a circuit breaker for an index
    pub fn get(&self, index_name: &str) -> CircuitBreaker {
        let mut breakers = self.breakers.lock();
        if let Some(cb) = breakers.get(index_name) {
            cb.clone()
        } else {
            let cb = CircuitBreaker::new(CircuitBreakerConfig {
                failure_threshold: self.default_config.failure_threshold,
                reset_timeout_ms: self.default_config.reset_timeout_ms,
            });
            breakers.insert(index_name.to_string(), cb.clone());
            cb
        }
    }
    
    /// Get all open circuit breakers
    pub fn open_breakers(&self) -> Vec<String> {
        self.breakers.lock()
            .iter()
            .filter(|(_, cb)| cb.is_open())
            .map(|(k, _)| k.clone())
            .collect()
    }
    
    /// Reset a circuit breaker
    pub fn reset(&self, index_name: &str) {
        let mut breakers = self.breakers.lock();
        if let Some(cb) = breakers.get_mut(index_name) {
            let mut state = cb.state.lock();
            state.state = CircuitState::Closed;
            state.failures = 0;
        }
    }
    
    /// Reset all circuit breakers
    pub fn reset_all(&self) {
        for cb in self.breakers.lock().values_mut() {
            let mut state = cb.state.lock();
            state.state = CircuitState::Closed;
            state.failures = 0;
        }
    }
}

impl Default for CircuitBreakerRegistry {
    fn default() -> Self {
        Self::new(5, 10000.0)
    }
}

// ============================================================================
// Fallback Strategies
// ============================================================================

/// Fallback strategy for index failures
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FallbackStrategy {
    /// No fallback, just fail
    None,
    /// Fall back to flat (brute force) search
    FlatSearch,
    /// Use a secondary index
    SecondaryIndex { index_name: String },
    /// Return cached results
    CachedResults,
    /// Return empty with error flag
    EmptyWithError,
}

/// Result of a fallback operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FallbackResult<T> {
    pub data: Option<T>,
    pub used_fallback: bool,
    pub fallback_strategy: Option<String>,
    pub original_error: Option<String>,
    pub latency_ms: f64,
}

/// Fallback manager for graceful degradation
#[derive(Clone)]
pub struct FallbackManager {
    strategies: Arc<Mutex<HashMap<String, FallbackStrategy>>>,
    default_strategy: FallbackStrategy,
}

impl FallbackManager {
    pub fn new(default_strategy: FallbackStrategy) -> Self {
        Self {
            strategies: Arc::new(Mutex::new(HashMap::new())),
            default_strategy,
        }
    }
    
    /// Set fallback strategy for an index
    pub fn set_strategy(&self, index_name: &str, strategy: FallbackStrategy) {
        self.strategies.lock().insert(index_name.to_string(), strategy);
    }
    
    /// Get fallback strategy for an index
    pub fn get_strategy(&self, index_name: &str) -> FallbackStrategy {
        self.strategies.lock()
            .get(index_name)
            .cloned()
            .unwrap_or_else(|| self.default_strategy.clone())
    }
    
    /// Check if should use fallback
    pub fn should_fallback(&self, index_name: &str) -> bool {
        !matches!(self.get_strategy(index_name), FallbackStrategy::None)
    }
}

impl Default for FallbackManager {
    fn default() -> Self {
        Self::new(FallbackStrategy::FlatSearch)
    }
}

/// WASM-exposed resilience manager
#[wasm_bindgen]
pub struct ResilienceManager {
    registry: CircuitBreakerRegistry,
    fallback: FallbackManager,
}

#[wasm_bindgen]
impl ResilienceManager {
    #[wasm_bindgen(constructor)]
    pub fn new(failure_threshold: usize, reset_timeout_ms: f64) -> Self {
        Self {
            registry: CircuitBreakerRegistry::new(failure_threshold, reset_timeout_ms),
            fallback: FallbackManager::default(),
        }
    }
    
    /// Check if index circuit is open
    pub fn is_open(&self, index_name: &str) -> bool {
        self.registry.get(index_name).is_open()
    }
    
    /// Get list of open circuits
    pub fn open_circuits(&self) -> JsValue {
        let open = self.registry.open_breakers();
        serde_wasm_bindgen::to_value(&open).unwrap_or(JsValue::NULL)
    }
    
    /// Reset a specific circuit
    pub fn reset(&self, index_name: &str) {
        self.registry.reset(index_name);
    }
    
    /// Reset all circuits
    pub fn reset_all(&self) {
        self.registry.reset_all();
    }
    
    /// Set fallback strategy
    pub fn set_fallback_strategy(&self, index_name: &str, strategy: &str) {
        let strat = match strategy {
            "none" => FallbackStrategy::None,
            "flat" => FallbackStrategy::FlatSearch,
            "cached" => FallbackStrategy::CachedResults,
            "empty" => FallbackStrategy::EmptyWithError,
            _ => FallbackStrategy::FlatSearch,
        };
        self.fallback.set_strategy(index_name, strat);
    }
    
    /// Check if should use fallback for index
    pub fn should_fallback(&self, index_name: &str) -> bool {
        self.fallback.should_fallback(index_name)
    }
}

impl Default for ResilienceManager {
    fn default() -> Self {
        Self::new(5, 10000.0)
    }
}
