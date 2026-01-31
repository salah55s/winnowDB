use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::Mutex;
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

/// A slow query entry for debugging
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlowQuery {
    pub timestamp: f64,
    pub latency_ms: f64,
    pub k: usize,
    pub filtered: bool,
}

/// Health status for the collection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub memory_usage_bytes: u64,
    pub memory_limit_bytes: u64,
    pub memory_percent: f64,
    pub vector_count: u64,
    pub active_writes: u64,
    pub circuit_breaker_open: bool,
    pub qps: f64,
    pub avg_latency_ms: f64,
    pub error_rate: f64,
}

#[derive(Clone, Debug)]
pub struct Metrics {
    pub search_count: Arc<AtomicU64>,
    pub insert_count: Arc<AtomicU64>,
    pub delete_count: Arc<AtomicU64>,
    pub error_count: Arc<AtomicU64>,
    
    // Gauges
    pub memory_usage: Arc<AtomicU64>,
    pub active_writes: Arc<AtomicU64>,
    pub vector_count: Arc<AtomicU64>,
    
    // Histograms (Simple Sum/Count for Avg, or Buckets if we want more)
    // For now, tracking Sum + Count allows calculating Average Latency
    pub search_latency_sum_us: Arc<AtomicU64>,
    pub search_latency_count: Arc<AtomicU64>,
    
    // Slow query log (ring buffer)
    slow_queries: Arc<Mutex<Vec<SlowQuery>>>,
    slow_query_threshold_ms: f64,
    
    // For QPS calculation
    start_time: f64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            search_count: Arc::new(AtomicU64::new(0)),
            insert_count: Arc::new(AtomicU64::new(0)),
            delete_count: Arc::new(AtomicU64::new(0)),
            error_count: Arc::new(AtomicU64::new(0)),
            memory_usage: Arc::new(AtomicU64::new(0)),
            active_writes: Arc::new(AtomicU64::new(0)),
            vector_count: Arc::new(AtomicU64::new(0)),
            search_latency_sum_us: Arc::new(AtomicU64::new(0)),
            search_latency_count: Arc::new(AtomicU64::new(0)),
            slow_queries: Arc::new(Mutex::new(Vec::with_capacity(100))),
            slow_query_threshold_ms: 100.0, // Default 100ms threshold
            start_time: js_sys::Date::now(),
        }
    }
    
    /// Create metrics with custom slow query threshold
    pub fn with_slow_threshold(threshold_ms: f64) -> Self {
        let mut m = Self::new();
        m.slow_query_threshold_ms = threshold_ms;
        m
    }

    pub fn increment_search(&self) {
        self.search_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn increment_insert(&self) {
        self.insert_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn increment_delete(&self) {
        self.delete_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn increment_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_memory(&self, bytes: u64) {
        self.memory_usage.store(bytes, Ordering::Relaxed);
    }
    
    pub fn set_vector_count(&self, count: u64) {
        self.vector_count.store(count, Ordering::Relaxed);
    }

    pub fn observe_search_latency(&self, us: u64) {
        self.search_latency_sum_us.fetch_add(us, Ordering::Relaxed);
        self.search_latency_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record a slow query if it exceeds the threshold
    pub fn record_slow_query(&self, latency_ms: f64, k: usize, filtered: bool) {
        if latency_ms >= self.slow_query_threshold_ms {
            let entry = SlowQuery {
                timestamp: js_sys::Date::now(),
                latency_ms,
                k,
                filtered,
            };
            
            let mut log = self.slow_queries.lock();
            if log.len() >= 100 {
                log.remove(0); // Ring buffer behavior
            }
            log.push(entry);
        }
    }
    
    /// Get slow queries (most recent first)
    pub fn get_slow_queries(&self) -> Vec<SlowQuery> {
        let log = self.slow_queries.lock();
        let mut result = log.clone();
        result.reverse();
        result
    }
    
    /// Calculate current QPS (queries per second)
    pub fn qps(&self) -> f64 {
        let elapsed_s = (js_sys::Date::now() - self.start_time) / 1000.0;
        if elapsed_s > 0.0 {
            self.search_count.load(Ordering::Relaxed) as f64 / elapsed_s
        } else {
            0.0
        }
    }
    
    /// Calculate average latency in milliseconds
    pub fn avg_latency_ms(&self) -> f64 {
        let sum = self.search_latency_sum_us.load(Ordering::Relaxed);
        let count = self.search_latency_count.load(Ordering::Relaxed);
        if count > 0 {
            (sum as f64 / count as f64) / 1000.0 // Convert us to ms
        } else {
            0.0
        }
    }
    
    /// Calculate error rate
    pub fn error_rate(&self) -> f64 {
        let total = self.search_count.load(Ordering::Relaxed) 
            + self.insert_count.load(Ordering::Relaxed)
            + self.delete_count.load(Ordering::Relaxed);
        let errors = self.error_count.load(Ordering::Relaxed);
        if total > 0 {
            errors as f64 / total as f64
        } else {
            0.0
        }
    }
    
    /// Get detailed health status
    pub fn health_status(&self, memory_limit: u64, circuit_open: bool) -> HealthStatus {
        let mem = self.memory_usage.load(Ordering::Relaxed);
        let mem_percent = if memory_limit > 0 {
            (mem as f64 / memory_limit as f64) * 100.0
        } else {
            0.0
        };
        
        let status = if circuit_open {
            "Unhealthy".to_string()
        } else if mem_percent > 90.0 {
            "Degraded".to_string()
        } else {
            "Healthy".to_string()
        };
        
        HealthStatus {
            status,
            memory_usage_bytes: mem,
            memory_limit_bytes: memory_limit,
            memory_percent: mem_percent,
            vector_count: self.vector_count.load(Ordering::Relaxed),
            active_writes: self.active_writes.load(Ordering::Relaxed),
            circuit_breaker_open: circuit_open,
            qps: self.qps(),
            avg_latency_ms: self.avg_latency_ms(),
            error_rate: self.error_rate(),
        }
    }
    
    pub fn scrape(&self) -> String {
        let mut buffer = String::new();
        
        // Counters
        buffer.push_str("# HELP winnow_search_total Total number of search operations.\n");
        buffer.push_str("# TYPE winnow_search_total counter\n");
        buffer.push_str(&format!("winnow_search_total {}\n", self.search_count.load(Ordering::Relaxed)));

        buffer.push_str("# HELP winnow_insert_total Total number of insert operations.\n");
        buffer.push_str("# TYPE winnow_insert_total counter\n");
        buffer.push_str(&format!("winnow_insert_total {}\n", self.insert_count.load(Ordering::Relaxed)));

        buffer.push_str("# HELP winnow_delete_total Total number of delete operations.\n");
        buffer.push_str("# TYPE winnow_delete_total counter\n");
        buffer.push_str(&format!("winnow_delete_total {}\n", self.delete_count.load(Ordering::Relaxed)));

        buffer.push_str("# HELP winnow_errors_total Total number of failed operations.\n");
        buffer.push_str("# TYPE winnow_errors_total counter\n");
        buffer.push_str(&format!("winnow_errors_total {}\n", self.error_count.load(Ordering::Relaxed)));

        // Gauges
        buffer.push_str("# HELP winnow_memory_bytes Current memory usage in bytes.\n");
        buffer.push_str("# TYPE winnow_memory_bytes gauge\n");
        buffer.push_str(&format!("winnow_memory_bytes {}\n", self.memory_usage.load(Ordering::Relaxed)));

        buffer.push_str("# HELP winnow_vector_count Total number of vectors indexed.\n");
        buffer.push_str("# TYPE winnow_vector_count gauge\n");
        buffer.push_str(&format!("winnow_vector_count {}\n", self.vector_count.load(Ordering::Relaxed)));
        
        // QPS
        buffer.push_str("# HELP winnow_qps Current queries per second.\n");
        buffer.push_str("# TYPE winnow_qps gauge\n");
        buffer.push_str(&format!("winnow_qps {:.2}\n", self.qps()));
        
        // Latency
        let sum = self.search_latency_sum_us.load(Ordering::Relaxed);
        let count = self.search_latency_count.load(Ordering::Relaxed);
        
        buffer.push_str("# HELP winnow_search_latency_seconds Search latency histogram.\n");
        buffer.push_str("# TYPE winnow_search_latency_seconds summary\n");
        buffer.push_str(&format!("winnow_search_latency_seconds_sum {}\n", sum as f64 / 1_000_000.0));
        buffer.push_str(&format!("winnow_search_latency_seconds_count {}\n", count));

        buffer
    }
}

