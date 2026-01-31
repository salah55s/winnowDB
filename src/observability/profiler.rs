//! Built-in profiler for WinnowDB
//!
//! Provides performance profiling and query analysis tools.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::Mutex;
use wasm_bindgen::prelude::*;

/// Operation types that can be profiled
#[derive(Clone, Debug, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum ProfiledOp {
    Search,
    SearchSparse,
    SearchHybrid,
    Insert,
    InsertBatch,
    Delete,
    Snapshot,
    LoadSnapshot,
}

/// A single profiled operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileEntry {
    pub op: ProfiledOp,
    pub start_time: f64,
    pub duration_ms: f64,
    pub vector_count: usize,
    pub k: Option<usize>,
    pub filtered: bool,
    pub success: bool,
}

/// Aggregated stats for an operation type
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OpStats {
    pub count: u64,
    pub total_time_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub avg_time_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub error_count: u64,
}

/// Profiler for tracking operation performance
#[derive(Clone)]
pub struct Profiler {
    enabled: Arc<Mutex<bool>>,
    entries: Arc<Mutex<Vec<ProfileEntry>>>,
    max_entries: usize,
    latencies: Arc<Mutex<HashMap<ProfiledOp, Vec<f64>>>>,
}

impl Profiler {
    pub fn new(max_entries: usize) -> Self {
        Self {
            enabled: Arc::new(Mutex::new(true)),
            entries: Arc::new(Mutex::new(Vec::with_capacity(max_entries))),
            max_entries,
            latencies: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    pub fn set_enabled(&self, enabled: bool) {
        *self.enabled.lock() = enabled;
    }
    
    pub fn is_enabled(&self) -> bool {
        *self.enabled.lock()
    }
    
    /// Record a profiled operation
    pub fn record(&self, entry: ProfileEntry) {
        if !*self.enabled.lock() { return; }
        
        let mut entries = self.entries.lock();
        if entries.len() >= self.max_entries {
            entries.remove(0);
        }
        
        // Track latency for percentile calculations
        let mut latencies = self.latencies.lock();
        latencies.entry(entry.op.clone())
            .or_insert_with(Vec::new)
            .push(entry.duration_ms);
        
        // Limit latency samples
        if let Some(lats) = latencies.get_mut(&entry.op) {
            if lats.len() > 10000 {
                lats.drain(0..5000);
            }
        }
        
        entries.push(entry);
    }
    
    /// Get aggregated stats for an operation type
    pub fn stats(&self, op: &ProfiledOp) -> OpStats {
        let entries = self.entries.lock();
        let filtered: Vec<_> = entries.iter().filter(|e| &e.op == op).collect();
        
        if filtered.is_empty() {
            return OpStats::default();
        }
        
        let count = filtered.len() as u64;
        let total_time_ms: f64 = filtered.iter().map(|e| e.duration_ms).sum();
        let min_time_ms = filtered.iter().map(|e| e.duration_ms).fold(f64::MAX, f64::min);
        let max_time_ms = filtered.iter().map(|e| e.duration_ms).fold(0.0, f64::max);
        let error_count = filtered.iter().filter(|e| !e.success).count() as u64;
        
        // Calculate percentiles
        let mut latencies: Vec<f64> = filtered.iter().map(|e| e.duration_ms).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p50 = percentile(&latencies, 50.0);
        let p95 = percentile(&latencies, 95.0);
        let p99 = percentile(&latencies, 99.0);
        
        OpStats {
            count,
            total_time_ms,
            min_time_ms,
            max_time_ms,
            avg_time_ms: total_time_ms / count as f64,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
            error_count,
        }
    }
    
    /// Get all stats
    pub fn all_stats(&self) -> HashMap<ProfiledOp, OpStats> {
        let mut result = HashMap::new();
        for op in [ProfiledOp::Search, ProfiledOp::SearchSparse, ProfiledOp::SearchHybrid,
                   ProfiledOp::Insert, ProfiledOp::InsertBatch, ProfiledOp::Delete,
                   ProfiledOp::Snapshot, ProfiledOp::LoadSnapshot] {
            let stats = self.stats(&op);
            if stats.count > 0 {
                result.insert(op, stats);
            }
        }
        result
    }
    
    /// Get recent entries
    pub fn recent(&self, count: usize) -> Vec<ProfileEntry> {
        let entries = self.entries.lock();
        entries.iter().rev().take(count).cloned().collect()
    }
    
    /// Clear all profiling data
    pub fn clear(&self) {
        self.entries.lock().clear();
        self.latencies.lock().clear();
    }
    
    /// Generate a profiling report
    pub fn report(&self) -> ProfileReport {
        let all_stats = self.all_stats();
        let total_ops: u64 = all_stats.values().map(|s| s.count).sum();
        let total_time: f64 = all_stats.values().map(|s| s.total_time_ms).sum();
        let total_errors: u64 = all_stats.values().map(|s| s.error_count).sum();
        
        ProfileReport {
            total_operations: total_ops,
            total_time_ms: total_time,
            total_errors,
            error_rate: if total_ops > 0 { total_errors as f64 / total_ops as f64 } else { 0.0 },
            ops_stats: all_stats,
            slowest_ops: self.slowest(5),
        }
    }
    
    /// Get slowest operations
    fn slowest(&self, count: usize) -> Vec<ProfileEntry> {
        let entries = self.entries.lock();
        let mut sorted: Vec<_> = entries.iter().cloned().collect();
        sorted.sort_by(|a, b| b.duration_ms.partial_cmp(&a.duration_ms).unwrap());
        sorted.into_iter().take(count).collect()
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new(10000)
    }
}

/// Profiling report
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileReport {
    pub total_operations: u64,
    pub total_time_ms: f64,
    pub total_errors: u64,
    pub error_rate: f64,
    pub ops_stats: HashMap<ProfiledOp, OpStats>,
    pub slowest_ops: Vec<ProfileEntry>,
}

/// Query explanation result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryExplanation {
    pub index_type: String,
    pub vector_count: u64,
    pub k: usize,
    pub estimated_scans: u64,
    pub estimated_comparisons: u64,
    pub estimated_latency_ms: f64,
    pub filter_selectivity: Option<f64>,
    pub recommendations: Vec<String>,
}

/// Query explainer
pub struct QueryExplainer;

impl QueryExplainer {
    /// Explain a search query
    pub fn explain(
        index_type: &str,
        vector_count: u64,
        dim: usize,
        k: usize,
        filter_count: Option<usize>,
    ) -> QueryExplanation {
        let mut recommendations = Vec::new();
        
        let (estimated_scans, estimated_comparisons, estimated_latency) = match index_type {
            "HNSW" => {
                // HNSW: O(log N) scans, ef_search * log(N) comparisons
                let scans = (vector_count as f64).log2().ceil() as u64 * 50; // ef_search assumption
                let comps = scans * 2;
                let latency = (comps as f64 * dim as f64) * 0.00001; // Approximate
                (scans, comps, latency)
            },
            "IVFPQ" => {
                // IVF-PQ: nprobe clusters + PQ distance
                let nprobe = 10u64;
                let scans = (vector_count / 100).max(1) * nprobe;
                let comps = scans * 2;
                let latency = (comps as f64 * 8.0) * 0.00001; // PQ is faster
                (scans, comps, latency)
            },
            "Flat" | _ => {
                // Brute force: N scans
                (vector_count, vector_count, (vector_count as f64 * dim as f64) * 0.0001)
            }
        };
        
        let filter_selectivity = filter_count.map(|fc| fc as f64 / vector_count as f64);
        
        // Generate recommendations
        if vector_count > 100_000 && index_type == "Flat" {
            recommendations.push("Consider using HNSW index for large collections".to_string());
        }
        
        if k > 100 {
            recommendations.push("Large k values may impact performance; consider pagination".to_string());
        }
        
        if let Some(sel) = filter_selectivity {
            if sel < 0.01 {
                recommendations.push("Very selective filter; pre-filtering may be more efficient".to_string());
            }
        }
        
        QueryExplanation {
            index_type: index_type.to_string(),
            vector_count,
            k,
            estimated_scans,
            estimated_comparisons,
            estimated_latency_ms: estimated_latency,
            filter_selectivity,
            recommendations,
        }
    }
}

/// WASM-exposed profiler
#[wasm_bindgen]
pub struct ProfileManager {
    profiler: Profiler,
}

#[wasm_bindgen]
impl ProfileManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            profiler: Profiler::default(),
        }
    }
    
    pub fn enable(&self, enabled: bool) {
        self.profiler.set_enabled(enabled);
    }
    
    pub fn is_enabled(&self) -> bool {
        self.profiler.is_enabled()
    }
    
    /// Record a search operation
    pub fn record_search(&self, duration_ms: f64, k: usize, filtered: bool, success: bool) {
        self.profiler.record(ProfileEntry {
            op: ProfiledOp::Search,
            start_time: js_sys::Date::now() - duration_ms,
            duration_ms,
            vector_count: 0,
            k: Some(k),
            filtered,
            success,
        });
    }
    
    /// Record an insert operation
    pub fn record_insert(&self, duration_ms: f64, batch_size: usize, success: bool) {
        self.profiler.record(ProfileEntry {
            op: if batch_size > 1 { ProfiledOp::InsertBatch } else { ProfiledOp::Insert },
            start_time: js_sys::Date::now() - duration_ms,
            duration_ms,
            vector_count: batch_size,
            k: None,
            filtered: false,
            success,
        });
    }
    
    /// Get stats for search operations
    pub fn search_stats(&self) -> JsValue {
        let stats = self.profiler.stats(&ProfiledOp::Search);
        serde_wasm_bindgen::to_value(&stats).unwrap_or(JsValue::NULL)
    }
    
    /// Get full report
    pub fn report(&self) -> JsValue {
        let report = self.profiler.report();
        serde_wasm_bindgen::to_value(&report).unwrap_or(JsValue::NULL)
    }
    
    /// Get recent operations
    pub fn recent(&self, count: usize) -> JsValue {
        let entries = self.profiler.recent(count);
        serde_wasm_bindgen::to_value(&entries).unwrap_or(JsValue::NULL)
    }
    
    /// Explain a query
    pub fn explain_query(&self, index_type: &str, vector_count: u64, dim: usize, k: usize) -> JsValue {
        let explanation = QueryExplainer::explain(index_type, vector_count, dim, k, None);
        serde_wasm_bindgen::to_value(&explanation).unwrap_or(JsValue::NULL)
    }
    
    /// Clear profiling data
    pub fn clear(&self) {
        self.profiler.clear();
    }
}

impl Default for ProfileManager {
    fn default() -> Self {
        Self::new()
    }
}
