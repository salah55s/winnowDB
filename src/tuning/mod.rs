//! Index Tuning and Comparison Module
//! 
//! Provides tools for benchmarking different index types and automatically 
//! recommending configurations based on dataset characteristics.

use serde::{Deserialize, Serialize};
use crate::vector::storage::VectorIndex;
use wasm_bindgen::prelude::*;
use std::collections::HashMap;

// ============================================================================
// Auto-Tuning Recommendations
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetProfile {
    pub vector_count: usize,
    pub dimensions: usize,
    pub write_heavy: bool,
    pub low_latency_required: bool,
    pub low_memory_required: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexRecommendation {
    pub index_type: String, // "Flat", "HNSW", "IVF-PQ"
    pub config: HashMap<String, String>,
    pub reasoning: String,
    pub estimated_memory_bytes: usize,
}

#[wasm_bindgen]
pub struct AutoTuner;

#[wasm_bindgen]
impl AutoTuner {
    /// Analyze dataset and Hardware constraints to recommend index config
    pub fn recommend(
        count: usize, 
        dim: usize, 
        write_heavy: bool, 
        low_latency: bool,
        low_mem: bool
    ) -> JsValue {
        let profile = DatasetProfile {
            vector_count: count,
            dimensions: dim,
            write_heavy,
            low_latency_required: low_latency,
            low_memory_required: low_mem,
        };

        let recommendation = Self::analyze(profile);
        serde_wasm_bindgen::to_value(&recommendation).unwrap_or(JsValue::NULL)
    }

    fn analyze(profile: DatasetProfile) -> IndexRecommendation {
        let float_size = 4; // f32
        let raw_data_size = profile.vector_count * profile.dimensions * float_size;

        // Rule 1: Small datasets (< 10k) should always use Flat
        if profile.vector_count <= 10_000 {
            return IndexRecommendation {
                index_type: "Flat".to_string(),
                config: HashMap::new(),
                reasoning: "Dataset is small enough for brute-force search. 100% recall, zero build time.".to_string(),
                estimated_memory_bytes: raw_data_size,
            };
        }

        // Rule 2: Memory Constraints (Use IVF-PQ or SQ)
        if profile.low_memory_required || profile.vector_count > 1_000_000 {
            let mut config = HashMap::new();
            config.insert("nfft".to_string(), clamp(profile.vector_count / 1000, 10, 4096).to_string());
            config.insert("nsub".to_string(), "8".to_string()); // 8 sub-vectors common for PQ
            
            return IndexRecommendation {
                index_type: "IVF-PQ".to_string(),
                config,
                reasoning: "Projected Quantization required to fit in memory / efficient scale.".to_string(),
                estimated_memory_bytes: raw_data_size / 8 + (profile.vector_count * 4), // Approx compression
            };
        }

        // Rule 3: Default High Performance (HNSW)
        let mut config = HashMap::new();
        let m = if profile.write_heavy { 16 } else { 32 };
        let ef = if profile.low_latency_required { 64 } else { 128 };
        
        config.insert("m".to_string(), m.to_string());
        config.insert("ef_construction".to_string(), ef.to_string());

        IndexRecommendation {
            index_type: "HNSW".to_string(),
            config,
            reasoning: "Graph-based HNSW offers best trade-off for speed vs recall on mid-sized datasets.".to_string(),
            estimated_memory_bytes: (raw_data_size as f64 * 1.5) as usize, // Graph overhead
        }
    }
}

fn clamp(v: usize, min: usize, max: usize) -> usize {
    if v < min { min } else if v > max { max } else { v }
}

// ============================================================================
// Index Benchmark / Comparison
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub index_name: String,
    pub build_time_ms: f64,
    pub p95_latency_ms: f64,
    pub recall: f64,
    pub memory_usage_mb: f64,
}

#[wasm_bindgen]
pub struct IndexComparator {
    // In a real implementation, this would hold references to loaded indexes
    // For this implementation, we simulated specific results or provide specific hooks
}

#[wasm_bindgen]
impl IndexComparator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Calculate recall between exact (Flat) results and candidate results
    /// IDs are assumed to be sorted by score? No, sets.
    pub fn calculate_recall(ground_truth_ids: Vec<u32>, candidate_ids: Vec<u32>) -> f32 {
        if ground_truth_ids.is_empty() {
            return 0.0;
        }
        
        // Count matches
        let mut matches = 0;
        for id in &candidate_ids {
            if ground_truth_ids.contains(id) {
                matches += 1;
            }
        }
        
        matches as f32 / ground_truth_ids.len() as f32
    }
    
    /// Run a simulation benchmark (Mock for WASM demo purposes)
    pub fn simulate_benchmark(&self, index_type: &str, count: usize) -> JsValue {
        let (build, lat, rec, mem) = match index_type {
            "Flat" => (count as f64 * 0.001, count as f64 * 0.0001, 1.0, count as f64 * 4.0 * 128.0 / 1024.0 / 1024.0),
            "HNSW" => (count as f64 * 0.1, 2.0, 0.98, count as f64 * 6.0 * 128.0 / 1024.0 / 1024.0),
            "IVF-PQ" => (count as f64 * 0.05, 5.0, 0.92, count as f64 * 0.5 * 128.0 / 1024.0 / 1024.0),
            _ => (0.0, 0.0, 0.0, 0.0),
        };
        
        let res = BenchmarkResult {
            index_name: index_type.to_string(),
            build_time_ms: build,
            p95_latency_ms: lat,
            recall: rec,
            memory_usage_mb: mem,
        };
        
        serde_wasm_bindgen::to_value(&res).unwrap_or(JsValue::NULL)
    }
}
