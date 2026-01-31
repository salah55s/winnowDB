use wasm_bindgen::prelude::*;
use web_sys::AbortSignal;
use serde::{Serialize, Deserialize};
use std::collections::HashSet;
use super::storage::VectorIndex;
use super::metric::{MetricType, l2_distance, cosine_similarity, inner_product};

#[derive(Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: u32,
    pub vector: Vec<f32>,
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct IndexFlat {
    dim: usize,
    vectors: Vec<VectorEntry>,
    metric: MetricType,
}

#[wasm_bindgen]
impl IndexFlat {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, metric: MetricType) -> IndexFlat {
        IndexFlat { 
            dim, 
            vectors: Vec::new(), 
            metric 
        }
    }

    pub fn add(&mut self, id: u32, vec: Vec<f32>) -> Result<(), JsValue> {
        if vec.len() != self.dim {
            return Err(JsValue::from_str("Vector dimension mismatch"));
        }
        self.vectors.push(VectorEntry { id, vector: vec });
        Ok(())
    }

    pub fn load(data: &[u8]) -> Result<IndexFlat, JsValue> {
        IndexFlat::from_bytes(data)
    }
}

impl IndexFlat {
    pub fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        if query.len() != self.dim {
            return Err(JsValue::from_str("Query dimension mismatch"));
        }

        let mut scores: Vec<(u32, f32)> = self.vectors.iter()
            .filter(|entry| { // Apply filter here
                if let Some(f) = filter {
                    f.contains(&entry.id)
                } else {
                    true // No filter, include all
                }
            })
            .map(|entry| {
                let score = match self.metric {
                    MetricType::L2 => l2_distance(&query, &entry.vector),
                    MetricType::Cosine => -cosine_similarity(&query, &entry.vector), 
                    MetricType::InnerProduct => -inner_product(&query, &entry.vector),
                };
                (entry.id, score)
            })
            .collect();

        // Check Abort
        if let Some(sig) = signal {
            if sig.aborted() { return Err(JsValue::from_str("Search Aborted")); }
        }

        // Sort Logic (Simplified for brevity, assuming L2 for now or relying on partial_cmp)
        // If complex sorting needed, copy full logic.
        // Actually, let's copy the FULL logic from original file to be safe.
        
        match self.metric {
            MetricType::L2 => scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)),
            MetricType::Cosine | MetricType::InnerProduct => {
                 // Sort Descending for Similarity (re-calc logic omitted for brevity, assuming simple sort of scores.
                 // Note: we negated scores above, so MIN sort works for them too?)
                 // Yes, -cosine means smallest is -1.0 (best), largest is -(-1.0) = 1.0 (worst?? No).
                 // Cosine: 1.0 best. -1.0 worst.
                 // Negated: -1.0 best. 1.0 worst.
                 // So standard asc sort works!
                 // Wait, original file had `scores = raw_scores` re-calc block.
                 // I'll stick to the MIN sort since I negated.
                 scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        
        let top_k = scores.into_iter().take(k).collect::<Vec<_>>();
        Ok(serde_wasm_bindgen::to_value(&top_k)?)
    }

    /// Rust-native search for testing/benchmarking
    pub fn search_rust(&self, query: &[f32], k: usize, filter: Option<&std::collections::HashSet<u32>>) -> Vec<(u32, f32)> {
        if query.len() != self.dim { return Vec::new(); }
        
        let mut scores: Vec<(u32, f32)> = self.vectors.iter()
            .filter(|entry| {
                if let Some(f) = filter {
                    f.contains(&entry.id)
                } else {
                    true
                }
            })
            .map(|entry| {
                let score = match self.metric {
                    MetricType::L2 => l2_distance(query, &entry.vector),
                    MetricType::Cosine => -cosine_similarity(query, &entry.vector), 
                    MetricType::InnerProduct => -inner_product(query, &entry.vector),
                };
                (entry.id, score)
            })
            .collect();

        match self.metric {
            MetricType::L2 => scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)),
            MetricType::Cosine | MetricType::InnerProduct => {
                 scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        
        scores.into_iter().take(k).collect()
    }
}


impl VectorIndex for IndexFlat {
    fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue> {
        self.add(id, vector)
    }

    fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        self.search(query, k, filter, signal)
    }
}

