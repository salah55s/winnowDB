use wasm_bindgen::prelude::*;
use web_sys::AbortSignal;
use serde::{Serialize, Deserialize};
use super::storage::VectorIndex;
use super::metric::MetricType;

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone)]
pub struct ScalarQuantizer {
    dim: usize,
    min: Vec<f32>,
    max: Vec<f32>,
    diff: Vec<f32>, // max - min, precomputed for speed
    trained: bool,
}

#[wasm_bindgen]
impl ScalarQuantizer {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize) -> ScalarQuantizer {
        ScalarQuantizer {
            dim,
            min: vec![0.0; dim],
            max: vec![0.0; dim],
            diff: vec![0.0; dim],
            trained: false,
        }
    }

    pub fn train(&mut self, data: &[f32]) -> Result<(), JsValue> {
        let count = data.len() / self.dim;
        if count == 0 { return Err(JsValue::from_str("No training data")); }

        // Initialize min/max with first vector
        for j in 0..self.dim {
            self.min[j] = data[j];
            self.max[j] = data[j];
        }

        // Scan all vectors
        for i in 1..count {
            let offset = i * self.dim;
            let vec = &data[offset..offset+self.dim];
            for j in 0..self.dim {
                if vec[j] < self.min[j] { self.min[j] = vec[j]; }
                if vec[j] > self.max[j] { self.max[j] = vec[j]; }
            }
        }

        // Compute diffs
        for j in 0..self.dim {
            self.diff[j] = self.max[j] - self.min[j];
            // Avoid division by zero
            if self.diff[j] == 0.0 { self.diff[j] = 1.0; } 
        }

        self.trained = true;
        Ok(())
    }

    pub fn encode(&self, vec: &[f32]) -> Vec<u8> {
        let mut code = Vec::with_capacity(self.dim);
        for j in 0..self.dim {
            let v = vec[j];
            // Normalize to [0, 1]
            let x = (v - self.min[j]) / self.diff[j];
            // Clamp
            let x_clamped = x.max(0.0).min(1.0);
            // Scale to [0, 255]
            let byte = (x_clamped * 255.0) as u8;
            code.push(byte);
        }
        code
    }

    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.dim);
        for j in 0..self.dim {
            let byte = code[j];
            let x = (byte as f32) / 255.0;
            let v = x * self.diff[j] + self.min[j];
            vec.push(v);
        }
        vec
    }
}

use std::collections::HashSet;

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone)]
pub struct IndexSQ {
    dim: usize,
    sq: ScalarQuantizer,
    codes: Vec<u8>, // Flattened [n * dim]
    ids: Vec<u32>,
    metric: MetricType,
    #[wasm_bindgen(skip)]
    pub deleted: HashSet<u32>,
}

#[wasm_bindgen]
impl IndexSQ {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, metric: MetricType) -> IndexSQ {
        IndexSQ {
            dim,
            sq: ScalarQuantizer::new(dim),
            codes: Vec::new(),
            ids: Vec::new(),
            metric,
            deleted: HashSet::new(),
        }
    }

    pub fn train(&mut self, data: &[f32]) -> Result<(), JsValue> {
        self.sq.train(data)
    }

    pub fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue> {
        if !self.sq.trained { return Err(JsValue::from_str("SQ not trained")); }
        
        // If re-adding, ensure it's not marked deleted
        if self.deleted.contains(&id) {
            self.deleted.remove(&id);
        }
        
        // Note: duplicates in ids/codes are possible if we just append.
        // For strict correctness, we should update existing index if id exists.
        // But for Soft Delete, appending is okay as long as we check deleted.
        // Ideally we should compact later. For now, append + un-delete is fine.
        
        let code = self.sq.encode(&vector);
        self.codes.extend(code);
        self.ids.push(id);
        Ok(())
    }
    
    pub fn delete(&mut self, id: u32) -> Result<(), JsValue> {
        self.deleted.insert(id);
        Ok(())
    }

    pub fn clear(&mut self) -> Result<(), JsValue> {
        self.codes.clear();
        self.ids.clear();
        self.deleted.clear();
        Ok(())
    }
}

impl IndexSQ {
    pub fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        if !self.sq.trained { return Err(JsValue::from_str("SQ not trained")); }
        
        let mut scores: Vec<(u32, f32)> = Vec::with_capacity(self.ids.len());
        
        for i in 0..self.ids.len() {
            // Check Abort
            if i % 1000 == 0 {
                if let Some(sig) = signal {
                    if sig.aborted() { return Err(JsValue::from_str("Search Aborted")); }
                }
            }

            let id = self.ids[i];
            if self.deleted.contains(&id) { continue; }
            if let Some(f) = filter { if !f.contains(&id) { continue; } }
            
            let start = i * self.dim;
            let code = &self.codes[start..start+self.dim];
            
            // On-the-fly Decode
            let mut reconstructed = Vec::with_capacity(self.dim);
            for j in 0..self.dim {
                let byte = code[j];
                let v = (byte as f32 / 255.0) * self.sq.diff[j] + self.sq.min[j];
                reconstructed.push(v);
            }
            
            let dist = match self.metric {
                MetricType::L2 => super::metric::l2_distance(&query, &reconstructed),
                MetricType::Cosine => super::metric::cosine_similarity(&query, &reconstructed),
                MetricType::InnerProduct => super::metric::inner_product(&query, &reconstructed),
            };
            
            let final_score = match self.metric {
                MetricType::L2 => dist,
                MetricType::Cosine => 1.0 - dist, // Cosine Distance
                MetricType::InnerProduct => -dist, // Negate for min-sort
            };
            
            scores.push((self.ids[i], final_score));
        }
        
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k: Vec<(u32, f32)> = scores.into_iter().take(k).collect();
        
        Ok(serde_wasm_bindgen::to_value(&top_k)?)
    }
}

impl VectorIndex for IndexSQ {
    fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue> {
        self.add(id, vector)
    }

    fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        self.search(query, k, filter, signal)
    }
}
