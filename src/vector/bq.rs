use wasm_bindgen::prelude::*;
use web_sys::AbortSignal;
use serde::{Serialize, Deserialize};
use super::storage::VectorIndex;

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone)]
pub struct BinaryQuantizer {
    dim: usize,
    thresholds: Vec<f32>, // Usually mean of each dimension
    pub trained: bool,
}

#[wasm_bindgen]
impl BinaryQuantizer {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize) -> BinaryQuantizer {
        BinaryQuantizer {
            dim,
            thresholds: vec![0.0; dim],
            trained: false,
        }
    }

    pub fn train(&mut self, data: &[f32]) -> Result<(), JsValue> {
        let count = data.len() / self.dim;
        if count == 0 { return Err(JsValue::from_str("No training data")); }

        // Compute Mean for each dimension
        let mut sums = vec![0.0; self.dim];
        
        for i in 0..count {
            let offset = i * self.dim;
            let vec = &data[offset..offset+self.dim];
            for j in 0..self.dim {
                sums[j] += vec[j];
            }
        }
        
        for j in 0..self.dim {
            self.thresholds[j] = sums[j] / count as f32;
        }

        self.trained = true;
        Ok(())
    }

    pub fn encode(&self, vec: &[f32]) -> Vec<u64> {
        let num_u64 = (self.dim + 63) / 64;
        let mut code = vec![0u64; num_u64];
        
        for j in 0..self.dim {
            if vec[j] > self.thresholds[j] {
                let chunk_idx = j / 64;
                let bit_idx = j % 64;
                code[chunk_idx] |= 1 << bit_idx;
            }
        }
        code
    }
}

pub fn hamming_distance(a: &[u64], b: &[u64]) -> f32 {
    let mut dist = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        dist += (x ^ y).count_ones();
    }
    dist as f32
}

use std::collections::HashSet;

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone)]
pub struct IndexBQ {
    dim: usize,
    bq: BinaryQuantizer,
    codes: Vec<u64>, // Flattened [n * num_u64]
    ids: Vec<u32>,
    num_u64: usize,
    #[wasm_bindgen(skip)]
    pub deleted: HashSet<u32>,
}

#[wasm_bindgen]
impl IndexBQ {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize) -> IndexBQ {
        let num_u64 = (dim + 63) / 64;
        IndexBQ {
            dim,
            bq: BinaryQuantizer::new(dim),
            codes: Vec::new(),
            ids: Vec::new(),
            num_u64,
            deleted: HashSet::new(),
        }
    }

    pub fn train(&mut self, data: &[f32]) -> Result<(), JsValue> {
        self.bq.train(data)
    }

    pub fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue> {
        if !self.bq.trained { return Err(JsValue::from_str("BQ not trained")); }
        
        if self.deleted.contains(&id) {
            self.deleted.remove(&id);
        }
        
        let code = self.bq.encode(&vector);
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

impl IndexBQ {
    pub fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        if !self.bq.trained { return Err(JsValue::from_str("BQ not trained")); }
        
        let query_code = self.bq.encode(&query);
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

            let start = i * self.num_u64;
            let code = &self.codes[start..start+self.num_u64];
            let dist = hamming_distance(&query_code, code);
            scores.push((self.ids[i], dist));
        }
        
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k: Vec<(u32, f32)> = scores.into_iter().take(k).collect();
        
        Ok(serde_wasm_bindgen::to_value(&top_k)?)
    }
}

impl VectorIndex for IndexBQ {
    fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue> {
        self.add(id, vector)
    }

    fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        self.search(query, k, filter, signal)
    }
}
