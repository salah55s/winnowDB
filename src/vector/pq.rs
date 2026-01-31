use wasm_bindgen::prelude::*;
use web_sys::AbortSignal;
use serde::{Serialize, Deserialize};
use super::kmeans::KMeans;
use std::collections::HashSet;

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone)]
pub struct ProductQuantizer {
    dim: usize,
    m: usize, // number of sub-vectors
    dsub: usize, // dimension of each sub-vector
    centroids: Vec<f32>, // Flattened: [m * 256 * dsub]
    trained: bool,
}

#[wasm_bindgen]
impl ProductQuantizer {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, m: usize) -> ProductQuantizer {
        assert!(dim % m == 0, "Dimension must be divisible by m");
        ProductQuantizer {
            dim,
            m,
            dsub: dim / m,
            centroids: Vec::new(),
            trained: false,
        }
    }

    pub fn train(&mut self, data: &[f32]) -> Result<(), JsValue> {
        let count = data.len() / self.dim;
        let k_sub = 256; // Fixed for 8-bit PQ
        
        self.centroids = vec![0.0; self.m * k_sub * self.dsub];
        
        // Train M sub-quantizers
        for i in 0..self.m {
            let mut sub_data = Vec::with_capacity(count * self.dsub);
            for r in 0..count {
                let start = r * self.dim + i * self.dsub;
                sub_data.extend_from_slice(&data[start..start+self.dsub]);
            }
            
            let mut kmeans = KMeans::new(k_sub, self.dsub);
            kmeans.train(&sub_data, 10)?; // 10 iterations
            
            // Copy centroids into flat array
            for (c_idx, c_vec) in kmeans.centroids.iter().enumerate() {
                let start = (i * k_sub + c_idx) * self.dsub;
                self.centroids[start..start+self.dsub].copy_from_slice(c_vec);
            }
        }
        
        self.trained = true;
        Ok(())
    }

    pub fn encode(&self, vec: &[f32]) -> Vec<u8> {
        if !self.trained { return vec![0u8; self.m]; }
        
        let mut code = Vec::with_capacity(self.m);
        let k_sub = 256;
        
        for i in 0..self.m {
            let start = i * self.dsub;
            let sub_vec = &vec[start..start+self.dsub];
            
            // Find nearest centroid in subspace i
            let mut best_dist = f32::MAX;
            let mut best_idx = 0;
            
            for c_idx in 0..k_sub {
                let c_start = (i * k_sub + c_idx) * self.dsub;
                let c_vec = &self.centroids[c_start..c_start+self.dsub];
                let d = super::metric::l2_distance(sub_vec, c_vec);
                if d < best_dist {
                    best_dist = d;
                    best_idx = c_idx;
                }
            }
            code.push(best_idx as u8);
        }
        code
    }
    
    // Asymmetric Distance Computation (ADC) Table
    // Returns flattened table[m * 256] distances
    pub fn compute_adc_table(&self, query: &[f32]) -> Vec<f32> {
        let k_sub = 256;
        let mut table = vec![0.0; self.m * k_sub];
        
        for i in 0..self.m {
             let start = i * self.dsub;
             let sub_query = &query[start..start+self.dsub];
             
             for c_idx in 0..k_sub {
                 let c_start = (i * k_sub + c_idx) * self.dsub;
                 let c_vec = &self.centroids[c_start..c_start+self.dsub];
                 let d = super::metric::l2_distance(sub_query, c_vec);
                 // We store SQUARED distance in the table, so we can sum them up validly.
                 // ||v||^2 = sum(||sub_v||^2)
                 table[i * k_sub + c_idx] = d * d;
             }
        }
        table
    }
    
    // Distance using ADC table and code
    pub fn distance_adc(&self, code: &[u8], adc_table: &[f32]) -> f32 {
        let mut dist = 0.0;
        let k_sub = 256;
        for i in 0..self.m {
            let c = code[i] as usize;
            dist += adc_table[i * k_sub + c]; // Lookup distance to centroid c
        }
        dist
    }
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct IndexPQ {
    dim: usize,
    pq: ProductQuantizer,
    codes: Vec<u8>, // [n * m]
    ids: Vec<u32>,
}

#[wasm_bindgen]
impl IndexPQ {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, m: usize) -> IndexPQ {
        IndexPQ {
            dim,
            pq: ProductQuantizer::new(dim, m),
            codes: Vec::new(),
            ids: Vec::new(),
        }
    }

    pub fn train(&mut self, data: &[f32]) -> Result<(), JsValue> {
        self.pq.train(data)
    }

    pub fn add(&mut self, id: u32, vec: Vec<f32>) -> Result<(), JsValue> {
        if !self.pq.trained { return Err(JsValue::from_str("PQ not trained")); }
        let code = self.pq.encode(&vec);
        self.codes.extend(code);
        self.ids.push(id);
        Ok(())
    }
}

impl IndexPQ {
    pub fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        if !self.pq.trained { return Err(JsValue::from_str("PQ not trained")); }
        let adc_table = self.pq.compute_adc_table(&query);
        let mut scores: Vec<(u32, f32)> = Vec::with_capacity(self.ids.len());
        
        for i in 0..self.ids.len() {
            let id = self.ids[i];
            if i % 1000 == 0 {
                if let Some(sig) = signal {
                    if sig.aborted() { return Err(JsValue::from_str("Search Aborted")); }
                }
            }
            if let Some(f) = filter { if !f.contains(&id) { continue; } }

            let start = i * self.pq.m;
            let code = &self.codes[start..start+self.pq.m];
            let dist = self.pq.distance_adc(code, &adc_table);
            scores.push((id, dist));
        }
        
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k: Vec<(u32, f32)> = scores.into_iter().take(k).collect();
        Ok(serde_wasm_bindgen::to_value(&top_k)?)
    }
}

use super::storage::VectorIndex;
impl VectorIndex for IndexPQ {
    fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue> {
        self.add(id, vector)
    }

    fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        self.search(query, k, filter, signal)
    }
}
