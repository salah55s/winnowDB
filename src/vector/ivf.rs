use wasm_bindgen::prelude::*;
use web_sys::AbortSignal;
use super::sq::ScalarQuantizer;
use super::kmeans::KMeans;
use super::metric::l2_distance;
use std::collections::HashSet;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct IvfList {
    pub centroid_id: usize,
    pub doc_ids: Vec<u32>,
    pub codes: Vec<u8>, // SQ8 encoded vectors
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct IndexIVF {
    dim: usize,
    nlist: usize, // number of clusters
    nprobe: usize, // number of clusters to search
    quantizer: ScalarQuantizer,
    centroids: Vec<f32>, // Flattened clusters
    invlists: Vec<IvfList>,
    trained: bool,
}

#[wasm_bindgen]
impl IndexIVF {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, nlist: usize, nprobe: usize) -> IndexIVF {
        IndexIVF {
            dim,
            nlist,
            nprobe,
            quantizer: ScalarQuantizer::new(dim),
            centroids: Vec::new(),
            invlists: Vec::new(),
            trained: false,
        }
    }

    pub fn train(&mut self, data: &[f32]) -> Result<(), JsValue> {
        // 1. Train Quantizer
        self.quantizer.train(data)?;

        // 2. Train KMeans
        let mut kmeans = KMeans::new(self.nlist, self.dim);
        kmeans.train(data, 10)?; // 10 iterations
        
        // Flatten centroids
        self.centroids = kmeans.centroids.into_iter().flatten().collect();
        
        // 3. Init Empty Lists
        self.invlists = Vec::with_capacity(self.nlist);
        for i in 0..self.nlist {
            self.invlists.push(IvfList { 
                centroid_id: i, 
                doc_ids: Vec::new(), 
                codes: Vec::new() 
            });
        }
        
        self.trained = true;
        Ok(())
    }

    pub fn add(&mut self, id: u32, vec: Vec<f32>) -> Result<(), JsValue> {
        if !self.trained { return Err(JsValue::from_str("Index not trained")); }
        
        // 1. Find nearest centroid
        let mut best_dist = f32::MAX;
        let mut best_cluster = 0;
        
        for i in 0..self.nlist {
            let start = i * self.dim;
            let c = &self.centroids[start..start+self.dim];
            let d = l2_distance(&vec, c);
            if d < best_dist {
                best_dist = d;
                best_cluster = i;
            }
        }
        
        // 2. Encode Vector
        let code = self.quantizer.encode(&vec);
        
        // 3. Push to List
        self.invlists[best_cluster].doc_ids.push(id);
        self.invlists[best_cluster].codes.extend(code);
        
        Ok(())
    }

    pub fn save(&self) -> Result<Vec<u8>, JsValue> {
        self.to_bytes()
    }

    pub fn load(data: &[u8]) -> Result<IndexIVF, JsValue> {
        IndexIVF::from_bytes(data)
    }
}

impl IndexIVF {
    pub fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        if !self.trained { return Err(JsValue::from_str("Index not trained")); }

        // 1. Find 'nprobe' nearest centroids
        let mut cluster_dists: Vec<(usize, f32)> = (0..self.nlist).map(|i| {
            let start = i * self.dim;
            let c = &self.centroids[start..start+self.dim];
            (i, l2_distance(&query, c))
        }).collect();
        
        cluster_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // 2. Scan candidates
        let mut scores: Vec<(u32, f32)> = Vec::new();
        
        for (i, _) in cluster_dists.iter().take(self.nprobe) {
            if let Some(sig) = signal {
                if sig.aborted() { return Err(JsValue::from_str("Search Aborted")); }
            }
            let list = &self.invlists[*i];
            let count = list.doc_ids.len();
            
            for j in 0..count {
                let id = list.doc_ids[j];
                // Filter
                if let Some(f) = filter { if !f.contains(&id) { continue; } }

                let start = j * self.dim;
                let code = &list.codes[start..start+self.dim];
                
                // Decode SQ8
                let rec_vec = self.quantizer.decode(code);
                let dist = l2_distance(&query, &rec_vec);
                
                scores.push((id, dist));
            }
        }
        
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k: Vec<(u32, f32)> = scores.into_iter().take(k).collect();
        
        Ok(serde_wasm_bindgen::to_value(&top_k)?)
    }
}

use super::storage::VectorIndex;
impl VectorIndex for IndexIVF {
    fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue> {
        self.add(id, vector)
    }

    fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        self.search(query, k, filter, signal)
    }
}

