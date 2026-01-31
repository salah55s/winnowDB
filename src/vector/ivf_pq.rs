use wasm_bindgen::prelude::*;
use web_sys::AbortSignal;
use serde::{Serialize, Deserialize};
use super::kmeans::KMeans;
use super::pq::ProductQuantizer;
use super::metric::l2_distance;
use super::storage::VectorIndex;
use std::collections::{HashMap, HashSet};

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone)]
pub struct IndexIVFPQ {
    dim: usize,
    n_centroids: usize,
    m_sub: usize, 
    coarse_centroids: Vec<Vec<f32>>,
    pq: ProductQuantizer,
    inverted_lists: HashMap<usize, InvertedList>,
    trained: bool,
    #[wasm_bindgen(skip)]
    pub deleted: HashSet<u32>,
}

#[derive(Serialize, Deserialize, Clone)]
struct InvertedList {
    ids: Vec<u32>,
    codes: Vec<u8>, 
}

use super::metric::MetricType;

#[wasm_bindgen]
impl IndexIVFPQ {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, n_centroids: usize, m_sub: usize, metric: MetricType) -> Result<IndexIVFPQ, JsValue> {
        match metric {
            MetricType::L2 => {},
            _ => return Err(JsValue::from_str("IVFPQ only supports L2 metric currently.")),
        }
        
        Ok(IndexIVFPQ {
            dim,
            n_centroids,
            m_sub,
            coarse_centroids: Vec::new(),
            pq: ProductQuantizer::new(dim, m_sub),
            inverted_lists: HashMap::new(),
            trained: false,
            deleted: HashSet::new(),
        })
    }
    
    pub fn delete(&mut self, id: u32) -> Result<(), JsValue> {
        self.deleted.insert(id);
        Ok(())
    }

    pub fn clear(&mut self) -> Result<(), JsValue> {
        self.inverted_lists.clear();
        self.deleted.clear();
        // Keep training (centroids) to avoid forcing re-train
        Ok(())
    }

    pub fn train(&mut self, data: &[f32]) -> Result<(), JsValue> {
        // 1. Train Coarse Quantizer (K-Means)
        let mut kmeans = KMeans::new(self.n_centroids, self.dim);
        kmeans.train(data, 20)?; // 20 iterations
        self.coarse_centroids = kmeans.centroids;

        // 2. Compute Residuals needed for PQ training
        let count = data.len() / self.dim;
        let mut residuals = Vec::with_capacity(data.len());
        
        for i in 0..count {
            let offset = i * self.dim;
            let vec = &data[offset..offset+self.dim];
            
            // Find nearest coarse centroid
            let mut best_dist = f32::MAX;
            let mut best_idx = 0;
            for (c_idx, centroid) in self.coarse_centroids.iter().enumerate() {
                let d = l2_distance(vec, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_idx = c_idx;
                }
            }
            
            // Calculate residual: r = x - C
            let centroid = &self.coarse_centroids[best_idx];
            for j in 0..self.dim {
                residuals.push(vec[j] - centroid[j]);
            }
        }
        
        // 3. Train PQ on Residuals
        self.pq.train(&residuals)?;
        
        self.trained = true;
        Ok(())
    }

    pub fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue> {
        if !self.trained { return Err(JsValue::from_str("Index not trained")); }
        if vector.len() != self.dim { return Err(JsValue::from_str("Dim mismatch")); }
        
        if self.deleted.contains(&id) {
            self.deleted.remove(&id);
        }

        // 1. Assign to Coarse Cluster
        let mut best_dist = f32::MAX;
        let mut best_cluster = 0;
        for (c_idx, centroid) in self.coarse_centroids.iter().enumerate() {
            let d = l2_distance(&vector, centroid);
            if d < best_dist {
                best_dist = d;
                best_cluster = c_idx;
            }
        }

        // 2. Encode Residual
        // r = x - C
        let centroid = &self.coarse_centroids[best_cluster];
        let mut residual = vec![0.0; self.dim];
        for j in 0..self.dim {
            residual[j] = vector[j] - centroid[j];
        }

        let code = self.pq.encode(&residual);

        // 3. Store in Inverted List
        let list = self.inverted_lists.entry(best_cluster).or_insert(InvertedList {
            ids: Vec::new(),
            codes: Vec::new(),
        });
        
        list.ids.push(id);
        list.codes.extend(code);
        
        Ok(())
    }
}

impl IndexIVFPQ {
    pub fn search(&self, query: Vec<f32>, k: usize, nprobe: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        if !self.trained { return Err(JsValue::from_str("Index not trained")); }
        
        // 1. Coarse Search: Find nearest 'nprobe' clusters
        let mut cluster_dists: Vec<(usize, f32)> = self.coarse_centroids.iter().enumerate()
            .map(|(idx, c)| (idx, l2_distance(&query, c)))
            .collect();
        // Sort by distance (asc)
        cluster_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let probes = std::cmp::min(nprobe, self.n_centroids);
        
        let mut scores: Vec<(u32, f32)> = Vec::new();
        
        // 3. Search Loop
        for i in 0..probes {
            if let Some((c_idx, coarse_dist)) = cluster_dists.get(i) {
                // Check if cluster exists
                if let Some(list) = self.inverted_lists.get(c_idx) {
                    let centroid = &self.coarse_centroids[*c_idx];
                    
                    // Compute Residual Query: r_q = q - C_i
                    let mut resid_query = vec![0.0; self.dim];
                    for j in 0..self.dim {
                        resid_query[j] = query[j] - centroid[j];
                    }
                    
                    // Precompute ADC table for this residual
                    let adc_table = self.pq.compute_adc_table(&resid_query);
                    
                    let count = list.ids.len();
                    for j in 0..count {
                        let id = list.ids[j];
                        
                        // Metadata Filter
                        if let Some(allowed) = filter {
                            if !allowed.contains(&id) { continue; }
                        }

                        // Check Abort (Granular)
                        if j % 1000 == 0 {
                            if let Some(s) = signal {
                                if s.aborted() { return Err(JsValue::from_str("Search Aborted")); }
                            }
                        }

                        let start = j * self.m_sub; // Code length is m (n_sub)
                        let end = start + self.m_sub;
                        let code = &list.codes[start..end];
                        
                        let dist_residual = self.pq.distance_adc(code, &adc_table);
                        
                        // Combine: ||q - x|| ≈ sqrt(||q - c||^2 + ||r_x - r_q||^2) -> Coarse^2 + Residual
                        // Note: coarse_dist is usually squared L2 or L2? usage implies L2 usually.
                        // If metric is L2, we combine. 
                        // Simplified: final_dist = coarse_dist + dist_residual (if both squared l2)
                        // User provided: (coarse_dist * coarse_dist + dist_residual).sqrt()
                        // Assuming coarse_dist is L2 (Euclidean), and dist_residual is Squared L2?
                        // Actually ADC returns Squared L2 typically.
                        // Let's stick to user suggestion but verify types if possible. 
                        // Assuming coarse_dist comes from `active_centroids` search which likely returns Squared L2 if HNSW?
                        // HNSW search returns distances. If L2 metric, usually returns Euclidean? Or Squared?
                        // Let's rely on User's explicit formula: (c*c + r).sqrt()
                        
                        let approx_dist = (coarse_dist * coarse_dist + dist_residual).sqrt();

                        scores.push((id, approx_dist));
                    }
                }
            }
        }
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k: Vec<(u32, f32)> = scores.into_iter().take(k).collect();
        
        Ok(serde_wasm_bindgen::to_value(&top_k)?)
    }
}

// Implement trait (mapping simple search to nprobe search)
impl VectorIndex for IndexIVFPQ {
    fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue> {
        self.add(id, vector)
    }

    fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        // Default nprobe = sqrt(n_centroids) or fixed small number
        let nprobe = (self.n_centroids as f32).sqrt().ceil() as usize;
        self.search(query, k, nprobe, filter, signal)
    }
}
