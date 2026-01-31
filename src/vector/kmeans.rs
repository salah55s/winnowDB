use wasm_bindgen::prelude::*;
use super::metric::{l2_distance};
use rand::Rng;

pub struct KMeans {
    k: usize,
    dim: usize,
    pub centroids: Vec<Vec<f32>>,
}

impl KMeans {
    pub fn new(k: usize, dim: usize) -> KMeans {
        KMeans {
            k,
            dim,
            centroids: Vec::new(),
        }
    }

    pub fn train(&mut self, data: &[f32], max_iter: usize) -> Result<(), JsValue> {
        let count = data.len() / self.dim;
        if count < self.k {
            return Err(JsValue::from_str("Not enough data to train K-Means"));
        }

        // K-Means++ Initialization
        self.centroids.clear();
        let mut rng = rand::thread_rng();
        
        // 1. Choose first centroid uniformly at random
        let first_idx = rng.gen_range(0..count);
        let offset = first_idx * self.dim;
        self.centroids.push(data[offset..offset+self.dim].to_vec());
        
        // 2. Choose remaining k-1 centroids
        let mut dists = vec![f32::MAX; count]; // Min distance from each point to any chosen centroid
        
        for _ in 1..self.k {
            let mut sum_dist_sq = 0.0;
            
            // Update distances
            for i in 0..count {
                let offset = i * self.dim;
                let vec = &data[offset..offset+self.dim];
                let last_centroid = self.centroids.last().unwrap();
                
                let d = l2_distance(vec, last_centroid);
                let d_sq = d * d;
                
                if d_sq < dists[i] {
                    dists[i] = d_sq;
                }
                sum_dist_sq += dists[i];
            }
            
            // Roulette selection
            let r = rng.gen_range(0.0..sum_dist_sq);
            let mut cum_sum = 0.0;
            let mut selected_idx = 0;
            
            for i in 0..count {
                cum_sum += dists[i];
                if cum_sum >= r {
                    selected_idx = i;
                    break;
                }
            }
            
            // Correct selected_idx if loop finishes without breaking (rounding errors)
            if cum_sum < r && count > 0 { selected_idx = count - 1; }

            let offset = selected_idx * self.dim;
            self.centroids.push(data[offset..offset+self.dim].to_vec());
        }

        let mut assignments = vec![0usize; count];
        
        for _iter in 0..max_iter {
            // 1. Assignment Step
            #[cfg(feature = "parallel")]
            let results: Vec<(usize, bool)> = {
                use rayon::prelude::*;
                (0..count).into_par_iter().map(|i| {
                    let offset = i * self.dim;
                    let vec = &data[offset..offset+self.dim];
                    
                    let mut best_dist = f32::MAX;
                    let mut best_cluster = 0;
                    
                    for (c_idx, centroid) in self.centroids.iter().enumerate() {
                        let d = l2_distance(vec, centroid);
                        if d < best_dist {
                            best_dist = d;
                            best_cluster = c_idx;
                        }
                    }
                    (best_cluster, false)
                }).collect()
            };
            
            #[cfg(not(feature = "parallel"))]
            let results: Vec<(usize, bool)> = {
                (0..count).map(|i| {
                    let offset = i * self.dim;
                    let vec = &data[offset..offset+self.dim];
                    
                    let mut best_dist = f32::MAX;
                    let mut best_cluster = 0;
                    
                    for (c_idx, centroid) in self.centroids.iter().enumerate() {
                        let d = l2_distance(vec, centroid);
                        if d < best_dist {
                            best_dist = d;
                            best_cluster = c_idx;
                        }
                    }
                    (best_cluster, false)
                }).collect()
            };
            
            // Re-calc changed count and update assignments
            let mut changed = 0;
            for i in 0..count {
                if assignments[i] != results[i].0 {
                    assignments[i] = results[i].0;
                    changed += 1;
                }
            }
            
            if changed == 0 { break; } // Converged

            // 2. Update Step
            let mut sums = vec![vec![0.0; self.dim]; self.k];
            let mut counts = vec![0usize; self.k];
            
            for i in 0..count {
                let cluster = assignments[i];
                let offset = i * self.dim;
                let vec = &data[offset..offset+self.dim];
                
                for j in 0..self.dim {
                    sums[cluster][j] += vec[j];
                }
                counts[cluster] += 1;
            }
            
            for c_idx in 0..self.k {
                if counts[c_idx] > 0 {
                    for j in 0..self.dim {
                        self.centroids[c_idx][j] = sums[c_idx][j] / counts[c_idx] as f32;
                    }
                } else {
                    // Handle empty cluster: Re-init to random point?
                    // For now, leave as is (unstable)
                }
            }
        }
        
        Ok(())
    }
}
