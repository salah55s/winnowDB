use crate::vector::storage::VectorIndex;
use crate::vector::metric::{l2_distance, MetricType};
use rand::Rng;
use std::collections::HashSet;
use serde_wasm_bindgen;
use js_sys::Date;

pub struct TestHarness<I: VectorIndex> {
    pub index: I,
    pub data: Vec<Vec<f32>>,
    pub ground_truth: Vec<Vec<(u32, f32)>>,
    pub test_queries: Vec<Vec<f32>>,
    pub dim: usize,
}

impl<I: VectorIndex> TestHarness<I> {
    /// Creates a new TestHarness by generating random data and computing ground truth.
    /// index_factory: Closure to create the index instance.
    pub fn new(
        dim: usize, 
        n_vectors: usize, 
        n_queries: usize, 
        input_metric: MetricType, 
        mut index_factory: impl FnMut(usize, MetricType) -> I
    ) -> Self {
        let data = generate_random_vectors(n_vectors, dim);
        let test_queries = generate_random_vectors(n_queries, dim);
        
        // Build Index
        let mut index = index_factory(dim, input_metric);
        for (i, vec) in data.iter().enumerate() {
            index.add(i as u32, vec.clone()).expect("Failed to add vector");
        }

        // Compute Ground Truth (Brute Force)
        let mut ground_truth = Vec::new();
        for query in &test_queries {
            let mut distances: Vec<(u32, f32)> = data.iter()
                .enumerate()
                .map(|(i, v)| {
                    let d = match input_metric {
                        MetricType::L2 => l2_distance(query, v),
                        // Simplified for harness: consistent metric usage
                        _ => l2_distance(query, v), 
                    };
                    (i as u32, d)
                })
                .collect();
            
            // Sort by distance (ASC for L2)
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            ground_truth.push(distances);
        }

        Self {
            index,
            data,
            ground_truth,
            test_queries,
            dim
        }
    }

    /// Measures Recall@K
    pub fn measure_recall(&self, k: usize) -> f32 {
        let mut total_recall = 0.0;
        
        for (i, query) in self.test_queries.iter().enumerate() {
            let results = self.index.search(query.clone(), k, None, None).expect("Search failed");
            
            // Parse results from JsValue
            let result_vec: Vec<(u32, f32)> = serde_wasm_bindgen::from_value(results).unwrap();
            let result_ids: HashSet<u32> = result_vec.iter().map(|(id, _)| *id).collect();
            
            let gt_ids: HashSet<u32> = self.ground_truth[i]
                .iter()
                .take(k)
                .map(|(id, _)| *id)
                .collect();
            
            let intersection = result_ids.intersection(&gt_ids).count();
            let recall = intersection as f32 / k as f32;
            total_recall += recall;
        }
        
        total_recall / self.test_queries.len() as f32
    }

    /// Measures QPS (Queries Per Second)
    pub fn measure_qps(&self, k: usize, duration_ms: f64) -> f32 {
        let start = Date::now();
        let mut query_count = 0;
        
        while Date::now() - start < duration_ms {
            let query_idx = query_count % self.test_queries.len();
            let _ = self.index.search(self.test_queries[query_idx].clone(), k, None, None);
            query_count += 1;
        }
        
        let elapsed_sec = (Date::now() - start) / 1000.0;
        query_count as f32 / elapsed_sec as f32
    }
}

pub fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| {
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }).collect()
}
