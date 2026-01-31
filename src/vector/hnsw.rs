use wasm_bindgen::prelude::*;
use web_sys::AbortSignal;
use super::metric::{l2_distance, MetricType};
use serde::{Serialize, Deserialize};
use std::collections::{BinaryHeap, HashSet, HashMap};
use std::cmp::Ordering;

// Helper for MinHeap (Smallest distance at top)
#[derive(Debug, PartialEq, Clone)]
struct MinOrdFloat(f32);
impl Eq for MinOrdFloat {}
impl PartialOrd for MinOrdFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.0.partial_cmp(&self.0) // Reverse for MinHeap
    }
}
impl Ord for MinOrdFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}
#[derive(Eq, PartialEq)]
struct Candidate {
    id: u32,
    distance: MinOrdFloat,
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


#[derive(Clone, Serialize, Deserialize)]
struct Node {
    id: u32,
    vector: Vec<f32>,
    layer: usize,
    neighbors: Vec<Vec<u32>>, // [layer][neighbor_idx]
}



// Float wrapper for Ord
#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
struct OrderedFloat(f32);
impl Eq for OrderedFloat {}
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// Max Heap Item (Worst candidate at top)
#[derive(Eq, PartialEq)]
struct DistNode {
    id: u32,
    dist: OrderedFloat,
}
impl Ord for DistNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.cmp(&other.dist)
    }
}
impl PartialOrd for DistNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Min Heap Item (Best candidate at top)
#[derive(Eq, PartialEq)]
struct MinDistNode {
    id: u32,
    dist: OrderedFloat,
}
impl Ord for MinDistNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for MinHeap behavior in Rust's BinaryHeap (which is MaxHeap)
        other.dist.cmp(&self.dist)
    }
}
impl PartialOrd for MinDistNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone)]
pub struct IndexHNSW {
    dim: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    #[wasm_bindgen(skip)]
    pub nodes: HashMap<u32, Node>,
    pub enter_point: Option<u32>,
    pub max_layer: usize,
    metric: MetricType,
    level_mult: f32, // 1 / ln(M)
    #[wasm_bindgen(skip)]
    pub deleted: HashSet<u32>,
}

#[wasm_bindgen]
impl IndexHNSW {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, m: usize, ef_construction: usize, ef_search: usize, metric: MetricType) -> IndexHNSW {
        IndexHNSW {
            dim,
            m,
            ef_construction,
            ef_search,
            nodes: HashMap::new(),
            enter_point: None,
            max_layer: 0,
            metric,
            level_mult: 1.0 / (m as f32).ln(),
            deleted: HashSet::new(),
        }
    }
    
    pub fn delete(&mut self, id: u32) -> Result<(), JsValue> {
        self.deleted.insert(id);
        Ok(())
    }

    pub fn clear(&mut self) -> Result<(), JsValue> {
        self.nodes.clear();
        self.enter_point = None;
        self.max_layer = 0;
        self.deleted.clear();
        Ok(())
    }

    fn random_level(&self) -> usize {
        let mut r = rand::random::<f32>();
        if r == 0.0 { r = 0.0000001; } // avoid log(0)
        let level = (-r.ln() * self.level_mult) as usize;
        level
    }

    // No-op for HNSW, but required for generic trait compatibility
    pub fn train(&mut self, _data: &[f32]) -> Result<(), JsValue> {
        Ok(())
    }

    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            MetricType::L2 => l2_distance(a, b),
            MetricType::Cosine => 1.0 - super::metric::cosine_similarity(a, b), // Cosine Distance = 1 - Similarity
            MetricType::InnerProduct => -super::metric::inner_product(a, b), // Negative IP for Min-Heap
        }
    }

    pub fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue> {
        if vector.len() != self.dim { return Err(JsValue::from_str("Dim mismatch")); }
        
        // 1. Determine Level
        let level = self.random_level();
        let node = Node {
            id,
            vector: vector.clone(),
            layer: level,
            neighbors: vec![Vec::new(); level + 1],
        };
        self.nodes.insert(id, node);

        // 2. Insert Node
        if let Some(ep_id) = self.enter_point {
            let mut curr_ep = ep_id;
            // let curr_vec = &self.nodes[&curr_ep].vector;
            
            // Phase 1: Zoom in from top layer down to level+1
            // Greedy search for closest entry point at layer 'level'
            let max_l = self.nodes[&ep_id].layer;
            
            for lc in (level + 1..=max_l).rev() {
                 let mut cur_dist = self.dist(&vector, &self.nodes[&curr_ep].vector);
                 loop {
                     let mut changed = false;
                     if let Some(neighbors) = self.nodes[&curr_ep].neighbors.get(lc) {
                         for &n_id in neighbors {
                             let d = self.dist(&vector, &self.nodes[&n_id].vector);
                             if d < cur_dist {
                                 cur_dist = d;
                                 curr_ep = n_id;
                                 changed = true;
                             }
                         }
                     }
                     if !changed { break; }
                 }
            }
            
            // Phase 2: Insert from 'level' down to 0
            // Run search_layer for neighbors at each level
            // Phase 2: Insert from 'level' down to 0
            for lc in (0..=std::cmp::min(level, max_l)).rev() {
                // 1. Search for ef_construction candidates (nearest to query)
                let candidates = self.search_layer(curr_ep, &vector, self.ef_construction, lc);
                
                // 2. Select M neighbors using Robust Pruning Heuristic
                let selected_neighbors = self.select_neighbors(&vector, &candidates, self.m, lc);
                
                // 3. Connect (bidirectional)
                for &neighbor_id in &selected_neighbors {
                    self.nodes.get_mut(&id).unwrap().neighbors[lc].push(neighbor_id);
                    
                    // Add back-link to neighbor
                    let max_conn = if lc == 0 { self.m * 2 } else { self.m };
                    let prune_needed = {
                         if let Some(n_node) = self.nodes.get_mut(&neighbor_id) {
                             n_node.neighbors[lc].push(id);
                             n_node.neighbors[lc].len() > max_conn
                         } else {
                             false
                         }
                    };

                    if prune_needed {
                         self.prune_node(neighbor_id, lc, max_conn);
                    }
                }
                
                // 4. Update entry point for next layer (closest candidate)
                if let Some(first) = candidates.first() {
                    curr_ep = first.id;
                }
            }
        } else {
            // First node
            self.enter_point = Some(id);
            self.max_layer = level;
        }


        Ok(())
    }

    // Robust Pruning Helper (Breaks borrow cycle by copying vectors)
    fn prune_node(&mut self, node_id: u32, layer: usize, max_conn: usize) {
        let (node_vec, neighbors) = {
             let n = &self.nodes[&node_id];
             (n.vector.clone(), n.neighbors[layer].clone())
        };

        // 1. Gather Candidates with Vectors
        let mut candidates: Vec<(Candidate, Vec<f32>)> = Vec::with_capacity(neighbors.len());
        let metric = self.metric;
        
        for &nid in &neighbors {
             if let Some(n) = self.nodes.get(&nid) {
                 let d = Self::dist_static(&node_vec, &n.vector, metric);
                 candidates.push((Candidate { id: nid, distance: MinOrdFloat(d) }, n.vector.clone()));
             }
        }
        
        // 2. Sort Low -> High
        candidates.sort_by(|a, b| a.0.cmp(&b.0));

        // 3. Select with Heuristic
        let selected = Self::select_neighbors_heuristic(&node_vec, &candidates, max_conn, metric);
        
        // 4. Write back
        if let Some(n) = self.nodes.get_mut(&node_id) {
            n.neighbors[layer] = selected;
        }
    }
    
    // Pure function version of heuristic
    fn select_neighbors_heuristic(query: &[f32], candidates: &[(Candidate, Vec<f32>)], m: usize, metric: MetricType) -> Vec<u32> {
        let mut result = Vec::with_capacity(m);
        let mut result_vecs: Vec<&[f32]> = Vec::with_capacity(m); // Explicit type: Store slices
        
        for (cand, cand_vec) in candidates {
            if result.len() >= m { break; }
            
            let dist_to_query = cand.distance.0;
            let mut good = true;
            
            for res_vec in &result_vecs {
                let dist_cand_to_res = Self::dist_static(cand_vec, res_vec, metric);
                if dist_cand_to_res < dist_to_query {
                    good = false;
                    break;
                }
            }
            
            if good {
                result.push(cand.id);
                result_vecs.push(cand_vec);
            }
        }
        
        // Fallback: If heuristic dropped too many, fill with closest
        if result.len() < m { // Optional: WinnowDB preference strictly enforces M? No, usually M is max.
             for (cand, _) in candidates {
                 if result.len() >= m { break; }
                 if !result.contains(&cand.id) {
                     result.push(cand.id);
                 }
             }
        }
        
        result
    }

    fn dist_static(a: &[f32], b: &[f32], metric: MetricType) -> f32 {
        match metric {
            MetricType::L2 => l2_distance(a, b),
            MetricType::Cosine => 1.0 - super::metric::cosine_similarity(a, b),
            MetricType::InnerProduct => -super::metric::inner_product(a, b),
        }
    }

    // "Robust Pruning" Heuristic (Algorithm 4 in HNSW Paper)
    fn select_neighbors(&self, query: &[f32], candidates: &[Candidate], m: usize, _lc: usize) -> Vec<u32> {
        // Adapt to static heuristic
        let mut cand_data = Vec::with_capacity(candidates.len());
        for c in candidates {
             cand_data.push((Candidate { id: c.id, distance: c.distance.clone() }, self.nodes[&c.id].vector.clone()));
        }
        Self::select_neighbors_heuristic(query, &cand_data, m, self.metric)
    }

    fn search_layer(&self, start_ep: u32, query: &[f32], ef: usize, lc: usize) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new(); // MinHeap (nearest at top)
        let _results: BinaryHeap<DistNode> = BinaryHeap::new(); // MaxHeap (furthest at top to track ef limit)
        
        // Standard BFS-like greedy search
        // 1. Initialize v (visited) and C (candidates)
        let dist = self.dist(query, &self.nodes[&start_ep].vector);
        visited.insert(start_ep);
        candidates.push(Candidate { id: start_ep, distance: MinOrdFloat(dist) });
        
        // W: Dynamic list of found nearest neighbors (MinHeap for fixed size? No, MaxHeap to prune furthest)
        let mut w: BinaryHeap<DistNode> = BinaryHeap::new(); // MaxHeap: Worst candidate at top
        w.push(DistNode { id: start_ep, dist: OrderedFloat(dist) });
        
        while let Some(cand) = candidates.pop() {
             // cand is nearest in C
             let c = cand;
             
             // If c is further than the worst element in W, and W is full, we stop
             // But here we want to traverse. 
             // Standard HNSW condition: c.dist > f.dist (furthest in W) check
             if let Some(furthest) = w.peek() {
                 if c.distance.0 > furthest.dist.0 {
                     break; 
                 }
             }
             
             // Expand c
             if let Some(layer_neighbors) = self.nodes[&c.id].neighbors.get(lc) {
                 for &neighbor_id in layer_neighbors {
                     if !visited.contains(&neighbor_id) {
                         visited.insert(neighbor_id);
                         
                         let f_opt = w.peek();
                         let furthest_dist = if let Some(f) = f_opt { f.dist.0 } else { f32::MAX };
                         
                         let d = self.dist(query, &self.nodes[&neighbor_id].vector);
                         
                         if d < furthest_dist || w.len() < ef {
                             candidates.push(Candidate { id: neighbor_id, distance: MinOrdFloat(d) });
                             w.push(DistNode { id: neighbor_id, dist: OrderedFloat(d) });
                             
                             if w.len() > ef {
                                 w.pop(); // Remove furthest
                             }
                         }
                     }
                 }
             }
        }
        
        w.into_iter().map(|dn| Candidate { id: dn.id, distance: MinOrdFloat(dn.dist.0) })
                     .collect()
    }

    pub fn save(&self) -> Result<Vec<u8>, JsValue> {
        self.to_bytes()
    }

    pub fn load(data: &[u8]) -> Result<IndexHNSW, JsValue> {
        IndexHNSW::from_bytes(data)
    }
}

impl IndexHNSW {
    pub fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        if self.nodes.is_empty() { return Ok(serde_wasm_bindgen::to_value(&Vec::<(u32, f32)>::new())?); }
        
        let ep = self.enter_point.unwrap();
        let mut curr_ep = ep;
        let mut _dist = self.dist(&query, &self.nodes[&curr_ep].vector);
        
        // Descent Logic:
        for lc in (1..=self.max_layer).rev() {
             let mut best_dist = self.dist(&query, &self.nodes[&curr_ep].vector);
             let mut changed = true;
             while changed {
                 changed = false;
                 if let Some(neighbors) = self.nodes[&curr_ep].neighbors.get(lc) {
                     for &n_id in neighbors {
                         // Check Abort
                         if let Some(sig) = signal {
                             if sig.aborted() { return Err(JsValue::from_str("Search Aborted")); }
                         }

                         let d = self.dist(&query, &self.nodes[&n_id].vector);
                         if d < best_dist {
                             curr_ep = n_id;
                             best_dist = d;
                             changed = true;
                         }
                     }
                 }
             }
        }
        
        // Base layer
        let candidates = self.search_layer(curr_ep, &query, self.ef_search, 0);
        
        let mut sorted: Vec<(u32, f32)> = Vec::new();
        for c in candidates {
            if self.deleted.contains(&c.id) { continue; }
            if let Some(f) = filter { if !f.contains(&c.id) { continue; } }
            sorted.push((c.id, c.distance.0));
        }
        
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);
        
        Ok(serde_wasm_bindgen::to_value(&sorted)?)
    }

    /// Rust-native search for testing/benchmarking
    pub fn search_rust(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>) -> Vec<(u32, f32)> {
        if self.nodes.is_empty() { return Vec::new(); }
        
        let ep = self.enter_point.unwrap();
        let mut curr_ep = ep;
        let mut _dist = self.dist(&query, &self.nodes[&curr_ep].vector);
        
        for lc in (1..=self.max_layer).rev() {
             let mut best_dist = self.dist(&query, &self.nodes[&curr_ep].vector);
             let mut changed = true;
             while changed {
                 changed = false;
                 if let Some(neighbors) = self.nodes[&curr_ep].neighbors.get(lc) {
                     for &n_id in neighbors {
                         let d = self.dist(&query, &self.nodes[&n_id].vector);
                         if d < best_dist {
                             curr_ep = n_id;
                             best_dist = d;
                             changed = true;
                         }
                     }
                 }
             }
        }
        
        let candidates = self.search_layer(curr_ep, &query, self.ef_search, 0);
        
        let mut sorted: Vec<(u32, f32)> = Vec::new();
        for c in candidates {
            if self.deleted.contains(&c.id) { continue; }
            if let Some(f) = filter { if !f.contains(&c.id) { continue; } }
            sorted.push((c.id, c.distance.0));
        }
        
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(k).collect()
    }
}

use super::storage::VectorIndex;
impl VectorIndex for IndexHNSW {
    fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue> {
        self.add(id, vector)
    }

    fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        self.search(query, k, filter, signal)
    }
}

