use wasm_bindgen::prelude::*;
use web_sys::AbortSignal;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone)]
pub struct IndexSparse {
    // Inverted Index: TokenID -> List of (DocID, Score)
    // We use u32 for TokenID (hash of word or dictionnary index)
    inverted_index: HashMap<u32, Vec<(u32, f32)>>,
    
    // Document Norms for Cosine/BM25 normalization
    doc_norms: HashMap<u32, f32>,
    
    #[wasm_bindgen(skip)]
    pub deleted: HashSet<u32>,
}

#[wasm_bindgen]
impl IndexSparse {
    #[wasm_bindgen(constructor)]
    pub fn new() -> IndexSparse {
        IndexSparse {
            inverted_index: HashMap::new(),
            doc_norms: HashMap::new(),
            deleted: HashSet::new(),
        }
    }

    pub fn add(&mut self, id: u32, indices: &[u32], values: &[f32]) -> Result<(), JsValue> {
        if indices.len() != values.len() {
            return Err(JsValue::from_str("Sparse indices and values length mismatch"));
        }
        
        if self.deleted.contains(&id) {
            self.deleted.remove(&id);
        }

        // Compute norm for this document (L2 norm)
        let mut norm_sq = 0.0;
        for &v in values {
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        self.doc_norms.insert(id, norm);

        // Add to inverted index
        for (i, &token_id) in indices.iter().enumerate() {
            let score = values[i];
            
            let list = self.inverted_index.entry(token_id).or_insert(Vec::new());
            // Check if we need to update existing entry for this doc?
            // For Append-Only / Replacement logic:
            // Since we don't efficiently support update in Inverted Index (requires identifying old entry),
            // We usually assume new ID or we just append.
            // If ID reuse is common, we should check. For now, assume unique ID or handled by caller (Collection).
            list.push((id, score));
        }

        Ok(())
    }
    
    pub fn delete(&mut self, id: u32) -> Result<(), JsValue> {
        self.deleted.insert(id);
        // We don't remove from inverted index immediately (expensive).
        // Filter at search time.
        Ok(())
    }
    
    pub fn clear(&mut self) -> Result<(), JsValue> {
        self.inverted_index.clear();
        self.doc_norms.clear();
        self.deleted.clear();
        Ok(())
    }
}

impl IndexSparse {
    pub fn search(&self, query_indices: &[u32], query_values: &[f32], k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        if query_indices.len() != query_values.len() {
             return Err(JsValue::from_str("Query length mismatch"));
        }

        // Dot Product accumulation
        let mut scores: HashMap<u32, f32> = HashMap::new();
        
        for (i, &token_id) in query_indices.iter().enumerate() {
            // Check Abort
            if i % 100 == 0 {
                 if let Some(sig) = signal {
                     if sig.aborted() { return Err(JsValue::from_str("Search Aborted")); }
                 }
            }
            
            let q_val = query_values[i];
            
            if let Some(posting_list) = self.inverted_index.get(&token_id) {
                for &(doc_id, doc_val) in posting_list {
                    if self.deleted.contains(&doc_id) { continue; }
                    if let Some(f) = filter { if !f.contains(&doc_id) { continue; } }
                    
                    let entry = scores.entry(doc_id).or_insert(0.0);
                    *entry += q_val * doc_val;
                }
            }
        }
        
        // Convert to list and Optional Normalization (Cosine)
        let mut result: Vec<(u32, f32)> = Vec::with_capacity(scores.len());
        
        for (doc_id, score) in scores {
            let final_score = if let Some(&norm) = self.doc_norms.get(&doc_id) {
                if norm > 0.0 { score / norm } else { 0.0 }
            } else {
                score
            };
            result.push((doc_id, final_score));
        }
        
        // Sort descending
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let top_k: Vec<(u32, f32)> = result.into_iter().take(k).collect();
        
        Ok(serde_wasm_bindgen::to_value(&top_k)?)
    }
}
