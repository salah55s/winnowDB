//! Advanced Search capabilities for WinnowDB
//!
//! Provides multi-vector search, batch operations, grouped search, and range queries.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// ============================================================================
// Multi-Vector Search
// ============================================================================

/// Multi-vector query for searching with multiple vectors
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiVectorQuery {
    /// Multiple query vectors
    pub vectors: Vec<Vec<f32>>,
    /// Combination strategy
    pub strategy: MultiVectorStrategy,
    /// Number of results
    pub k: usize,
}

/// Strategy for combining multiple vector results
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MultiVectorStrategy {
    /// Return union of results, re-ranked by best score
    Union,
    /// Return only results matching all vectors
    Intersection,
    /// Average scores across all vectors
    Average,
    /// Maximum score across all vectors
    Max,
    /// Minimum score (most restrictive)
    Min,
    /// Weighted combination
    Weighted { weights: Vec<f32> },
}

/// Result from multi-vector search
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiVectorResult {
    pub id: u32,
    pub score: f32,
    pub scores_per_vector: Vec<f32>,
}

/// Combine scores based on strategy
pub fn combine_scores(scores: &[f32], strategy: &MultiVectorStrategy) -> f32 {
    match strategy {
        MultiVectorStrategy::Union | MultiVectorStrategy::Max => {
            scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        },
        MultiVectorStrategy::Intersection | MultiVectorStrategy::Min => {
            scores.iter().cloned().fold(f32::INFINITY, f32::min)
        },
        MultiVectorStrategy::Average => {
            if scores.is_empty() { 0.0 } else { scores.iter().sum::<f32>() / scores.len() as f32 }
        },
        MultiVectorStrategy::Weighted { weights } => {
            scores.iter().zip(weights.iter())
                .map(|(s, w)| s * w)
                .sum::<f32>()
                / weights.iter().sum::<f32>()
        },
    }
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Batch search request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchSearchRequest {
    pub queries: Vec<BatchQuery>,
}

/// Individual query in a batch
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchQuery {
    pub id: String,
    pub vector: Vec<f32>,
    pub k: usize,
    pub filter: Option<serde_json::Value>,
}

/// Batch search result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchSearchResult {
    pub query_id: String,
    pub results: Vec<SearchHit>,
    pub latency_ms: f64,
    pub error: Option<String>,
}

/// Individual search hit
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: u32,
    pub score: f32,
    pub payload: Option<serde_json::Value>,
}

// ============================================================================
// Grouped Search
// ============================================================================

/// Grouped search request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroupedSearchRequest {
    pub vector: Vec<f32>,
    pub group_by: String,
    pub groups_limit: usize,
    pub group_size: usize,
    pub k: usize,
}

/// Single group result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchGroup {
    pub group_value: String,
    pub hits: Vec<SearchHit>,
    pub best_score: f32,
}

/// Grouped search result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroupedSearchResult {
    pub groups: Vec<SearchGroup>,
    pub total_groups: usize,
}

// ============================================================================
// Range Search
// ============================================================================

/// Range search request (score threshold instead of top-k)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeSearchRequest {
    pub vector: Vec<f32>,
    pub score_threshold: f32,
    pub limit: usize,
    pub order: RangeOrder,
}

/// Ordering for range results
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RangeOrder {
    ScoreDesc,
    ScoreAsc,
    IdAsc,
    IdDesc,
}

/// Range search result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeSearchResult {
    pub hits: Vec<SearchHit>,
    pub total_matching: usize,
}

// ============================================================================
// Scroll / Pagination
// ============================================================================

/// Scroll request for iterating through large result sets
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScrollRequest {
    pub filter: Option<serde_json::Value>,
    pub limit: usize,
    pub offset: String,
    pub order_by: Option<String>,
    pub with_payload: bool,
    pub with_vector: bool,
}

/// Scroll result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScrollResult {
    pub points: Vec<ScrollPoint>,
    pub next_offset: Option<String>,
    pub has_more: bool,
}

/// Point in scroll result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScrollPoint {
    pub id: u32,
    pub payload: Option<serde_json::Value>,
    pub vector: Option<Vec<f32>>,
}

// ============================================================================
// Recommend API
// ============================================================================

/// Recommendation request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecommendRequest {
    /// Positive examples (similar to these)
    pub positive_ids: Vec<u32>,
    /// Negative examples (not similar to these)
    pub negative_ids: Vec<u32>,
    /// Number of recommendations
    pub k: usize,
    /// Strategy for combining examples
    pub strategy: RecommendStrategy,
}

/// Recommendation strategy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RecommendStrategy {
    /// Average all positive, subtract average of negative
    AverageVector,
    /// Use best match from positives
    BestScore,
}

/// Recommendation result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecommendResult {
    pub recommendations: Vec<SearchHit>,
}

// ============================================================================
// Faceted Search
// ============================================================================

/// Faceted search request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FacetedSearchRequest {
    pub vector: Vec<f32>,
    pub k: usize,
    pub facets: Vec<FacetRequest>,
}

/// Individual facet request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FacetRequest {
    pub field: String,
    pub limit: usize,
}

/// Facet result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FacetResult {
    pub field: String,
    pub values: Vec<FacetValue>,
}

/// Facet value with count
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FacetValue {
    pub value: String,
    pub count: usize,
}

/// Faceted search result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FacetedSearchResult {
    pub hits: Vec<SearchHit>,
    pub facets: Vec<FacetResult>,
}

// ============================================================================
// WASM Bindings
// ============================================================================

#[wasm_bindgen]
pub struct AdvancedSearch {
    // Configuration
    default_k: usize,
}

#[wasm_bindgen]
impl AdvancedSearch {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            default_k: 10,
        }
    }
    
    /// Create a multi-vector query
    pub fn multi_vector_query(&self, vectors: JsValue, strategy: &str, k: usize) -> Result<JsValue, JsValue> {
        let vecs: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(vectors)?;
        
        let strategy = match strategy {
            "union" => MultiVectorStrategy::Union,
            "intersection" => MultiVectorStrategy::Intersection,
            "average" => MultiVectorStrategy::Average,
            "max" => MultiVectorStrategy::Max,
            "min" => MultiVectorStrategy::Min,
            _ => MultiVectorStrategy::Union,
        };
        
        let query = MultiVectorQuery { vectors: vecs, strategy, k };
        Ok(serde_wasm_bindgen::to_value(&query).unwrap())
    }
    
    /// Create a batch search request
    pub fn batch_search_request(&self, queries: JsValue) -> Result<JsValue, JsValue> {
        let qs: Vec<BatchQuery> = serde_wasm_bindgen::from_value(queries)?;
        let request = BatchSearchRequest { queries: qs };
        Ok(serde_wasm_bindgen::to_value(&request).unwrap())
    }
    
    /// Create a grouped search request
    pub fn grouped_search_request(&self, vector: Vec<f32>, group_by: &str, groups_limit: usize, group_size: usize) -> JsValue {
        let request = GroupedSearchRequest {
            vector,
            group_by: group_by.to_string(),
            groups_limit,
            group_size,
            k: groups_limit * group_size,
        };
        serde_wasm_bindgen::to_value(&request).unwrap_or(JsValue::NULL)
    }
    
    /// Create a range search request
    pub fn range_search_request(&self, vector: Vec<f32>, score_threshold: f32, limit: usize) -> JsValue {
        let request = RangeSearchRequest {
            vector,
            score_threshold,
            limit,
            order: RangeOrder::ScoreDesc,
        };
        serde_wasm_bindgen::to_value(&request).unwrap_or(JsValue::NULL)
    }
    
    /// Create a recommend request
    pub fn recommend_request(&self, positive_ids: Vec<u32>, negative_ids: Vec<u32>, k: usize) -> JsValue {
        let request = RecommendRequest {
            positive_ids,
            negative_ids,
            k,
            strategy: RecommendStrategy::AverageVector,
        };
        serde_wasm_bindgen::to_value(&request).unwrap_or(JsValue::NULL)
    }
    
    /// Create a scroll request
    pub fn scroll_request(&self, limit: usize, offset: &str) -> JsValue {
        let request = ScrollRequest {
            filter: None,
            limit,
            offset: offset.to_string(),
            order_by: None,
            with_payload: true,
            with_vector: false,
        };
        serde_wasm_bindgen::to_value(&request).unwrap_or(JsValue::NULL)
    }
    
    /// Create a faceted search request
    pub fn faceted_search_request(&self, vector: Vec<f32>, k: usize, facet_fields: Vec<String>) -> JsValue {
        let facets: Vec<FacetRequest> = facet_fields.into_iter()
            .map(|f| FacetRequest { field: f, limit: 10 })
            .collect();
        
        let request = FacetedSearchRequest { vector, k, facets };
        serde_wasm_bindgen::to_value(&request).unwrap_or(JsValue::NULL)
    }
    
    /// Combine multi-vector scores
    pub fn combine_multi_vector_scores(&self, scores: Vec<f32>, strategy: &str) -> f32 {
        let strat = match strategy {
            "union" | "max" => MultiVectorStrategy::Max,
            "intersection" | "min" => MultiVectorStrategy::Min,
            "average" => MultiVectorStrategy::Average,
            _ => MultiVectorStrategy::Average,
        };
        combine_scores(&scores, &strat)
    }
    
    /// Group search results by field
    pub fn group_results(&self, results: JsValue, group_by: &str, group_size: usize) -> Result<JsValue, JsValue> {
        let hits: Vec<SearchHit> = serde_wasm_bindgen::from_value(results)?;
        let mut groups: HashMap<String, Vec<SearchHit>> = HashMap::new();
        
        for hit in hits {
            let group_value = hit.payload.as_ref()
                .and_then(|p| p.get(group_by))
                .and_then(|v| v.as_str())
                .unwrap_or("_default")
                .to_string();
            
            groups.entry(group_value).or_insert_with(Vec::new).push(hit);
        }
        
        let search_groups: Vec<SearchGroup> = groups.into_iter()
            .map(|(value, mut hits)| {
                hits.truncate(group_size);
                let best_score = hits.iter().map(|h| h.score).fold(0.0f32, f32::max);
                SearchGroup {
                    group_value: value,
                    hits,
                    best_score,
                }
            })
            .collect();
        
        let result = GroupedSearchResult {
            total_groups: search_groups.len(),
            groups: search_groups,
        };
        
        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }
}

impl Default for AdvancedSearch {
    fn default() -> Self {
        Self::new()
    }
}
