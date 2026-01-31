//! Query Language for WinnowDB
//!
//! Provides SQL-like DSL, JSON query parsing, and query building.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ============================================================================
// Query AST
// ============================================================================

/// Search operation type
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SearchType {
    Dense,
    Sparse,
    Hybrid,
}

/// Filter condition operators
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FilterOp {
    Eq,      // =
    Ne,      // !=
    Gt,      // >
    Gte,     // >=
    Lt,      // <
    Lte,     // <=
    In,      // IN (...)
    NotIn,   // NOT IN (...)
    Contains, // CONTAINS
    StartsWith,
    EndsWith,
    Exists,
}

/// Filter value types
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterValue {
    String(String),
    Number(f64),
    Bool(bool),
    Array(Vec<FilterValue>),
    Null,
}

/// Single filter condition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilterCondition {
    pub field: String,
    pub op: FilterOp,
    pub value: FilterValue,
}

/// Compound filter (AND/OR)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Filter {
    Condition(FilterCondition),
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

/// Vector query options
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorQuery {
    pub vector: Option<Vec<f32>>,
    pub sparse_vector: Option<SparseVector>,
    pub collection: String,
    pub k: usize,
    pub filter: Option<Filter>,
    pub search_type: SearchType,
    pub include_payload: bool,
    pub include_vector: bool,
    pub score_threshold: Option<f32>,
    pub offset: usize,
}

/// Sparse vector representation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// Query result ordering
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderBy {
    pub field: String,
    pub descending: bool,
}

// ============================================================================
// Full Query AST
// ============================================================================

/// Complete query AST
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Query {
    pub operation: QueryOperation,
    pub collection: String,
    pub options: QueryOptions,
}

/// Query operation types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum QueryOperation {
    /// Vector search
    Search(VectorQuery),
    /// Get by IDs
    Get { ids: Vec<u32> },
    /// Count with filter
    Count { filter: Option<Filter> },
    /// Scroll through results
    Scroll { filter: Option<Filter>, limit: usize, offset: usize },
    /// Aggregate
    Aggregate { field: String, agg_type: AggregationType },
}

/// Aggregation types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AggregationType {
    Count,
    Min,
    Max,
    Avg,
    Sum,
    Terms { limit: usize },
}

/// Query options
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QueryOptions {
    pub timeout_ms: Option<u64>,
    pub consistency: Option<String>,
    pub read_concern: Option<String>,
}

// ============================================================================
// JSON DSL Parser
// ============================================================================

/// Parse JSON query DSL
pub fn parse_json_query(json: &str) -> Result<Query, String> {
    serde_json::from_str(json).map_err(|e| format!("Parse error: {}", e))
}

/// Build filter from JSON
pub fn parse_filter_json(json: &str) -> Result<Filter, String> {
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| format!("JSON parse error: {}", e))?;
    
    parse_filter_value(&value)
}

fn parse_filter_value(value: &serde_json::Value) -> Result<Filter, String> {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(and) = map.get("$and") {
                let conditions = and.as_array()
                    .ok_or("$and must be an array")?
                    .iter()
                    .map(parse_filter_value)
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(Filter::And(conditions));
            }
            
            if let Some(or) = map.get("$or") {
                let conditions = or.as_array()
                    .ok_or("$or must be an array")?
                    .iter()
                    .map(parse_filter_value)
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(Filter::Or(conditions));
            }
            
            if let Some(not) = map.get("$not") {
                let inner = parse_filter_value(not)?;
                return Ok(Filter::Not(Box::new(inner)));
            }
            
            // Field conditions
            let mut conditions = Vec::new();
            for (field, val) in map {
                if field.starts_with('$') {
                    continue;
                }
                
                match val {
                    serde_json::Value::Object(ops) => {
                        for (op_str, op_val) in ops {
                            let op = match op_str.as_str() {
                                "$eq" => FilterOp::Eq,
                                "$ne" => FilterOp::Ne,
                                "$gt" => FilterOp::Gt,
                                "$gte" => FilterOp::Gte,
                                "$lt" => FilterOp::Lt,
                                "$lte" => FilterOp::Lte,
                                "$in" => FilterOp::In,
                                "$nin" => FilterOp::NotIn,
                                "$contains" => FilterOp::Contains,
                                "$startsWith" => FilterOp::StartsWith,
                                "$endsWith" => FilterOp::EndsWith,
                                "$exists" => FilterOp::Exists,
                                _ => continue,
                            };
                            
                            let fv = json_to_filter_value(op_val);
                            conditions.push(Filter::Condition(FilterCondition {
                                field: field.clone(),
                                op,
                                value: fv,
                            }));
                        }
                    },
                    // Direct equality
                    _ => {
                        conditions.push(Filter::Condition(FilterCondition {
                            field: field.clone(),
                            op: FilterOp::Eq,
                            value: json_to_filter_value(val),
                        }));
                    }
                }
            }
            
            if conditions.len() == 1 {
                Ok(conditions.pop().unwrap())
            } else if conditions.is_empty() {
                Err("Empty filter".to_string())
            } else {
                Ok(Filter::And(conditions))
            }
        },
        _ => Err("Filter must be an object".to_string()),
    }
}

fn json_to_filter_value(val: &serde_json::Value) -> FilterValue {
    match val {
        serde_json::Value::String(s) => FilterValue::String(s.clone()),
        serde_json::Value::Number(n) => FilterValue::Number(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::Bool(b) => FilterValue::Bool(*b),
        serde_json::Value::Array(arr) => {
            FilterValue::Array(arr.iter().map(json_to_filter_value).collect())
        },
        serde_json::Value::Null => FilterValue::Null,
        serde_json::Value::Object(_) => FilterValue::Null,
    }
}

// ============================================================================
// SQL-like Parser (Simple)
// ============================================================================

/// Parse simple SQL-like query
/// Examples:
/// - SEARCH collection_name VECTOR [1.0, 2.0, 3.0] TOP 10
/// - SEARCH collection_name WHERE category = "tech" TOP 5
/// - COUNT collection_name WHERE status = "active"
pub fn parse_sql_query(sql: &str) -> Result<Query, String> {
    let tokens: Vec<&str> = sql.split_whitespace().collect();
    
    if tokens.is_empty() {
        return Err("Empty query".to_string());
    }
    
    match tokens[0].to_uppercase().as_str() {
        "SEARCH" => parse_search_sql(&tokens[1..]),
        "COUNT" => parse_count_sql(&tokens[1..]),
        "GET" => parse_get_sql(&tokens[1..]),
        _ => Err(format!("Unknown operation: {}", tokens[0])),
    }
}

fn parse_search_sql(tokens: &[&str]) -> Result<Query, String> {
    if tokens.is_empty() {
        return Err("SEARCH requires collection name".to_string());
    }
    
    let collection = tokens[0].to_string();
    let mut k = 10;
    let mut filter = None;
    let mut i = 1;
    
    while i < tokens.len() {
        match tokens[i].to_uppercase().as_str() {
            "TOP" | "LIMIT" => {
                if i + 1 < tokens.len() {
                    k = tokens[i + 1].parse().unwrap_or(10);
                    i += 2;
                } else {
                    i += 1;
                }
            },
            "WHERE" => {
                // Simple WHERE parsing: field op value
                if i + 3 < tokens.len() {
                    let field = tokens[i + 1].to_string();
                    let op = match tokens[i + 2] {
                        "=" | "==" => FilterOp::Eq,
                        "!=" | "<>" => FilterOp::Ne,
                        ">" => FilterOp::Gt,
                        ">=" => FilterOp::Gte,
                        "<" => FilterOp::Lt,
                        "<=" => FilterOp::Lte,
                        _ => FilterOp::Eq,
                    };
                    let value_str = tokens[i + 3].trim_matches('"').trim_matches('\'');
                    let value = if let Ok(n) = value_str.parse::<f64>() {
                        FilterValue::Number(n)
                    } else if value_str == "true" {
                        FilterValue::Bool(true)
                    } else if value_str == "false" {
                        FilterValue::Bool(false)
                    } else {
                        FilterValue::String(value_str.to_string())
                    };
                    
                    filter = Some(Filter::Condition(FilterCondition { field, op, value }));
                    i += 4;
                } else {
                    i += 1;
                }
            },
            _ => i += 1,
        }
    }
    
    Ok(Query {
        operation: QueryOperation::Search(VectorQuery {
            vector: None,
            sparse_vector: None,
            collection: collection.clone(),
            k,
            filter,
            search_type: SearchType::Dense,
            include_payload: true,
            include_vector: false,
            score_threshold: None,
            offset: 0,
        }),
        collection,
        options: QueryOptions::default(),
    })
}

fn parse_count_sql(tokens: &[&str]) -> Result<Query, String> {
    if tokens.is_empty() {
        return Err("COUNT requires collection name".to_string());
    }
    
    let collection = tokens[0].to_string();
    
    Ok(Query {
        operation: QueryOperation::Count { filter: None },
        collection,
        options: QueryOptions::default(),
    })
}

fn parse_get_sql(tokens: &[&str]) -> Result<Query, String> {
    if tokens.is_empty() {
        return Err("GET requires collection name".to_string());
    }
    
    let collection = tokens[0].to_string();
    let mut ids = Vec::new();
    
    for token in &tokens[1..] {
        if let Ok(id) = token.parse::<u32>() {
            ids.push(id);
        }
    }
    
    Ok(Query {
        operation: QueryOperation::Get { ids },
        collection,
        options: QueryOptions::default(),
    })
}

// ============================================================================
// Query Builder
// ============================================================================

/// Fluent query builder
#[derive(Clone, Debug, Default)]
pub struct QueryBuilder {
    collection: String,
    vector: Option<Vec<f32>>,
    sparse: Option<SparseVector>,
    k: usize,
    filter: Option<Filter>,
    search_type: SearchType,
    include_payload: bool,
    include_vector: bool,
    score_threshold: Option<f32>,
    offset: usize,
    timeout_ms: Option<u64>,
}

impl QueryBuilder {
    pub fn new(collection: &str) -> Self {
        Self {
            collection: collection.to_string(),
            k: 10,
            search_type: SearchType::Dense,
            include_payload: true,
            ..Default::default()
        }
    }
    
    pub fn vector(mut self, v: Vec<f32>) -> Self {
        self.vector = Some(v);
        self
    }
    
    pub fn sparse(mut self, indices: Vec<u32>, values: Vec<f32>) -> Self {
        self.sparse = Some(SparseVector { indices, values });
        self.search_type = SearchType::Sparse;
        self
    }
    
    pub fn hybrid(mut self, dense: Vec<f32>, sparse_indices: Vec<u32>, sparse_values: Vec<f32>) -> Self {
        self.vector = Some(dense);
        self.sparse = Some(SparseVector { indices: sparse_indices, values: sparse_values });
        self.search_type = SearchType::Hybrid;
        self
    }
    
    pub fn top_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }
    
    pub fn filter(mut self, f: Filter) -> Self {
        self.filter = Some(f);
        self
    }
    
    pub fn where_eq(self, field: &str, value: &str) -> Self {
        self.filter(Filter::Condition(FilterCondition {
            field: field.to_string(),
            op: FilterOp::Eq,
            value: FilterValue::String(value.to_string()),
        }))
    }
    
    pub fn where_gt(self, field: &str, value: f64) -> Self {
        self.filter(Filter::Condition(FilterCondition {
            field: field.to_string(),
            op: FilterOp::Gt,
            value: FilterValue::Number(value),
        }))
    }
    
    pub fn where_in(self, field: &str, values: Vec<String>) -> Self {
        self.filter(Filter::Condition(FilterCondition {
            field: field.to_string(),
            op: FilterOp::In,
            value: FilterValue::Array(values.into_iter().map(FilterValue::String).collect()),
        }))
    }
    
    pub fn include_payload(mut self, include: bool) -> Self {
        self.include_payload = include;
        self
    }
    
    pub fn include_vector(mut self, include: bool) -> Self {
        self.include_vector = include;
        self
    }
    
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = Some(threshold);
        self
    }
    
    pub fn offset(mut self, off: usize) -> Self {
        self.offset = off;
        self
    }
    
    pub fn timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }
    
    pub fn build(self) -> Query {
        Query {
            operation: QueryOperation::Search(VectorQuery {
                vector: self.vector,
                sparse_vector: self.sparse,
                collection: self.collection.clone(),
                k: self.k,
                filter: self.filter,
                search_type: self.search_type,
                include_payload: self.include_payload,
                include_vector: self.include_vector,
                score_threshold: self.score_threshold,
                offset: self.offset,
            }),
            collection: self.collection,
            options: QueryOptions {
                timeout_ms: self.timeout_ms,
                ..Default::default()
            },
        }
    }
}

impl Default for SearchType {
    fn default() -> Self {
        SearchType::Dense
    }
}

// ============================================================================
// WASM Bindings
// ============================================================================

#[wasm_bindgen]
pub struct QueryParser;

#[wasm_bindgen]
impl QueryParser {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self
    }
    
    /// Parse JSON query DSL
    pub fn parse_json(&self, json: &str) -> Result<JsValue, JsValue> {
        parse_json_query(json)
            .map(|q| serde_wasm_bindgen::to_value(&q).unwrap())
            .map_err(|e| JsValue::from_str(&e))
    }
    
    /// Parse SQL-like query
    pub fn parse_sql(&self, sql: &str) -> Result<JsValue, JsValue> {
        parse_sql_query(sql)
            .map(|q| serde_wasm_bindgen::to_value(&q).unwrap())
            .map_err(|e| JsValue::from_str(&e))
    }
    
    /// Parse filter from JSON
    pub fn parse_filter(&self, json: &str) -> Result<JsValue, JsValue> {
        parse_filter_json(json)
            .map(|f| serde_wasm_bindgen::to_value(&f).unwrap())
            .map_err(|e| JsValue::from_str(&e))
    }
    
    /// Build a search query
    pub fn build_search(&self, collection: &str, vector: Vec<f32>, k: usize) -> JsValue {
        let query = QueryBuilder::new(collection)
            .vector(vector)
            .top_k(k)
            .build();
        serde_wasm_bindgen::to_value(&query).unwrap_or(JsValue::NULL)
    }
    
    /// Build a filtered search query
    pub fn build_filtered_search(&self, collection: &str, vector: Vec<f32>, k: usize, filter_json: &str) -> Result<JsValue, JsValue> {
        let filter = parse_filter_json(filter_json)?;
        let query = QueryBuilder::new(collection)
            .vector(vector)
            .top_k(k)
            .filter(filter)
            .build();
        Ok(serde_wasm_bindgen::to_value(&query).unwrap())
    }
}

impl Default for QueryParser {
    fn default() -> Self {
        Self::new()
    }
}
