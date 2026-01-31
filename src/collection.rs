use wasm_bindgen::prelude::*;
use web_sys::AbortSignal;
use wasm_bindgen_futures::JsFuture;
use js_sys::Promise;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use crate::vector::enum_wrapper::{VectorIndexEnum, VectorIndexTrait};
use crate::vector::storage::VectorStorage;
use crate::vector::metric::MetricType;
use crate::error::WinnowError;

// Import specific indices
use crate::vector::ivf_pq::IndexIVFPQ;
use crate::vector::sq::IndexSQ;
use crate::vector::bq::IndexBQ;
use crate::vector::hnsw::IndexHNSW;
use crate::vector::sparse::IndexSparse;
use crate::filter::matches; // Import filter logic
use crate::observability::Metrics;

use crate::storage::op_wal::{OpWal, WalEntry};
use crate::storage::{FileSystemSyncAccessHandle, PayloadStore, wrap_snapshot, unwrap_snapshot, SnapshotError};
use crate::resilience::{CircuitBreaker, CircuitBreakerConfig}; 
use crate::security::RateLimiter;
use std::sync::Arc;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

struct WriteGuard {
    counter: Arc<AtomicUsize>,
}

impl WriteGuard {
    fn try_acquire(counter: Arc<AtomicUsize>, limit: usize) -> Result<Self, JsValue> {
        let current = counter.fetch_add(1, Ordering::SeqCst);
        if current >= limit {
            counter.fetch_sub(1, Ordering::SeqCst);
            return Err(JsValue::from_str("Write Queue Full (Backpressure)"));
        }
        Ok(Self { counter })
    }
}

impl Drop for WriteGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
    }
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum IndexType {
    HNSW = 0,
    IVFPQ = 1,
    SQ = 2,
    BQ = 3,
}

/// Options for search with graceful degradation
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct SearchOptions {
    /// Return partial results if timeout/error occurs
    pub allow_partial: bool,
    /// Maximum time in milliseconds before returning partial results
    pub timeout_ms: Option<u32>,
    /// Enable best-effort mode (ignore non-critical errors)
    pub best_effort: bool,
}

/// Result with metadata for graceful degradation
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SearchResult {
    /// Search results as (id, score) pairs
    pub results: Vec<(u32, f32)>,
    /// True if results are incomplete due to timeout/error
    pub is_partial: bool,
    /// Number of vectors scanned
    pub scanned_count: usize,
    /// Total vectors in index
    pub total_count: usize,
    /// Latency in milliseconds
    pub latency_ms: f64,
    /// Error message if any
    pub error: Option<String>,
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone)]
pub struct CollectionConfig {
    pub dim: usize,
    pub max_elements: usize,
    pub index_type: IndexType,
    pub metric: Option<MetricType>, 
    // Sparse
    pub sparse_enabled: Option<bool>,
    // HNSW specific
    pub m: Option<usize>,
    pub ef_construction: Option<usize>,
    pub ef_search: Option<usize>, 
    // IVF-PQ specific
    pub n_centroids: Option<usize>,
    pub n_sub: Option<usize>,
    
    // Operational
    pub max_memory: Option<usize>, // Max bytes (default 512MB)
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Payload {
    InMemory(serde_json::Value),
    OnDisk { offset: u64, len: u32 },
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct WinnowCollection {
    name: String,
    config: CollectionConfig,
    index: Arc<RwLock<VectorIndexEnum>>, 
    sparse_index: Arc<RwLock<Option<IndexSparse>>>,
    payloads: Arc<RwLock<HashMap<u32, Payload>>>,
    wal: Arc<RwLock<Option<OpWal>>>, 
    payload_store: Arc<RwLock<Option<PayloadStore>>>,
    active_ids: Arc<RwLock<HashSet<u32>>>, 
    memory_usage: Arc<RwLock<usize>>,
    active_writes: Arc<AtomicUsize>,
    metrics: Arc<Metrics>,
    breaker: Arc<CircuitBreaker>,
    rate_limiter: Arc<RateLimiter>,
    vectors: Arc<RwLock<VectorStorage>>,
}

#[wasm_bindgen]
impl WinnowCollection {
    #[wasm_bindgen(constructor)]
    pub fn new(name: String, config: JsValue, wal_handle: JsValue) -> Result<WinnowCollection, JsValue> {
        let config: CollectionConfig = serde_wasm_bindgen::from_value(config)?;
        
        let mut index = Self::create_index_internal(&config)?;
        
        let mut sparse_index = if config.sparse_enabled.unwrap_or(false) {
            Some(IndexSparse::new())
        } else {
            None
        };

        let mut payloads = HashMap::new();
        let mut active_ids = HashSet::new(); // Initialize
        let mut wal = None;
        let memory_usage = 0;

        // Initialize WAL if handle provided
        if !wal_handle.is_undefined() && !wal_handle.is_null() {
            let handle = FileSystemSyncAccessHandle::unchecked_from_js(wal_handle);
            let mut op_wal = OpWal::new(handle).map_err(|e| JsValue::from_str(&e.to_string()))?;
            
            // RECOVERY
            let entries = op_wal.recover().map_err(|e| JsValue::from_str(&e.to_string()))?;
            web_sys::console::log_1(&JsValue::from_str(&format!("Recovered {} entries from WAL", entries.len())));
            
            for entry in entries {
                match entry {
                    WalEntry::Insert { id, vector, sparse_indices, sparse_values, payload } => {
                        // Hardened Recovery: Ensure Upsert Semantics.
                        // If we encounter an Insert for an already-active ID (e.g. caused by crash before Delete was logged,
                        // or duplicate Inserts), we must remove the stale version first to ensure Index consistency.
                        // Since we are rebuilding, 'active_ids' tracks what we have restored so far.
                        if active_ids.contains(&id) {
                            index.delete(id)?;
                            if let Some(sparse) = &mut sparse_index {
                                sparse.delete(id)?;
                            }
                            payloads.remove(&id);
                            // active_ids is already set, but we re-insert later.
                        }

                        index.add(id, vector)?; 
                        
                        if let Some(sparse) = &mut sparse_index {
                            if let (Some(idxs), Some(vals)) = (sparse_indices, sparse_values) {
                                sparse.add(id, &idxs, &vals)?;
                            }
                        }
                        
                        if let Some(p) = payload {
                            // Try parsing as JSON, fallback to String
                            let val = serde_json::from_str(&p).unwrap_or(serde_json::Value::String(p));
                            payloads.insert(id, Payload::InMemory(val));
                        }
                        active_ids.insert(id);
                    },
                    WalEntry::Delete { id } => {
                        index.delete(id)?;
                        if let Some(sparse) = &mut sparse_index {
                             sparse.delete(id)?;
                        }
                        payloads.remove(&id);
                        active_ids.remove(&id);
                    },
                    WalEntry::Clear => {
                        index.clear()?;
                        if let Some(sparse) = &mut sparse_index {
                             sparse.clear()?;
                        }
                        payloads.clear();
                        active_ids.clear();
                    }
                }
            }
            wal = Some(op_wal);
        }

        Ok(WinnowCollection {
            name,
            config,
            index: Arc::new(RwLock::new(index)),
            sparse_index: Arc::new(RwLock::new(sparse_index)),
            payloads: Arc::new(RwLock::new(payloads)),
            wal: Arc::new(RwLock::new(wal)),
            payload_store: Arc::new(RwLock::new(None)),
            active_ids: Arc::new(RwLock::new(active_ids)),
            memory_usage: Arc::new(RwLock::new(memory_usage)),
            active_writes: Arc::new(AtomicUsize::new(0)),
            metrics: Arc::new(Metrics::new()),
            breaker: Arc::new(CircuitBreaker::new(CircuitBreakerConfig::default())),
            rate_limiter: Arc::new(RateLimiter::new(100000.0, 10000.0)),
            vectors: Arc::new(RwLock::new(VectorStorage::new())),
        })
    }

    pub fn add(&mut self, id: u32, vector: Vec<f32>, sparse_indices: Option<Vec<u32>>, sparse_values: Option<Vec<f32>>, payload: JsValue) -> Result<(), JsValue> {
        // Dimension validation
        if vector.len() != self.config.dim {
            return Err(WinnowError::dimension_mismatch(self.config.dim, vector.len()).into());
        }
        
        // Sparse vector validation
        if let (Some(ref idx), Some(ref val)) = (&sparse_indices, &sparse_values) {
            if idx.len() != val.len() {
                return Err(WinnowError::invalid_filter(
                    &format!("Sparse indices length ({}) != values length ({})", idx.len(), val.len())
                ).into());
            }
        }
        
        // Backpressure check
        let _guard = WriteGuard::try_acquire(self.active_writes.clone(), 10)
            .map_err(|_| WinnowError::rate_limit_exceeded(1000))?;
        
        // Rate Limit Check
        self.rate_limiter.check()
            .map_err(|_| WinnowError::rate_limit_exceeded(1000))?;

        self.metrics.increment_insert(); 

        // Acquire Write Locks (Global for atomicity)
        let mut index = self.index.write();
        let mut sparse_lock = self.sparse_index.write();
        let mut payloads = self.payloads.write();
        let mut active = self.active_ids.write();
        let mut wal_lock = self.wal.write();
        let mut mem = self.memory_usage.write();
        let mut store_lock = self.payload_store.write(); // Acquire store lock

        // UPSERT Logic (Inline to avoid deadlock/race)
        if active.contains(&id) {
            // Delete existing data
            if let Some(payload_enum) = payloads.remove(&id) {
                if let Payload::InMemory(val) = payload_enum {
                    if let Ok(s) = serde_json::to_string(&val) {
                        if *mem >= s.len() { *mem -= s.len(); }
                    }
                }
                // If OnDisk, we don't track its size in 'mem', so no decrement needed.
            }
            index.delete(id)?;
            if let Some(sparse) = sparse_lock.as_mut() {
                sparse.delete(id)?;
            }
            // active set remains 'contains', prevents flake.
            // But we must decrement mem for vector
            let vec_size = self.config.dim * 4;
            if *mem >= vec_size { *mem -= vec_size; }
            
            // Log Delete to WAL? 
            // Strictly speaking, Upsert = Delete + Insert in log, OR just Insert (if we rely on latest wins).
            // But if we crash after Delete logic but before Insert logic? 
            // In memory, we are holding locks, so crash means everything lost.
            // On disk (WAL), we should append Insert. Replay handles overwrite (assuming we fixed recovery).
            // My previous recovery fix handled duplicates.
            // So we don't strictly need to log 'Delete' here if we log 'Insert' immediately after.
        }

        // 0. Check Memory Budget
        let vector_size = vector.len() * 4;
        let sparse_size = if let (Some(idx), Some(val)) = (&sparse_indices, &sparse_values) {
            idx.len() * 4 + val.len() * 4
        } else { 0 };
        // Estimate payload size (rough)
        let payload_size = 0; // Calculated later

        let estimated_total = vector_size + sparse_size + payload_size;
        let limit = self.config.max_memory.unwrap_or(512 * 1024 * 1024);
        
        if *mem + estimated_total > limit {
             // Try to evict
             if let Some(store) = store_lock.as_mut() {
                 let needed = (*mem + estimated_total) - limit;
                 // Target slightly more to avoid thrashing? e.g. needed * 1.5
                 let target = needed + (1024 * 1024); 
                 
                 let _freed = Self::evict_payloads_internal(&mut payloads, store, &mut mem, target)?;
             }
             
             // Check again
             if *mem + estimated_total > limit {
                  return Err(JsValue::from_str("Memory Limit Exceeded (Eviction Failed or No Store)"));
             }
        }
        
        // Handle Payload conversion
        let (payload_string, payload_value) = if !payload.is_undefined() && !payload.is_null() {
            let val: serde_json::Value = serde_wasm_bindgen::from_value(payload.clone())?;
            let s = serde_json::to_string(&val).map_err(|e| JsValue::from_str(&e.to_string()))?;
            (Some(s), Some(val))
        } else {
            (None, None)
        };

        // 1. Write to WAL (Stringified)
        if let Some(wal) = wal_lock.as_mut() {
            let entry = WalEntry::Insert { 
                id, 
                vector: vector.clone(), 
                sparse_indices: sparse_indices.clone(),
                sparse_values: sparse_values.clone(),
                payload: payload_string.clone() 
            };
            wal.append(&entry).map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        // 2. Add to Dense Index
        index.add(id, vector.clone())?;
        
        // 3. Add to Sparse Index
        if let Some(sparse) = sparse_lock.as_mut() {
             if let (Some(idxs), Some(vals)) = (&sparse_indices, &sparse_values) {
                 sparse.add(id, idxs, vals)?;
             }
        }
        
        // 4. Store Payload (Value)
        let mut added_payload_size = 0;
        if let Some(val) = payload_value {
            if let Some(s) = &payload_string {
                added_payload_size = s.len();
            }
            payloads.insert(id, Payload::InMemory(val));
        }
        
        // 5. Mark Active & Update Usage
        active.insert(id);
        self.vectors.write().add(id, vector.clone()); // Store raw vector
        *mem += vector_size + sparse_size + added_payload_size;
        self.metrics.set_memory(*mem as u64);
        self.metrics.set_vector_count(active.len() as u64);
        
        Ok(())
    }

    pub fn delete(&mut self, id: u32) -> Result<(), JsValue> {
        self.metrics.increment_delete();
        
        // Check if ID exists
        {
            let active = self.active_ids.read();
            if !active.contains(&id) {
                return Err(WinnowError::not_found(id).into());
            }
        }
        
        let mut index = self.index.write();
        let mut sparse_lock = self.sparse_index.write();
        let mut payloads = self.payloads.write();
        let mut active = self.active_ids.write();
        let mut wal_lock = self.wal.write();
        let mut mem = self.memory_usage.write();

        // 1. Write to WAL
        if let Some(wal) = wal_lock.as_mut() {
            let entry = WalEntry::Delete { id };
            wal.append(&entry).map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        // 2. Delete from Indices
        index.delete(id)?;
        if let Some(sparse) = sparse_lock.as_mut() {
             sparse.delete(id)?;
        }

        // 3. Remove Payload and Active
        if let Some(payload_enum) = payloads.remove(&id) {
            if let Payload::InMemory(val) = payload_enum {
                if let Ok(s) = serde_json::to_string(&val) {
                    if *mem >= s.len() { *mem -= s.len(); }
                }
            }
        }
        active.remove(&id);
        self.vectors.write().remove(id); // Remove from storage
        
        let vec_size = self.config.dim * 4;
        if *mem >= vec_size { *mem -= vec_size; }
        self.metrics.set_memory(*mem as u64);
        self.metrics.set_vector_count(active.len() as u64);
        
        Ok(())
    }

    pub fn clear(&mut self) -> Result<(), JsValue> {
        let mut index = self.index.write();
        let mut sparse_lock = self.sparse_index.write();
        let mut payloads = self.payloads.write();
        let mut active = self.active_ids.write();
        let mut wal_lock = self.wal.write();
        let mut mem = self.memory_usage.write();

        // 1. Write to WAL
        if let Some(wal) = wal_lock.as_mut() {
            let entry = WalEntry::Clear;
            wal.append(&entry).map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        
        // 2. Clear Indices and Payloads
        index.clear()?;
        if let Some(sparse) = sparse_lock.as_mut() {
             sparse.clear()?;
        }
        payloads.clear();
        active.clear();
        self.vectors.write().clear(); // Clear storage
        *mem = 0;
        
        Ok(())
    }
    
    pub fn count(&self) -> usize {
        self.active_ids.read().len()
    }

    pub fn search(&self, query: Vec<f32>, k: usize, filter: Option<JsValue>, signal: Option<AbortSignal>) -> Result<JsValue, JsValue> {
        // Dimension validation
        if query.len() != self.config.dim {
            return Err(WinnowError::dimension_mismatch(self.config.dim, query.len()).into());
        }
        
        // Clamp k to valid range
        let count = self.active_ids.read().len();
        let k = if k == 0 { 1 } else { k.min(count.max(1)) };
        
        // Rate Limit Check
        self.rate_limiter.check()
            .map_err(|_| WinnowError::rate_limit_exceeded(1000))?;
        
        let start = js_sys::Date::now();
        self.metrics.increment_search();
        
        let allow_list = self.process_filter(filter)?;
        
        // Circuit Breaker Wrapper
        let result = self.breaker.call(|| {
            let index = self.index.read();
            index.search(query, k, allow_list.as_ref(), signal.as_ref())
                 .map_err(|e| e.as_string().unwrap_or_else(|| "Unknown JS Error".to_string()))
        });

        let duration = js_sys::Date::now() - start;
        self.metrics.observe_search_latency((duration * 1000.0) as u64);
        
        // Record slow query if threshold exceeded
        let filtered = allow_list.is_some();
        self.metrics.record_slow_query(duration, k, filtered);

        match result {
            Ok(val) => Ok(val),
            Err(e) => {
                self.metrics.increment_error();
                Err(JsValue::from_str(&e))
            }
        }
    }

    pub fn search_sparse(&self, query_indices: Vec<u32>, query_values: Vec<f32>, k: usize, filter: Option<JsValue>, signal: Option<AbortSignal>) -> Result<JsValue, JsValue> {
        let allow_list = self.process_filter(filter)?;
        let sparse_lock = self.sparse_index.read();
        if let Some(sparse) = sparse_lock.as_ref() {
            sparse.search(&query_indices, &query_values, k, allow_list.as_ref(), signal.as_ref())
        } else {
            Err(JsValue::from_str("Sparse index not enabled"))
        }
    }
    
    /// Search with graceful degradation - returns partial results on timeout/error
    pub fn search_graceful(&self, query: Vec<f32>, k: usize, filter: Option<JsValue>, options: JsValue) -> Result<JsValue, JsValue> {
        let opts: SearchOptions = if options.is_null() || options.is_undefined() {
            SearchOptions::default()
        } else {
            serde_wasm_bindgen::from_value(options)?
        };
        
        // Dimension validation
        if query.len() != self.config.dim {
            if opts.best_effort {
                return Ok(serde_wasm_bindgen::to_value(&SearchResult {
                    results: vec![],
                    is_partial: true,
                    scanned_count: 0,
                    total_count: self.active_ids.read().len(),
                    latency_ms: 0.0,
                    error: Some(format!("Dimension mismatch: expected {}, got {}", self.config.dim, query.len())),
                })?);
            }
            return Err(WinnowError::dimension_mismatch(self.config.dim, query.len()).into());
        }
        
        let total_count = self.active_ids.read().len();
        let k = if k == 0 { 1 } else { k.min(total_count.max(1)) };
        
        let start = js_sys::Date::now();
        self.metrics.increment_search();
        
        let allow_list = match self.process_filter(filter) {
            Ok(al) => al,
            Err(e) => {
                if opts.best_effort {
                    None // Ignore filter error in best-effort mode
                } else if opts.allow_partial {
                    return Ok(serde_wasm_bindgen::to_value(&SearchResult {
                        results: vec![],
                        is_partial: true,
                        scanned_count: 0,
                        total_count,
                        latency_ms: js_sys::Date::now() - start,
                        error: Some(e.as_string().unwrap_or_else(|| "Filter error".to_string())),
                    })?);
                } else {
                    return Err(e);
                }
            }
        };
        
        // Execute search with circuit breaker
        let result = self.breaker.call(|| {
            let index = self.index.read();
            index.search(query, k, allow_list.as_ref(), None) // No signal - we handle timeout differently
                 .map_err(|e| e.as_string().unwrap_or_else(|| "Search error".to_string()))
        });
        
        let latency_ms = js_sys::Date::now() - start;
        self.metrics.observe_search_latency((latency_ms * 1000.0) as u64);
        
        match result {
            Ok(val) => {
                // Parse results and wrap in SearchResult
                let results: Vec<(u32, f32)> = serde_wasm_bindgen::from_value(val)?;
                let scanned_count = results.len();
                
                Ok(serde_wasm_bindgen::to_value(&SearchResult {
                    results,
                    is_partial: false,
                    scanned_count,
                    total_count,
                    latency_ms,
                    error: None,
                })?)
            },
            Err(e) => {
                self.metrics.increment_error();
                
                if opts.allow_partial || opts.best_effort {
                    // Return partial result with error
                    Ok(serde_wasm_bindgen::to_value(&SearchResult {
                        results: vec![],
                        is_partial: true,
                        scanned_count: 0,
                        total_count,
                        latency_ms,
                        error: Some(e),
                    })?)
                } else {
                    Err(JsValue::from_str(&e))
                }
            }
        }
    }

    pub fn search_hybrid(&self, query: Vec<f32>, query_indices: Vec<u32>, query_values: Vec<f32>, k: usize, filter: Option<JsValue>, signal: Option<AbortSignal>) -> Result<JsValue, JsValue> {
        // Dimension validation
        if query.len() != self.config.dim {
            return Err(WinnowError::dimension_mismatch(self.config.dim, query.len()).into());
        }
        
        // Sparse query validation
        if query_indices.len() != query_values.len() {
            return Err(WinnowError::invalid_filter(
                &format!("Sparse query indices ({}) != values ({})", query_indices.len(), query_values.len())
            ).into());
        }
        
        let allow_list = self.process_filter(filter)?;
        let allow_ref = allow_list.as_ref();
        let sig_ref = signal.as_ref();

        // 1. Dense Search 
        let k_factor = 2;
        let dense_results: Vec<(u32, f32)> = {
             let index = self.index.read();
             let dense_js = index.search(query, k * k_factor, allow_ref, sig_ref)?;
             serde_wasm_bindgen::from_value(dense_js)?
        };

        // Check Abort between stages
        if let Some(s) = sig_ref {
            if s.aborted() { return Err(JsValue::from_str("Search Aborted")); }
        }

        // 2. Sparse Search
        let sparse_results: Vec<(u32, f32)> = {
             let sparse_lock = self.sparse_index.read();
             if let Some(sparse) = sparse_lock.as_ref() {
                 let sparse_js = sparse.search(&query_indices, &query_values, k * k_factor, allow_ref, sig_ref)?;
                 serde_wasm_bindgen::from_value(sparse_js)?
             } else {
                 Vec::new() 
             }
        };

        // 3. RRF Fusion
        let mut rrf_scores: HashMap<u32, f32> = HashMap::new();
        let c = 60.0;
        
        for (rank, (id, _)) in dense_results.iter().enumerate() {
            let score = 1.0 / (c + rank as f32);
            *rrf_scores.entry(*id).or_insert(0.0) += score;
        }

        for (rank, (id, _)) in sparse_results.iter().enumerate() {
            let score = 1.0 / (c + rank as f32);
            *rrf_scores.entry(*id).or_insert(0.0) += score;
        }
        
        // 4. Sort and Top K
        let mut final_results: Vec<(u32, f32)> = rrf_scores.into_iter().collect();
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let top_k: Vec<(u32, f32)> = final_results.into_iter().take(k).collect();
        
        Ok(serde_wasm_bindgen::to_value(&top_k)?)
    }

    pub fn get_payload(&self, id: u32) -> JsValue {
        // 4. Readback
        let payloads = self.payloads.read();
        let payload_store_lock = self.payload_store.read();
        
        if let Some(payload_enum) = payloads.get(&id) {
             match payload_enum {
                 Payload::InMemory(val) => serde_wasm_bindgen::to_value(val).unwrap_or(JsValue::NULL),
                 Payload::OnDisk { offset, len } => {
                     // Load from store
                     if let Some(store) = payload_store_lock.as_ref() {
                         if let Ok(bytes) = store.load(*offset, *len) {
                             if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                                 return serde_wasm_bindgen::to_value(&val).unwrap_or(JsValue::NULL);
                             }
                         }
                     }
                     JsValue::NULL // Store missing or load failed
                 }
             }
        } else {
             JsValue::NULL
        }
    }

    /// Get raw vector by ID
    pub fn get_vector(&self, id: u32) -> Result<JsValue, JsValue> {
        let vectors = self.vectors.read();
        if let Some(vec) = vectors.get(id) {
            Ok(serde_wasm_bindgen::to_value(vec)?)
        } else {
            Err(WinnowError::not_found(id).into())
        }
    }
    
    /// Check if ID exists
    pub fn exists(&self, id: u32) -> bool {
        self.active_ids.read().contains(&id)
    }
    
    /// Update payload without re-inserting vector
    pub fn update_payload(&mut self, id: u32, payload: JsValue) -> Result<(), JsValue> {
        if !self.active_ids.read().contains(&id) {
            return Err(WinnowError::not_found(id).into());
        }
        
        let val: serde_json::Value = serde_wasm_bindgen::from_value(payload)?;
        let mut payloads = self.payloads.write();
        payloads.insert(id, Payload::InMemory(val));
        
        Ok(())
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.config).unwrap_or(JsValue::NULL)
    }
    
    /// Get collection name
    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn train(&mut self, data: &[f32]) -> Result<(), JsValue> {
        // Validate training data
        if data.is_empty() {
            return Err(WinnowError::internal("Training data is empty").into());
        }
        if data.len() % self.config.dim != 0 {
            return Err(WinnowError::dimension_mismatch(
                self.config.dim, 
                data.len() % self.config.dim
            ).into());
        }
        
        let mut index = self.index.write();
        index.train(data)
    }
    
    // Internal helper for filtering
    fn process_filter(&self, filter: Option<JsValue>) -> Result<Option<HashSet<u32>>, JsValue> {
        if let Some(f) = filter {
            if f.is_null() || f.is_undefined() { return Ok(None); }
            
            let filter_val: serde_json::Value = serde_wasm_bindgen::from_value(f)?;
            let mut matched_ids = HashSet::new();
            
            let active = self.active_ids.read();
            let payloads = self.payloads.read();
            let payload_store_lock = self.payload_store.read();

            // Linear Scan (Pre-filter)
            for &id in active.iter() {
                if let Some(payload_enum) = payloads.get(&id) {
                    match payload_enum {
                        Payload::InMemory(val) => {
                            if matches(val, &filter_val) {
                                matched_ids.insert(id);
                            }
                        },
                        Payload::OnDisk { offset, len } => {
                            // Load OnDisk payload for filtering
                            if let Some(store) = payload_store_lock.as_ref() {
                                if let Ok(bytes) = store.load(*offset, *len) {
                                    if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                                        if matches(&val, &filter_val) {
                                            matched_ids.insert(id);
                                        }
                                    }
                                }
                            }
                            // If store missing or load failed, exclude from results (conservative)
                        }
                    }
                }
            }
            Ok(Some(matched_ids))
        } else {
            Ok(None)
        }
    }

    pub fn set_payload_store(&self, handle: JsValue) -> Result<(), JsValue> {
        if !handle.is_undefined() && !handle.is_null() {
             let fs_handle = FileSystemSyncAccessHandle::unchecked_from_js(handle);
             let store = PayloadStore::new(fs_handle).map_err(|e| JsValue::from_str(&e.to_string()))?;
             let mut lock = self.payload_store.write();
             *lock = Some(store);
             Ok(())
        } else {
             Err(JsValue::from_str("Invalid Handle"))
        }
    }

    /// Attempts to free memory by moving payloads to disk.
    /// Internal helper: Assumes locks are held.
    fn evict_payloads_internal(payloads: &mut HashMap<u32, Payload>, store: &mut PayloadStore, mem: &mut usize, target_amount: usize) -> Result<usize, JsValue> {
        let mut freed = 0;
        
        // Simple strategy: Linear scan and evict first found InMemory.
        let mut candidates = Vec::new();
        for (&id, val) in payloads.iter() {
            if let Payload::InMemory(_) = val {
                candidates.push(id);
                if candidates.len() >= 50 { break; } // limit batch
            }
        }
        
        for id in candidates {
            if freed >= target_amount { break; }
            
            if let Some(Payload::InMemory(val)) = payloads.remove(&id) {
                // Serialize
                let bytes = serde_json::to_vec(&val).map_err(|e| JsValue::from_str(&e.to_string()))?;
                let size = bytes.len();
                
                // Write to store
                let (offset, len) = store.save(&bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
                
                // Insert OnDisk
                payloads.insert(id, Payload::OnDisk { offset, len });
                
                // Update Memory
                freed += size;
                if *mem >= size {
                    *mem -= size;
                }
            }
        }
        
        Ok(freed)
    }

    pub fn snapshot(&self) -> Result<Vec<u8>, JsValue> {
        // Acquire all read locks upfront for snapshot consistency
        let index = self.index.read();
        let sparse_index = self.sparse_index.read();
        let payloads = self.payloads.read();
        let active_ids = self.active_ids.read();
        let vectors = self.vectors.read();
        
        let snap = CollectionSnapshot {
            name: self.name.clone(),
            config: self.config.clone(),
            index: &*index,
            sparse_index: &*sparse_index,
            payloads: &*payloads,
            active_ids: &*active_ids,
            vectors: &*vectors,
        };
        let payload = bincode::serialize(&snap)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        // Wrap with versioned header
        Ok(wrap_snapshot(&payload))
    }

    pub fn load_snapshot(data: &[u8], wal_handle: JsValue) -> Result<WinnowCollection, JsValue> {
        // Unwrap versioned snapshot with automatic migration
        let (_header, payload) = unwrap_snapshot(data)
            .map_err(|e: SnapshotError| JsValue::from(e))?;
        
        let snap: CollectionSnapshotOwned = bincode::deserialize(&payload)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let mut wal = None;
        if !wal_handle.is_undefined() && !wal_handle.is_null() {
             let handle = FileSystemSyncAccessHandle::unchecked_from_js(wal_handle);
             let op_wal = OpWal::new(handle).map_err(|e| JsValue::from_str(&e.to_string()))?;
             // Note: If loading from snapshot, we assume WAL matches or is strictly for new appends?
             // Usually loading snapshot implies ignoring previous WAL history up to that point.
             // We just attach the WAL for future writes.
             wal = Some(op_wal);
        }
        
        // Re-calculate memory usage
        let mut memory_usage = 0;
        let vec_size = snap.active_ids.len() * snap.config.dim * 4;
        memory_usage += vec_size;
        
        // Estimate payload usage (Only count InMemory)
        for val in snap.payloads.values() {
             if let Payload::InMemory(v) = val {
                 if let Ok(s) = serde_json::to_string(v) {
                     memory_usage += s.len();
                 }
             }
        }

        Ok(WinnowCollection {
            name: snap.name,
            config: snap.config,
            index: Arc::new(RwLock::new(snap.index)),
            sparse_index: Arc::new(RwLock::new(snap.sparse_index)),
            payloads: Arc::new(RwLock::new(snap.payloads)),
            active_ids: Arc::new(RwLock::new(snap.active_ids)),
            wal: Arc::new(RwLock::new(wal)),
            payload_store: Arc::new(RwLock::new(None)),
            memory_usage: Arc::new(RwLock::new(memory_usage)),
            active_writes: Arc::new(AtomicUsize::new(0)),
            metrics: Arc::new(Metrics::new()),
            breaker: Arc::new(CircuitBreaker::new(CircuitBreakerConfig::default())),
            rate_limiter: Arc::new(RateLimiter::new(100000.0, 10000.0)),
            vectors: Arc::new(RwLock::new(snap.vectors)),
        })
    }

    pub fn get_metrics(&self) -> String {
        self.metrics.scrape()
    }

    pub fn health_check(&self) -> Result<JsValue, JsValue> {
        let mem = *self.memory_usage.read();
        let writes = self.active_writes.load(Ordering::Relaxed);
        let open = self.breaker.is_open();
        let status = if open { "Unhealthy" } else { "Healthy" };
        
        let json = format!(r#"{{"status": "{}", "memory_usage": {}, "active_writes": {}, "circuit_open": {}}}"#, 
            status, mem, writes, open);
        
        js_sys::JSON::parse(&json)
    }

    pub async fn search_with_timeout(&self, query: Vec<f32>, k: usize, filter: Option<JsValue>, timeout_ms: u32) -> Result<JsValue, JsValue> {
        let this = self.clone();
        
        let controller = web_sys::AbortController::new().map_err(|e| JsValue::from_str(&e.as_string().unwrap_or_default()))?;
        let signal = controller.signal();
        
        // Wrap search in a promise. 
        // Note: In a browser, if this blocks the main thread, the timeout won't fire until it's done.
        // But AbortSignal checks in index loops help if execution yields.
        let search_promise = wasm_bindgen_futures::future_to_promise(async move {
            this.search(query, k, filter, Some(signal))
        });

        let timeout_promise = Promise::new(&mut |_, reject| {
             if let Some(window) = web_sys::window() {
                 let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&reject, timeout_ms as i32);
             }
        });
        
        // Race
        let racing = js_sys::Promise::race(&js_sys::Array::of2(
            &search_promise, &timeout_promise
        ));
        
        match JsFuture::from(racing).await {
            Ok(val) => Ok(val),
            Err(e) => {
                controller.abort();
                Err(e)
            }
        }
    }

    fn create_index_internal(config: &CollectionConfig) -> Result<VectorIndexEnum, JsValue> {
        match config.index_type {
            IndexType::HNSW => {
                let m = config.m.unwrap_or(16);
                let ef_c = config.ef_construction.unwrap_or(200);
                let ef_s = config.ef_search.unwrap_or(10);
                let metric = config.metric.unwrap_or(MetricType::L2);
                Ok(VectorIndexEnum::Hnsw(IndexHNSW::new(config.dim, m, ef_c, ef_s, metric)))
            },
            IndexType::IVFPQ => {
                let n_centroids = config.n_centroids.unwrap_or(256);
                let n_sub = config.n_sub.unwrap_or(8);
                let metric = config.metric.unwrap_or(MetricType::L2);
                Ok(VectorIndexEnum::IvfPq(IndexIVFPQ::new(config.dim, n_centroids, n_sub, metric)?))
            },
            IndexType::SQ => {
                let metric = config.metric.unwrap_or(MetricType::L2);
                Ok(VectorIndexEnum::Sq(IndexSQ::new(config.dim, metric)))
            },
            IndexType::BQ => {
                Ok(VectorIndexEnum::Bq(IndexBQ::new(config.dim)))
            },
        }
    }

    pub fn compact(&self) -> Result<(), JsValue> {
        // Stop the world for writes during compaction
        let active = self.active_ids.read();
        let vectors = self.vectors.read();
        let config = &self.config;

        let mut new_index = Self::create_index_internal(config)?;

        // 1. Prepare Sampled data for Training (Avoid OOM)
        // Cap training data at 10,000 vectors or 10% of total
        let training_limit = 10_000.min(active.len());
        let mut training_vecs = Vec::with_capacity(training_limit * config.dim);
        
        let mut count = 0;
        for &id in active.iter() {
            if count >= training_limit { break; }
            if let Some(v) = vectors.get(id) {
                training_vecs.extend_from_slice(v);
                count += 1;
            }
        }

        // 2. Train if necessary (IVFPQ, SQ)
        if !training_vecs.is_empty() {
            match &new_index {
                VectorIndexEnum::IvfPq(_) | VectorIndexEnum::Sq(_) => {
                    new_index.train(&training_vecs)?;
                },
                _ => {}
            }
        }
        
        // Free memory early
        drop(training_vecs);

        // 3. Re-Populate (Batched/Linear Iteration)
        for &id in active.iter() {
            if let Some(v) = vectors.get(id) {
                new_index.add(id, v.clone())?;
            }
        }

        // 4. Swap
        let mut index_lock = self.index.write();
        *index_lock = new_index;

        // 5. Truncate WAL
        if let Some(wal) = self.wal.write().as_mut() {
            wal.reset().map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[derive(Serialize)]

struct CollectionSnapshot<'a> {
    name: String,
    config: CollectionConfig,
    index: &'a VectorIndexEnum,
    sparse_index: &'a Option<IndexSparse>,
    payloads: &'a HashMap<u32, Payload>,
    active_ids: &'a HashSet<u32>,
    vectors: &'a VectorStorage,
}

#[derive(Deserialize)]
struct CollectionSnapshotOwned {
    name: String,
    config: CollectionConfig,
    index: VectorIndexEnum,
    sparse_index: Option<IndexSparse>,
    payloads: HashMap<u32, Payload>,
    active_ids: HashSet<u32>,
    vectors: VectorStorage,
}
