//! Audit logging for WinnowDB
//!
//! Provides structured event logging for security and compliance.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::Mutex;
use wasm_bindgen::prelude::*;

/// Types of auditable events
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AuditEventType {
    // Auth events
    AuthSuccess,
    AuthFailure,
    KeyCreated,
    KeyRevoked,
    
    // Data events
    VectorInsert,
    VectorDelete,
    VectorUpdate,
    CollectionCreated,
    CollectionDropped,
    CollectionCleared,
    
    // Query events
    Search,
    SearchSlow,
    
    // Admin events
    SnapshotCreated,
    SnapshotLoaded,
    ConfigChanged,
    
    // Error events
    RateLimited,
    CircuitBreakerTripped,
    PermissionDenied,
}

/// A structured audit log entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Timestamp (ms since epoch)
    pub timestamp: f64,
    /// Event type
    pub event_type: AuditEventType,
    /// User ID (if authenticated)
    pub user_id: Option<String>,
    /// Collection name (if applicable)
    pub collection: Option<String>,
    /// Additional details
    pub details: String,
    /// Whether operation succeeded
    pub success: bool,
    /// Latency in ms (for operations)
    pub latency_ms: Option<f64>,
}

impl AuditEntry {
    pub fn new(event_type: AuditEventType, success: bool, details: &str) -> Self {
        Self {
            timestamp: js_sys::Date::now(),
            event_type,
            user_id: None,
            collection: None,
            details: details.to_string(),
            success,
            latency_ms: None,
        }
    }
    
    pub fn with_user(mut self, user_id: &str) -> Self {
        self.user_id = Some(user_id.to_string());
        self
    }
    
    pub fn with_collection(mut self, collection: &str) -> Self {
        self.collection = Some(collection.to_string());
        self
    }
    
    pub fn with_latency(mut self, latency_ms: f64) -> Self {
        self.latency_ms = Some(latency_ms);
        self
    }
}

/// Audit log with configurable retention
#[derive(Clone)]
pub struct AuditLog {
    entries: Arc<Mutex<Vec<AuditEntry>>>,
    max_entries: usize,
    enabled: bool,
}

impl AuditLog {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::with_capacity(max_entries))),
            max_entries,
            enabled: true,
        }
    }
    
    /// Enable or disable logging
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Log an audit event
    pub fn log(&self, entry: AuditEntry) {
        if !self.enabled { return; }
        
        let mut entries = self.entries.lock();
        if entries.len() >= self.max_entries {
            entries.remove(0); // Ring buffer
        }
        entries.push(entry);
    }
    
    /// Get recent entries (most recent first)
    pub fn recent(&self, count: usize) -> Vec<AuditEntry> {
        let entries = self.entries.lock();
        let skip = entries.len().saturating_sub(count);
        entries.iter().skip(skip).rev().cloned().collect()
    }
    
    /// Get entries by type
    pub fn by_type(&self, event_type: &AuditEventType, count: usize) -> Vec<AuditEntry> {
        let entries = self.entries.lock();
        entries.iter()
            .filter(|e| std::mem::discriminant(&e.event_type) == std::mem::discriminant(event_type))
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
    
    /// Get entries for a specific user
    pub fn by_user(&self, user_id: &str, count: usize) -> Vec<AuditEntry> {
        let entries = self.entries.lock();
        entries.iter()
            .filter(|e| e.user_id.as_deref() == Some(user_id))
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
    
    /// Get entries for a specific collection
    pub fn by_collection(&self, collection: &str, count: usize) -> Vec<AuditEntry> {
        let entries = self.entries.lock();
        entries.iter()
            .filter(|e| e.collection.as_deref() == Some(collection))
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
    
    /// Get failed operations
    pub fn failures(&self, count: usize) -> Vec<AuditEntry> {
        let entries = self.entries.lock();
        entries.iter()
            .filter(|e| !e.success)
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
    
    /// Clear all entries
    pub fn clear(&self) {
        self.entries.lock().clear();
    }
    
    /// Export to JSON
    pub fn export_json(&self) -> String {
        let entries = self.entries.lock();
        serde_json::to_string(&*entries).unwrap_or_else(|_| "[]".to_string())
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new(1000) // Default 1000 entry retention
    }
}

/// WASM-exposed audit log
#[wasm_bindgen]
pub struct AuditLogger {
    log: AuditLog,
}

#[wasm_bindgen]
impl AuditLogger {
    #[wasm_bindgen(constructor)]
    pub fn new(max_entries: usize) -> Self {
        Self {
            log: AuditLog::new(max_entries),
        }
    }
    
    pub fn log_auth_success(&self, user_id: &str) {
        self.log.log(
            AuditEntry::new(AuditEventType::AuthSuccess, true, "Authentication successful")
                .with_user(user_id)
        );
    }
    
    pub fn log_auth_failure(&self, reason: &str) {
        self.log.log(
            AuditEntry::new(AuditEventType::AuthFailure, false, reason)
        );
    }
    
    pub fn log_search(&self, collection: &str, user_id: Option<String>, latency_ms: f64, k: usize) {
        let mut entry = AuditEntry::new(
            AuditEventType::Search, 
            true, 
            &format!("Search k={}", k)
        ).with_collection(collection).with_latency(latency_ms);
        
        if let Some(uid) = user_id {
            entry = entry.with_user(&uid);
        }
        self.log.log(entry);
    }
    
    pub fn log_insert(&self, collection: &str, user_id: Option<String>, id: u32) {
        let mut entry = AuditEntry::new(
            AuditEventType::VectorInsert,
            true,
            &format!("Vector {} inserted", id)
        ).with_collection(collection);
        
        if let Some(uid) = user_id {
            entry = entry.with_user(&uid);
        }
        self.log.log(entry);
    }
    
    pub fn log_delete(&self, collection: &str, user_id: Option<String>, id: u32) {
        let mut entry = AuditEntry::new(
            AuditEventType::VectorDelete,
            true,
            &format!("Vector {} deleted", id)
        ).with_collection(collection);
        
        if let Some(uid) = user_id {
            entry = entry.with_user(&uid);
        }
        self.log.log(entry);
    }
    
    pub fn recent(&self, count: usize) -> JsValue {
        let entries = self.log.recent(count);
        serde_wasm_bindgen::to_value(&entries).unwrap_or(JsValue::NULL)
    }
    
    pub fn failures(&self, count: usize) -> JsValue {
        let entries = self.log.failures(count);
        serde_wasm_bindgen::to_value(&entries).unwrap_or(JsValue::NULL)
    }
    
    pub fn export_json(&self) -> String {
        self.log.export_json()
    }
    
    pub fn clear(&self) {
        self.log.clear();
    }
}
