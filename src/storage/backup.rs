//! Backup automation for WinnowDB
//!
//! Provides scheduled backups with multiple storage backends.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::Mutex;
use wasm_bindgen::prelude::*;

/// Backup storage backend types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BackupBackend {
    /// Local filesystem (OPFS)
    Local { path: String },
    /// Amazon S3
    S3 { bucket: String, prefix: String, region: String },
    /// Google Cloud Storage
    GCS { bucket: String, prefix: String },
    /// Custom HTTP endpoint
    Http { url: String },
}

/// Backup configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Backup interval in milliseconds (0 = disabled)
    pub interval_ms: u64,
    /// Maximum backups to retain (0 = unlimited)
    pub retention_count: usize,
    /// Storage backend
    pub backend: BackupBackend,
    /// Include payload data
    pub include_payloads: bool,
    /// Compress backup
    pub compress: bool,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            interval_ms: 3600_000, // 1 hour
            retention_count: 24,   // Keep 24 backups
            backend: BackupBackend::Local { path: "backups".to_string() },
            include_payloads: true,
            compress: true,
        }
    }
}

/// A completed backup record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackupRecord {
    /// Unique backup ID
    pub id: String,
    /// Timestamp (ms since epoch)
    pub timestamp: f64,
    /// Size in bytes
    pub size_bytes: u64,
    /// Number of vectors
    pub vector_count: u64,
    /// Backend used
    pub backend: String,
    /// Whether backup succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Backup scheduler and manager
#[derive(Clone)]
pub struct BackupManager {
    config: Arc<Mutex<BackupConfig>>,
    history: Arc<Mutex<Vec<BackupRecord>>>,
    last_backup: Arc<Mutex<f64>>,
}

impl BackupManager {
    pub fn new(config: BackupConfig) -> Self {
        Self {
            config: Arc::new(Mutex::new(config)),
            history: Arc::new(Mutex::new(Vec::new())),
            last_backup: Arc::new(Mutex::new(0.0)),
        }
    }
    
    /// Check if backup is due
    pub fn is_backup_due(&self) -> bool {
        let config = self.config.lock();
        if config.interval_ms == 0 {
            return false;
        }
        
        let last = *self.last_backup.lock();
        let now = js_sys::Date::now();
        (now - last) >= config.interval_ms as f64
    }
    
    /// Record a backup completion
    pub fn record_backup(&self, id: String, size_bytes: u64, vector_count: u64, success: bool, error: Option<String>) {
        let backend = {
            let config = self.config.lock();
            match &config.backend {
                BackupBackend::Local { .. } => "local",
                BackupBackend::S3 { .. } => "s3",
                BackupBackend::GCS { .. } => "gcs",
                BackupBackend::Http { .. } => "http",
            }.to_string()
        };
        
        let record = BackupRecord {
            id,
            timestamp: js_sys::Date::now(),
            size_bytes,
            vector_count,
            backend,
            success,
            error,
        };
        
        let mut history = self.history.lock();
        history.push(record);
        
        // Enforce retention
        let retention = self.config.lock().retention_count;
        if retention > 0 && history.len() > retention {
            let to_remove = history.len() - retention;
            history.drain(0..to_remove);
        }
        
        if success {
            *self.last_backup.lock() = js_sys::Date::now();
        }
    }
    
    /// Get backup history
    pub fn history(&self) -> Vec<BackupRecord> {
        self.history.lock().clone()
    }
    
    /// Get last successful backup
    pub fn last_successful(&self) -> Option<BackupRecord> {
        self.history.lock().iter().rev().find(|r| r.success).cloned()
    }
    
    /// Update config
    pub fn update_config(&self, config: BackupConfig) {
        *self.config.lock() = config;
    }
    
    /// Generate backup ID
    pub fn generate_id(&self) -> String {
        let now = js_sys::Date::now() as u64;
        format!("backup_{}", now)
    }
}

impl Default for BackupManager {
    fn default() -> Self {
        Self::new(BackupConfig::default())
    }
}

/// WASM-exposed backup manager
#[wasm_bindgen]
pub struct BackupScheduler {
    manager: BackupManager,
}

#[wasm_bindgen]
impl BackupScheduler {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            manager: BackupManager::default(),
        }
    }
    
    /// Configure with JSON
    pub fn configure(&self, config: JsValue) -> Result<(), JsValue> {
        let cfg: BackupConfig = serde_wasm_bindgen::from_value(config)?;
        self.manager.update_config(cfg);
        Ok(())
    }
    
    /// Check if backup is due
    pub fn is_due(&self) -> bool {
        self.manager.is_backup_due()
    }
    
    /// Record backup result
    pub fn record(&self, size_bytes: u64, vector_count: u64, success: bool, error: Option<String>) -> String {
        let id = self.manager.generate_id();
        self.manager.record_backup(id.clone(), size_bytes, vector_count, success, error);
        id
    }
    
    /// Get backup history
    pub fn history(&self) -> JsValue {
        let history = self.manager.history();
        serde_wasm_bindgen::to_value(&history).unwrap_or(JsValue::NULL)
    }
    
    /// Get last successful backup
    pub fn last_successful(&self) -> JsValue {
        match self.manager.last_successful() {
            Some(record) => serde_wasm_bindgen::to_value(&record).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }
}

impl Default for BackupScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Incremental Backup Support
// ============================================================================

/// Checkpoint for incremental backups
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackupCheckpoint {
    /// Checkpoint ID
    pub id: String,
    /// Base backup ID this checkpoint is based on
    pub base_backup_id: String,
    /// Timestamp
    pub timestamp: f64,
    /// Operation sequence number at checkpoint
    pub sequence_number: u64,
    /// Number of changes since base
    pub change_count: u64,
    /// Size of delta in bytes
    pub delta_size: u64,
}

/// Delta entry for incremental backup
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackupDelta {
    /// Operation type
    pub op_type: DeltaOpType,
    /// Vector ID affected
    pub vector_id: u32,
    /// Sequence number
    pub seq: u64,
    /// Timestamp
    pub timestamp: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DeltaOpType {
    Insert,
    Update,
    Delete,
}

/// Incremental backup manager
#[derive(Clone)]
pub struct IncrementalBackup {
    checkpoints: Arc<Mutex<Vec<BackupCheckpoint>>>,
    pending_deltas: Arc<Mutex<Vec<BackupDelta>>>,
    current_seq: Arc<Mutex<u64>>,
    max_delta_size: usize,
}

impl IncrementalBackup {
    pub fn new(max_delta_size: usize) -> Self {
        Self {
            checkpoints: Arc::new(Mutex::new(Vec::new())),
            pending_deltas: Arc::new(Mutex::new(Vec::new())),
            current_seq: Arc::new(Mutex::new(0)),
            max_delta_size,
        }
    }
    
    /// Record a delta operation
    pub fn record_delta(&self, op_type: DeltaOpType, vector_id: u32) {
        let mut seq = self.current_seq.lock();
        *seq += 1;
        let delta = BackupDelta {
            op_type,
            vector_id,
            seq: *seq,
            timestamp: js_sys::Date::now(),
        };
        self.pending_deltas.lock().push(delta);
    }
    
    /// Check if delta threshold exceeded (need new checkpoint)
    pub fn needs_checkpoint(&self) -> bool {
        self.pending_deltas.lock().len() >= self.max_delta_size
    }
    
    /// Create a checkpoint
    pub fn create_checkpoint(&self, base_backup_id: &str) -> BackupCheckpoint {
        let mut deltas = self.pending_deltas.lock();
        let seq = *self.current_seq.lock();
        
        let checkpoint = BackupCheckpoint {
            id: format!("checkpoint_{}", js_sys::Date::now() as u64),
            base_backup_id: base_backup_id.to_string(),
            timestamp: js_sys::Date::now(),
            sequence_number: seq,
            change_count: deltas.len() as u64,
            delta_size: deltas.len() as u64 * 16, // Approximate
        };
        
        // Clear pending deltas
        deltas.clear();
        
        // Store checkpoint
        self.checkpoints.lock().push(checkpoint.clone());
        
        checkpoint
    }
    
    /// Get pending deltas for incremental backup
    pub fn get_pending_deltas(&self) -> Vec<BackupDelta> {
        self.pending_deltas.lock().clone()
    }
    
    /// Get all checkpoints since a base backup
    pub fn checkpoints_since(&self, base_backup_id: &str) -> Vec<BackupCheckpoint> {
        self.checkpoints.lock()
            .iter()
            .filter(|c| c.base_backup_id == base_backup_id)
            .cloned()
            .collect()
    }
    
    /// Get latest checkpoint
    pub fn latest_checkpoint(&self) -> Option<BackupCheckpoint> {
        self.checkpoints.lock().last().cloned()
    }
}

impl Default for IncrementalBackup {
    fn default() -> Self {
        Self::new(1000) // Default 1000 operations before checkpoint
    }
}

// ============================================================================
// Point-in-Time Recovery
// ============================================================================

/// Recovery point for PITR
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecoveryPoint {
    /// Recovery point ID
    pub id: String,
    /// Timestamp this point represents
    pub target_time: f64,
    /// Base backup to start from
    pub base_backup_id: String,
    /// Checkpoints to apply
    pub checkpoints: Vec<String>,
    /// Final sequence number
    pub final_seq: u64,
}

/// Point-in-time recovery manager
#[derive(Clone)]
pub struct PITRManager {
    recovery_points: Arc<Mutex<Vec<RecoveryPoint>>>,
    incremental: IncrementalBackup,
}

impl PITRManager {
    pub fn new() -> Self {
        Self {
            recovery_points: Arc::new(Mutex::new(Vec::new())),
            incremental: IncrementalBackup::default(),
        }
    }
    
    /// Find recovery point closest to target time
    pub fn find_recovery_point(&self, target_time: f64, backups: &[BackupRecord]) -> Option<RecoveryPoint> {
        // Find the latest backup before target time
        let base = backups.iter()
            .filter(|b| b.success && b.timestamp <= target_time)
            .max_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap())?;
        
        // Get checkpoints between backup and target time
        let checkpoints: Vec<String> = self.incremental.checkpoints.lock()
            .iter()
            .filter(|c| c.base_backup_id == base.id && c.timestamp <= target_time)
            .map(|c| c.id.clone())
            .collect();
        
        let final_seq = checkpoints.iter()
            .filter_map(|id| {
                self.incremental.checkpoints.lock()
                    .iter()
                    .find(|c| &c.id == id)
                    .map(|c| c.sequence_number)
            })
            .max()
            .unwrap_or(0);
        
        Some(RecoveryPoint {
            id: format!("recovery_{}", js_sys::Date::now() as u64),
            target_time,
            base_backup_id: base.id.clone(),
            checkpoints,
            final_seq,
        })
    }
    
    /// Record a delta (delegates to incremental)
    pub fn record_delta(&self, op_type: DeltaOpType, vector_id: u32) {
        self.incremental.record_delta(op_type, vector_id);
    }
    
    /// Create checkpoint if needed
    pub fn maybe_checkpoint(&self, base_backup_id: &str) -> Option<BackupCheckpoint> {
        if self.incremental.needs_checkpoint() {
            Some(self.incremental.create_checkpoint(base_backup_id))
        } else {
            None
        }
    }
    
    /// List available recovery windows
    pub fn recovery_windows(&self, backups: &[BackupRecord]) -> Vec<(f64, f64)> {
        let mut windows = Vec::new();
        
        for backup in backups.iter().filter(|b| b.success) {
            let start = backup.timestamp;
            let end = self.incremental.checkpoints.lock()
                .iter()
                .filter(|c| c.base_backup_id == backup.id)
                .map(|c| c.timestamp)
                .fold(start, f64::max);
            windows.push((start, end));
        }
        
        windows
    }
}

impl Default for PITRManager {
    fn default() -> Self {
        Self::new()
    }
}

/// WASM-exposed PITR manager
#[wasm_bindgen]
pub struct RecoveryManager {
    pitr: PITRManager,
    backup: BackupManager,
}

#[wasm_bindgen]
impl RecoveryManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            pitr: PITRManager::default(),
            backup: BackupManager::default(),
        }
    }
    
    /// Record an insert operation
    pub fn record_insert(&self, vector_id: u32) {
        self.pitr.record_delta(DeltaOpType::Insert, vector_id);
    }
    
    /// Record a delete operation
    pub fn record_delete(&self, vector_id: u32) {
        self.pitr.record_delta(DeltaOpType::Delete, vector_id);
    }
    
    /// Find recovery point for target timestamp
    pub fn find_recovery_point(&self, target_time: f64) -> JsValue {
        let backups = self.backup.history();
        match self.pitr.find_recovery_point(target_time, &backups) {
            Some(point) => serde_wasm_bindgen::to_value(&point).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }
    
    /// Get recovery windows
    pub fn recovery_windows(&self) -> JsValue {
        let backups = self.backup.history();
        let windows = self.pitr.recovery_windows(&backups);
        serde_wasm_bindgen::to_value(&windows).unwrap_or(JsValue::NULL)
    }
}

impl Default for RecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}
