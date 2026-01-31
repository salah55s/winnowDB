//! Schema versioning and migration for WinnowDB snapshots
//!
//! Provides versioned snapshots and automatic migration on load.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use crc32fast::Hasher;

/// Current snapshot schema version
pub const SNAPSHOT_VERSION: u32 = 2;

/// Magic bytes to identify WinnowDB snapshots
pub const SNAPSHOT_MAGIC: &[u8; 8] = b"WINNOWDB";

/// Snapshot header with version and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotHeader {
    /// Magic bytes for identification
    pub magic: [u8; 8],
    /// Schema version
    pub version: u32,
    /// Creation timestamp (ms since epoch)
    pub created_at: u64,
    /// CRC32 checksum of payload
    pub checksum: u32,
    /// Size of payload in bytes
    pub payload_size: u64,
}

impl SnapshotHeader {
    /// Create a new header for the given payload
    pub fn new(payload: &[u8]) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(payload);
        let checksum = hasher.finalize();
        
        Self {
            magic: *SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
            created_at: js_sys::Date::now() as u64,
            checksum,
            payload_size: payload.len() as u64,
        }
    }
    
    /// Serialize header to bytes (fixed 32 bytes)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        bytes.extend_from_slice(&self.magic);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&self.created_at.to_le_bytes());
        bytes.extend_from_slice(&self.checksum.to_le_bytes());
        bytes.extend_from_slice(&self.payload_size.to_le_bytes());
        bytes
    }
    
    /// Parse header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SnapshotError> {
        if bytes.len() < 32 {
            return Err(SnapshotError::HeaderTooSmall);
        }
        
        let magic: [u8; 8] = bytes[0..8].try_into().unwrap();
        if &magic != SNAPSHOT_MAGIC {
            return Err(SnapshotError::InvalidMagic);
        }
        
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let created_at = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        let checksum = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let payload_size = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        
        Ok(Self {
            magic,
            version,
            created_at,
            checksum,
            payload_size,
        })
    }
    
    /// Verify checksum matches payload
    pub fn verify(&self, payload: &[u8]) -> Result<(), SnapshotError> {
        if payload.len() as u64 != self.payload_size {
            return Err(SnapshotError::SizeMismatch {
                expected: self.payload_size,
                got: payload.len() as u64,
            });
        }
        
        let mut hasher = Hasher::new();
        hasher.update(payload);
        let computed = hasher.finalize();
        
        if computed != self.checksum {
            return Err(SnapshotError::ChecksumMismatch {
                expected: self.checksum,
                computed,
            });
        }
        
        Ok(())
    }
}

/// Snapshot-related errors
#[derive(Debug, Clone)]
pub enum SnapshotError {
    HeaderTooSmall,
    InvalidMagic,
    SizeMismatch { expected: u64, got: u64 },
    ChecksumMismatch { expected: u32, computed: u32 },
    UnsupportedVersion { version: u32, max_supported: u32 },
    MigrationFailed { from: u32, to: u32, reason: String },
    DeserializationFailed(String),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HeaderTooSmall => write!(f, "Snapshot header too small"),
            Self::InvalidMagic => write!(f, "Invalid snapshot magic bytes"),
            Self::SizeMismatch { expected, got } => 
                write!(f, "Payload size mismatch: expected {}, got {}", expected, got),
            Self::ChecksumMismatch { expected, computed } => 
                write!(f, "Checksum mismatch: expected {}, computed {}", expected, computed),
            Self::UnsupportedVersion { version, max_supported } => 
                write!(f, "Unsupported snapshot version: {}, max supported: {}", version, max_supported),
            Self::MigrationFailed { from, to, reason } => 
                write!(f, "Migration from v{} to v{} failed: {}", from, to, reason),
            Self::DeserializationFailed(msg) => 
                write!(f, "Deserialization failed: {}", msg),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl From<SnapshotError> for JsValue {
    fn from(err: SnapshotError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}

/// Migration registry for schema upgrades
pub struct MigrationRegistry {
    migrations: Vec<Migration>,
}

struct Migration {
    from_version: u32,
    to_version: u32,
    migrate: fn(&[u8]) -> Result<Vec<u8>, String>,
}

impl MigrationRegistry {
    pub fn new() -> Self {
        let mut registry = Self { migrations: Vec::new() };
        
        // Register V1 -> V2 migration
        registry.register(1, 2, migrate_v1_to_v2);
        
        registry
    }
    
    fn register(&mut self, from: u32, to: u32, migrate: fn(&[u8]) -> Result<Vec<u8>, String>) {
        self.migrations.push(Migration {
            from_version: from,
            to_version: to,
            migrate,
        });
    }
    
    /// Apply all necessary migrations to upgrade from source version to target
    pub fn migrate(&self, data: &[u8], from: u32, to: u32) -> Result<Vec<u8>, SnapshotError> {
        if from >= to {
            return Ok(data.to_vec());
        }
        
        let mut current_version = from;
        let mut current_data = data.to_vec();
        
        while current_version < to {
            let migration = self.migrations.iter()
                .find(|m| m.from_version == current_version)
                .ok_or_else(|| SnapshotError::UnsupportedVersion {
                    version: current_version,
                    max_supported: SNAPSHOT_VERSION,
                })?;
            
            current_data = (migration.migrate)(&current_data)
                .map_err(|reason| SnapshotError::MigrationFailed {
                    from: current_version,
                    to: migration.to_version,
                    reason,
                })?;
            
            current_version = migration.to_version;
        }
        
        Ok(current_data)
    }
    
    /// Check if migration is possible
    pub fn can_migrate(&self, from: u32, to: u32) -> bool {
        if from >= to {
            return true;
        }
        
        let mut current = from;
        while current < to {
            if let Some(m) = self.migrations.iter().find(|m| m.from_version == current) {
                current = m.to_version;
            } else {
                return false;
            }
        }
        true
    }
}

impl Default for MigrationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Migration V1 -> V2: Add sparse index field (optional)
fn migrate_v1_to_v2(data: &[u8]) -> Result<Vec<u8>, String> {
    // V1 format: bincode-serialized without sparse_index Option wrapper
    // V2 format: bincode-serialized with sparse_index as Option<IndexSparse>
    
    // In practice, we need to deserialize V1, transform, re-serialize as V2
    // For now, V1 snapshots without versioned header are treated as legacy
    // The transformation depends on actual struct layout changes
    
    // Simple pass-through for now (actual migration logic would go here)
    // This handles the case where V1 and V2 formats are compatible
    Ok(data.to_vec())
}

/// Wrap raw snapshot payload with versioned header
pub fn wrap_snapshot(payload: &[u8]) -> Vec<u8> {
    let header = SnapshotHeader::new(payload);
    let mut result = Vec::with_capacity(32 + payload.len());
    result.extend_from_slice(&header.to_bytes());
    result.extend_from_slice(payload);
    result
}

/// Unwrap versioned snapshot, applying migrations if needed
pub fn unwrap_snapshot(data: &[u8]) -> Result<(SnapshotHeader, Vec<u8>), SnapshotError> {
    // Check if this is a versioned snapshot (has magic header)
    if data.len() >= 8 && &data[0..8] == SNAPSHOT_MAGIC {
        let header = SnapshotHeader::from_bytes(data)?;
        let payload = &data[32..];
        
        header.verify(payload)?;
        
        // Check version and migrate if needed
        if header.version > SNAPSHOT_VERSION {
            return Err(SnapshotError::UnsupportedVersion {
                version: header.version,
                max_supported: SNAPSHOT_VERSION,
            });
        }
        
        if header.version < SNAPSHOT_VERSION {
            let registry = MigrationRegistry::new();
            let migrated = registry.migrate(payload, header.version, SNAPSHOT_VERSION)?;
            Ok((header, migrated))
        } else {
            Ok((header, payload.to_vec()))
        }
    } else {
        // Legacy snapshot without header (V1)
        // Wrap as if it were V1 and migrate
        let header = SnapshotHeader {
            magic: *SNAPSHOT_MAGIC,
            version: 1,
            created_at: 0, // Unknown
            checksum: 0,   // Not verified for legacy
            payload_size: data.len() as u64,
        };
        
        let registry = MigrationRegistry::new();
        let migrated = registry.migrate(data, 1, SNAPSHOT_VERSION)?;
        Ok((header, migrated))
    }
}

/// Rollback result for migration failures
#[derive(Debug, Clone)]
pub enum RollbackResult {
    /// Migration succeeded
    Success { header: SnapshotHeader, payload: Vec<u8> },
    /// Migration failed, returning original data for manual recovery
    Rollback { original: Vec<u8>, error: SnapshotError },
}

/// Unwrap with automatic rollback on failure
/// Returns original data if migration fails, allowing manual recovery
pub fn unwrap_snapshot_with_rollback(data: &[u8]) -> RollbackResult {
    match unwrap_snapshot(data) {
        Ok((header, payload)) => RollbackResult::Success { header, payload },
        Err(error) => RollbackResult::Rollback { 
            original: data.to_vec(), 
            error 
        },
    }
}

/// Validate snapshot before loading (dry-run)
pub fn validate_snapshot(data: &[u8]) -> Result<SnapshotHeader, SnapshotError> {
    if data.len() < 8 {
        return Err(SnapshotError::HeaderTooSmall);
    }
    
    if &data[0..8] == SNAPSHOT_MAGIC {
        let header = SnapshotHeader::from_bytes(data)?;
        let payload = &data[32..];
        header.verify(payload)?;
        
        // Check if migration is possible
        if header.version < SNAPSHOT_VERSION {
            let registry = MigrationRegistry::new();
            if !registry.can_migrate(header.version, SNAPSHOT_VERSION) {
                return Err(SnapshotError::UnsupportedVersion {
                    version: header.version,
                    max_supported: SNAPSHOT_VERSION,
                });
            }
        }
        
        Ok(header)
    } else {
        // Legacy format - assume V1
        Ok(SnapshotHeader {
            magic: *SNAPSHOT_MAGIC,
            version: 1,
            created_at: 0,
            checksum: 0,
            payload_size: data.len() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_header_roundtrip() {
        let payload = b"test payload data";
        let header = SnapshotHeader::new(payload);
        let bytes = header.to_bytes();
        let parsed = SnapshotHeader::from_bytes(&bytes).unwrap();
        
        assert_eq!(parsed.version, SNAPSHOT_VERSION);
        assert_eq!(parsed.payload_size, payload.len() as u64);
        assert!(parsed.verify(payload).is_ok());
    }
    
    #[test]
    fn test_wrap_unwrap() {
        let payload = b"collection snapshot data";
        let wrapped = wrap_snapshot(payload);
        let (header, unwrapped) = unwrap_snapshot(&wrapped).unwrap();
        
        assert_eq!(header.version, SNAPSHOT_VERSION);
        assert_eq!(unwrapped, payload);
    }
    
    #[test]
    fn test_checksum_mismatch() {
        let payload = b"original data";
        let mut wrapped = wrap_snapshot(payload);
        // Corrupt payload
        wrapped[33] ^= 0xFF;
        
        let result = unwrap_snapshot(&wrapped);
        assert!(matches!(result, Err(SnapshotError::ChecksumMismatch { .. })));
    }
}
