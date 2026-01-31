//! Structured error types for WinnowDB
//! 
//! Provides rich error information with recovery suggestions.

use serde::{Deserialize, Serialize};
use std::fmt;
use wasm_bindgen::prelude::*;

/// Enumeration of all WinnowDB error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WinnowErrorKind {
    /// Vector dimension doesn't match collection configuration
    DimensionMismatch { expected: usize, got: usize },
    
    /// Index requires training before use (IVF-PQ, SQ)
    IndexNotTrained { index_type: String },
    
    /// Rate limit exceeded (Token Bucket depleted)
    RateLimitExceeded { retry_after_ms: u64 },
    
    /// Memory budget exceeded
    MemoryLimitExceeded { limit_bytes: usize, requested_bytes: usize },
    
    /// Circuit breaker is open (system under stress)
    CircuitOpen { failures: usize, reset_after_ms: u64 },
    
    /// Search timeout exceeded
    TimeoutExceeded { timeout_ms: u64 },
    
    /// Vector ID not found
    NotFound { id: u32 },
    
    /// Invalid filter syntax
    InvalidFilter { message: String },
    
    /// WAL corruption detected
    WalCorrupted { offset: u64, message: String },
    
    /// Snapshot format invalid
    SnapshotInvalid { version: u32, expected: u32 },
    
    /// WebGPU not available
    WebGpuUnavailable,
    
    /// Generic internal error
    Internal { message: String },
    
    /// Invalid or missing API key
    InvalidApiKey,
    
    /// Token/key has expired
    AuthExpired,
    
    /// Insufficient permissions for operation
    InsufficientPermissions { required: String },
    
    /// Access denied to collection
    AccessDenied { collection: String },
}

/// Rich error with context and recovery suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WinnowError {
    /// Error code for programmatic handling
    pub code: String,
    
    /// Human-readable message
    pub message: String,
    
    /// Detailed error kind
    pub kind: WinnowErrorKind,
    
    /// Documentation URL
    pub docs_url: String,
    
    /// Suggested fix
    pub fix: String,
}

impl WinnowError {
    pub fn dimension_mismatch(expected: usize, got: usize) -> Self {
        Self {
            code: "DIMENSION_MISMATCH".to_string(),
            message: format!("Vector dimension mismatch: expected {}, got {}", expected, got),
            kind: WinnowErrorKind::DimensionMismatch { expected, got },
            docs_url: "https://docs.winnowdb.com/errors/dimension-mismatch".to_string(),
            fix: format!("Ensure all vectors have {} dimensions matching the collection config", expected),
        }
    }
    
    pub fn index_not_trained(index_type: &str) -> Self {
        Self {
            code: "INDEX_NOT_TRAINED".to_string(),
            message: format!("{} index requires training before use", index_type),
            kind: WinnowErrorKind::IndexNotTrained { index_type: index_type.to_string() },
            docs_url: "https://docs.winnowdb.com/errors/index-not-trained".to_string(),
            fix: "Call collection.train(trainingData) with representative vectors before adding".to_string(),
        }
    }
    
    pub fn rate_limit_exceeded(retry_after_ms: u64) -> Self {
        Self {
            code: "RATE_LIMIT_EXCEEDED".to_string(),
            message: "Rate limit exceeded".to_string(),
            kind: WinnowErrorKind::RateLimitExceeded { retry_after_ms },
            docs_url: "https://docs.winnowdb.com/errors/rate-limit".to_string(),
            fix: format!("Wait {}ms before retrying", retry_after_ms),
        }
    }
    
    pub fn memory_limit_exceeded(limit: usize, requested: usize) -> Self {
        Self {
            code: "MEMORY_LIMIT_EXCEEDED".to_string(),
            message: format!("Memory limit exceeded: {} bytes requested, {} bytes available", requested, limit),
            kind: WinnowErrorKind::MemoryLimitExceeded { limit_bytes: limit, requested_bytes: requested },
            docs_url: "https://docs.winnowdb.com/errors/memory-limit".to_string(),
            fix: "Call collection.compact() to reclaim memory or increase memory budget".to_string(),
        }
    }
    
    pub fn circuit_open(failures: usize, reset_after_ms: u64) -> Self {
        Self {
            code: "CIRCUIT_OPEN".to_string(),
            message: "Circuit breaker is open - system under stress".to_string(),
            kind: WinnowErrorKind::CircuitOpen { failures, reset_after_ms },
            docs_url: "https://docs.winnowdb.com/errors/circuit-breaker".to_string(),
            fix: format!("System is recovering. Wait {}ms before retrying", reset_after_ms),
        }
    }
    
    pub fn timeout_exceeded(timeout_ms: u64) -> Self {
        Self {
            code: "TIMEOUT_EXCEEDED".to_string(),
            message: format!("Search timeout exceeded: {}ms", timeout_ms),
            kind: WinnowErrorKind::TimeoutExceeded { timeout_ms },
            docs_url: "https://docs.winnowdb.com/errors/timeout".to_string(),
            fix: "Reduce k, use filter to narrow results, or increase timeout".to_string(),
        }
    }
    
    pub fn not_found(id: u32) -> Self {
        Self {
            code: "NOT_FOUND".to_string(),
            message: format!("Vector with ID {} not found", id),
            kind: WinnowErrorKind::NotFound { id },
            docs_url: "https://docs.winnowdb.com/errors/not-found".to_string(),
            fix: "Check if the vector exists or was previously deleted".to_string(),
        }
    }
    
    pub fn invalid_filter(message: &str) -> Self {
        Self {
            code: "INVALID_FILTER".to_string(),
            message: format!("Invalid filter: {}", message),
            kind: WinnowErrorKind::InvalidFilter { message: message.to_string() },
            docs_url: "https://docs.winnowdb.com/errors/invalid-filter".to_string(),
            fix: "Check filter syntax. Supported: $eq, $ne, $gt, $lt, $in, $and, $or".to_string(),
        }
    }
    
    pub fn wal_corrupted(offset: u64, message: &str) -> Self {
        Self {
            code: "WAL_CORRUPTED".to_string(),
            message: format!("WAL corrupted at offset {}: {}", offset, message),
            kind: WinnowErrorKind::WalCorrupted { offset, message: message.to_string() },
            docs_url: "https://docs.winnowdb.com/errors/wal-corruption".to_string(),
            fix: "Restore from latest snapshot. Data after corruption point may be lost".to_string(),
        }
    }
    
    pub fn snapshot_invalid(version: u32, expected: u32) -> Self {
        Self {
            code: "SNAPSHOT_INVALID".to_string(),
            message: format!("Snapshot version {} not supported, expected {}", version, expected),
            kind: WinnowErrorKind::SnapshotInvalid { version, expected },
            docs_url: "https://docs.winnowdb.com/errors/snapshot-version".to_string(),
            fix: "Upgrade WinnowDB or use a compatible snapshot".to_string(),
        }
    }
    
    pub fn webgpu_unavailable() -> Self {
        Self {
            code: "WEBGPU_UNAVAILABLE".to_string(),
            message: "WebGPU is not available in this browser".to_string(),
            kind: WinnowErrorKind::WebGpuUnavailable,
            docs_url: "https://docs.winnowdb.com/errors/webgpu".to_string(),
            fix: "Use Chrome 113+, Firefox 118+, or Safari 17+ with WebGPU enabled".to_string(),
        }
    }
    
    pub fn internal(message: &str) -> Self {
        Self {
            code: "INTERNAL_ERROR".to_string(),
            message: message.to_string(),
            kind: WinnowErrorKind::Internal { message: message.to_string() },
            docs_url: "https://docs.winnowdb.com/errors/internal".to_string(),
            fix: "This is an internal error. Please report it with reproduction steps".to_string(),
        }
    }
    
    pub fn invalid_api_key() -> Self {
        Self {
            code: "INVALID_API_KEY".to_string(),
            message: "Invalid or missing API key".to_string(),
            kind: WinnowErrorKind::InvalidApiKey,
            docs_url: "https://docs.winnowdb.com/errors/auth".to_string(),
            fix: "Provide a valid API key in the Authorization header".to_string(),
        }
    }
    
    pub fn auth_expired() -> Self {
        Self {
            code: "AUTH_EXPIRED".to_string(),
            message: "Authentication token has expired".to_string(),
            kind: WinnowErrorKind::AuthExpired,
            docs_url: "https://docs.winnowdb.com/errors/auth".to_string(),
            fix: "Request a new API key or refresh your token".to_string(),
        }
    }
    
    pub fn insufficient_permissions(required: &str) -> Self {
        Self {
            code: "INSUFFICIENT_PERMISSIONS".to_string(),
            message: format!("Insufficient permissions for {} operation", required),
            kind: WinnowErrorKind::InsufficientPermissions { required: required.to_string() },
            docs_url: "https://docs.winnowdb.com/errors/auth".to_string(),
            fix: "Request elevated permissions from an admin".to_string(),
        }
    }
    
    pub fn access_denied(collection: &str) -> Self {
        Self {
            code: "ACCESS_DENIED".to_string(),
            message: format!("Access denied to collection '{}'", collection),
            kind: WinnowErrorKind::AccessDenied { collection: collection.to_string() },
            docs_url: "https://docs.winnowdb.com/errors/auth".to_string(),
            fix: "Request access to this collection from an admin".to_string(),
        }
    }
    
    /// Convert to JsValue for WASM boundary
    pub fn to_js(&self) -> JsValue {
        serde_wasm_bindgen::to_value(self).unwrap_or_else(|_| JsValue::from_str(&self.message))
    }
}

impl fmt::Display for WinnowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}\n\nFix: {}\nDocs: {}", 
            self.code, self.message, self.fix, self.docs_url)
    }
}

impl std::error::Error for WinnowError {}

impl From<WinnowError> for JsValue {
    fn from(err: WinnowError) -> Self {
        err.to_js()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_serialization() {
        let err = WinnowError::dimension_mismatch(768, 512);
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("DIMENSION_MISMATCH"));
        assert!(json.contains("768"));
        assert!(json.contains("512"));
    }
}
