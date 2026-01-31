//! Compliance and Governance Module
//! 
//! Handles GDPR, SOC2, and HIPAA compliance requirements including
//! data retention, deletion (Right to be Forgotten), and audit policy enforcement.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// ============================================================================
// Compliance Standards
// ============================================================================

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum ComplianceStandard {
    Gdpr,
    Hipaa,
    Soc2,
    Ccmpa,
}

// ============================================================================
// Compliance Policy
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Data retention in days
    pub retention_days: u32,
    /// Whether to archive before deletion
    pub archive_before_delete: bool,
    /// Specific data categories this applies to
    pub categories: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncryptionPolicy {
    /// Encryption at rest enabled
    pub at_rest_enabled: bool,
    /// Key rotation interval (days)
    pub key_rotation_days: u32,
    /// Minimum TLS version
    pub min_tls_version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompliancePolicy {
    /// Enabled standards
    pub standards: Vec<ComplianceStandard>,
    /// Retention settings
    pub retention: RetentionPolicy,
    /// Encryption settings
    pub encryption: EncryptionPolicy,
    /// Geo-fencing (allowed regions)
    pub allowed_regions: Vec<String>,
}

impl Default for CompliancePolicy {
    fn default() -> Self {
        Self {
            standards: Vec::new(),
            retention: RetentionPolicy {
                retention_days: 365,
                archive_before_delete: true,
                categories: vec!["all".to_string()],
            },
            encryption: EncryptionPolicy {
                at_rest_enabled: true,
                key_rotation_days: 90,
                min_tls_version: "1.2".to_string(),
            },
            allowed_regions: vec!["us-east-1".to_string()],
        }
    }
}

// ============================================================================
// Audit Logging (Enforcement)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataAccessLog {
    pub user_id: String,
    pub collection: String,
    pub operation: String,
    pub timestamp: f64,
    pub purpose: Option<String>, // Required for some GDPR/HIPAA access
}

// ============================================================================
// Compliance Manager
// ============================================================================

#[wasm_bindgen]
pub struct ComplianceManager {
    policy: CompliancePolicy,
}

#[wasm_bindgen]
impl ComplianceManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            policy: CompliancePolicy::default(),
        }
    }

    /// Enable a compliance standard
    pub fn enable_standard(&mut self, standard: &str) -> Result<(), String> {
        let std = match standard.to_lowercase().as_str() {
            "gdpr" => ComplianceStandard::Gdpr,
            "hipaa" => ComplianceStandard::Hipaa,
            "soc2" => ComplianceStandard::Soc2,
            "ccmpa" => ComplianceStandard::Ccmpa,
            _ => return Err(format!("Unknown standard: {}", standard)),
        };
        
        if !self.policy.standards.contains(&std) {
            self.policy.standards.push(std);
        }
        
        // Auto-configure strict defaults if enabling HIPAA
        if std == ComplianceStandard::Hipaa {
            self.policy.encryption.at_rest_enabled = true;
            self.policy.retention.retention_days = 2190; // 6 years often required
        }
        
        Ok(())
    }

    /// Update retention policy
    pub fn set_retention(&mut self, days: u32) {
        self.policy.retention.retention_days = days;
    }

    /// Register a data deletion request (GDPR Right to be Forgotten)
    /// Returns a confirmation ID
    pub fn request_deletion(&self, user_id: &str, justification: &str) -> String {
        // In a real system, this would trigger an async job
        // Here we generate a mock ID
        format!("del_req_{}_{}", user_id, 12345)
    }

    /// Generate compliance report (Mock)
    pub fn generate_report(&self) -> JsValue {
        let mut report = HashMap::new();
        report.insert("encryption_status", if self.policy.encryption.at_rest_enabled { "COMPLIANT" } else { "NON_COMPLIANT" });
        report.insert("retention_policy", "ACTIVE");
        report.insert("active_standards", "CHECKED");
        
        serde_wasm_bindgen::to_value(&report).unwrap_or(JsValue::NULL)
    }
    
    /// Get current policy
    pub fn get_policy(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.policy).unwrap_or(JsValue::NULL)
    }
}

impl Default for ComplianceManager {
    fn default() -> Self {
        Self::new()
    }
}
