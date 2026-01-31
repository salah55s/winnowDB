//! Multi-Tenancy support for WinnowDB
//!
//! Provides tenant isolation, resource quotas, and usage tracking.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::{Mutex, RwLock};
use wasm_bindgen::prelude::*;

// ============================================================================
// Tenant Model
// ============================================================================

/// Tenant status
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum TenantStatus {
    Active,
    Suspended,
    Deleted,
    Trial,
}

/// Tenant tier for quotas
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum TenantTier {
    Free,
    Starter,
    Professional,
    Enterprise,
}

/// Tenant configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tenant {
    pub id: String,
    pub name: String,
    pub tier: TenantTier,
    pub status: TenantStatus,
    pub created_at: f64,
    pub metadata: HashMap<String, String>,
}

impl Tenant {
    pub fn new(id: &str, name: &str, tier: TenantTier) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            tier,
            status: TenantStatus::Active,
            created_at: js_sys::Date::now(),
            metadata: HashMap::new(),
        }
    }
}

// ============================================================================
// Resource Quotas
// ============================================================================

/// Resource quota limits
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceQuota {
    /// Maximum number of collections
    pub max_collections: u32,
    /// Maximum vectors per collection
    pub max_vectors_per_collection: u64,
    /// Maximum total vectors across all collections
    pub max_total_vectors: u64,
    /// Maximum storage in bytes
    pub max_storage_bytes: u64,
    /// Maximum queries per minute
    pub max_qpm: u32,
    /// Maximum concurrent operations
    pub max_concurrent_ops: u32,
    /// Maximum dimensions per vector
    pub max_dimensions: u32,
}

impl ResourceQuota {
    pub fn for_tier(tier: TenantTier) -> Self {
        match tier {
            TenantTier::Free => Self {
                max_collections: 3,
                max_vectors_per_collection: 10_000,
                max_total_vectors: 25_000,
                max_storage_bytes: 100 * 1024 * 1024, // 100 MB
                max_qpm: 60,
                max_concurrent_ops: 2,
                max_dimensions: 768,
            },
            TenantTier::Starter => Self {
                max_collections: 10,
                max_vectors_per_collection: 100_000,
                max_total_vectors: 500_000,
                max_storage_bytes: 1024 * 1024 * 1024, // 1 GB
                max_qpm: 600,
                max_concurrent_ops: 5,
                max_dimensions: 1536,
            },
            TenantTier::Professional => Self {
                max_collections: 50,
                max_vectors_per_collection: 1_000_000,
                max_total_vectors: 10_000_000,
                max_storage_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
                max_qpm: 6000,
                max_concurrent_ops: 20,
                max_dimensions: 4096,
            },
            TenantTier::Enterprise => Self {
                max_collections: u32::MAX,
                max_vectors_per_collection: u64::MAX,
                max_total_vectors: u64::MAX,
                max_storage_bytes: u64::MAX,
                max_qpm: u32::MAX,
                max_concurrent_ops: u32::MAX,
                max_dimensions: u32::MAX,
            },
        }
    }
}

// ============================================================================
// Usage Tracking
// ============================================================================

/// Current resource usage
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub collections: u32,
    pub total_vectors: u64,
    pub storage_bytes: u64,
    pub queries_this_minute: u32,
    pub current_concurrent_ops: u32,
    pub last_query_time: f64,
}

/// Quota check result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuotaCheckResult {
    pub allowed: bool,
    pub reason: Option<String>,
    pub usage_percent: f64,
}

// ============================================================================
// Tenant Manager
// ============================================================================

/// Manages tenant lifecycle and quotas
#[derive(Clone)]
pub struct TenantManager {
    tenants: Arc<RwLock<HashMap<String, Tenant>>>,
    quotas: Arc<RwLock<HashMap<String, ResourceQuota>>>,
    usage: Arc<RwLock<HashMap<String, ResourceUsage>>>,
    collection_owners: Arc<RwLock<HashMap<String, String>>>, // collection_id -> tenant_id
}

impl TenantManager {
    pub fn new() -> Self {
        Self {
            tenants: Arc::new(RwLock::new(HashMap::new())),
            quotas: Arc::new(RwLock::new(HashMap::new())),
            usage: Arc::new(RwLock::new(HashMap::new())),
            collection_owners: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Create a new tenant
    pub fn create_tenant(&self, id: &str, name: &str, tier: TenantTier) -> Result<Tenant, String> {
        let mut tenants = self.tenants.write();
        if tenants.contains_key(id) {
            return Err(format!("Tenant {} already exists", id));
        }
        
        let tenant = Tenant::new(id, name, tier);
        tenants.insert(id.to_string(), tenant.clone());
        
        // Set up quota and usage tracking
        self.quotas.write().insert(id.to_string(), ResourceQuota::for_tier(tier));
        self.usage.write().insert(id.to_string(), ResourceUsage::default());
        
        Ok(tenant)
    }
    
    /// Get tenant by ID
    pub fn get_tenant(&self, id: &str) -> Option<Tenant> {
        self.tenants.read().get(id).cloned()
    }
    
    /// Update tenant tier
    pub fn update_tier(&self, id: &str, tier: TenantTier) -> Result<(), String> {
        let mut tenants = self.tenants.write();
        let tenant = tenants.get_mut(id).ok_or_else(|| format!("Tenant {} not found", id))?;
        tenant.tier = tier;
        
        // Update quota
        self.quotas.write().insert(id.to_string(), ResourceQuota::for_tier(tier));
        
        Ok(())
    }
    
    /// Suspend tenant
    pub fn suspend_tenant(&self, id: &str) -> Result<(), String> {
        let mut tenants = self.tenants.write();
        let tenant = tenants.get_mut(id).ok_or_else(|| format!("Tenant {} not found", id))?;
        tenant.status = TenantStatus::Suspended;
        Ok(())
    }
    
    /// Check if operation is allowed by quota
    pub fn check_quota(&self, tenant_id: &str, operation: &str, amount: u64) -> QuotaCheckResult {
        let quotas = self.quotas.read();
        let usage = self.usage.read();
        
        let quota = match quotas.get(tenant_id) {
            Some(q) => q,
            None => return QuotaCheckResult {
                allowed: false,
                reason: Some("Tenant not found".to_string()),
                usage_percent: 0.0,
            },
        };
        
        let current = usage.get(tenant_id).cloned().unwrap_or_default();
        
        match operation {
            "create_collection" => {
                let allowed = current.collections < quota.max_collections;
                QuotaCheckResult {
                    allowed,
                    reason: if !allowed { Some("Collection limit reached".to_string()) } else { None },
                    usage_percent: (current.collections as f64 / quota.max_collections as f64) * 100.0,
                }
            },
            "insert_vectors" => {
                let new_total = current.total_vectors + amount;
                let allowed = new_total <= quota.max_total_vectors;
                QuotaCheckResult {
                    allowed,
                    reason: if !allowed { Some("Vector limit reached".to_string()) } else { None },
                    usage_percent: (current.total_vectors as f64 / quota.max_total_vectors as f64) * 100.0,
                }
            },
            "query" => {
                let now = js_sys::Date::now();
                let minute_start = (now / 60_000.0).floor() * 60_000.0;
                let queries = if current.last_query_time >= minute_start {
                    current.queries_this_minute
                } else {
                    0
                };
                let allowed = queries < quota.max_qpm;
                QuotaCheckResult {
                    allowed,
                    reason: if !allowed { Some("Rate limit exceeded".to_string()) } else { None },
                    usage_percent: (queries as f64 / quota.max_qpm as f64) * 100.0,
                }
            },
            _ => QuotaCheckResult {
                allowed: true,
                reason: None,
                usage_percent: 0.0,
            },
        }
    }
    
    /// Record resource usage
    pub fn record_usage(&self, tenant_id: &str, operation: &str, amount: u64) {
        let mut usage = self.usage.write();
        let current = usage.entry(tenant_id.to_string()).or_insert_with(ResourceUsage::default);
        
        match operation {
            "create_collection" => current.collections += 1,
            "delete_collection" => current.collections = current.collections.saturating_sub(1),
            "insert_vectors" => current.total_vectors += amount,
            "delete_vectors" => current.total_vectors = current.total_vectors.saturating_sub(amount),
            "storage" => current.storage_bytes = amount,
            "query" => {
                let now = js_sys::Date::now();
                let minute_start = (now / 60_000.0).floor() * 60_000.0;
                if current.last_query_time < minute_start {
                    current.queries_this_minute = 1;
                } else {
                    current.queries_this_minute += 1;
                }
                current.last_query_time = now;
            },
            _ => {},
        }
    }
    
    /// Register collection ownership
    pub fn register_collection(&self, collection_id: &str, tenant_id: &str) {
        self.collection_owners.write().insert(collection_id.to_string(), tenant_id.to_string());
    }
    
    /// Get tenant for collection
    pub fn get_collection_tenant(&self, collection_id: &str) -> Option<String> {
        self.collection_owners.read().get(collection_id).cloned()
    }
    
    /// Get usage for tenant
    pub fn get_usage(&self, tenant_id: &str) -> Option<ResourceUsage> {
        self.usage.read().get(tenant_id).cloned()
    }
    
    /// Get quota for tenant
    pub fn get_quota(&self, tenant_id: &str) -> Option<ResourceQuota> {
        self.quotas.read().get(tenant_id).cloned()
    }
    
    /// List all tenants
    pub fn list_tenants(&self) -> Vec<Tenant> {
        self.tenants.read().values().cloned().collect()
    }
}

impl Default for TenantManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Billing Integration
// ============================================================================

/// Billing event types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BillingEvent {
    VectorsStored { count: u64 },
    QueriesExecuted { count: u32 },
    StorageUsed { bytes: u64 },
    TierUpgrade { from: TenantTier, to: TenantTier },
}

/// Billing record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BillingRecord {
    pub tenant_id: String,
    pub event: BillingEvent,
    pub timestamp: f64,
    pub amount_cents: u64,
}

/// Simple billing tracker
#[derive(Clone)]
pub struct BillingTracker {
    records: Arc<Mutex<Vec<BillingRecord>>>,
    pricing: BillingPricing,
}

/// Pricing configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BillingPricing {
    /// Price per 1M vectors stored per month (cents)
    pub vectors_per_million_cents: u64,
    /// Price per 1M queries (cents)
    pub queries_per_million_cents: u64,
    /// Price per GB storage per month (cents)
    pub storage_per_gb_cents: u64,
}

impl Default for BillingPricing {
    fn default() -> Self {
        Self {
            vectors_per_million_cents: 100, // $1 per million
            queries_per_million_cents: 50,  // $0.50 per million
            storage_per_gb_cents: 10,       // $0.10 per GB
        }
    }
}

impl BillingTracker {
    pub fn new(pricing: BillingPricing) -> Self {
        Self {
            records: Arc::new(Mutex::new(Vec::new())),
            pricing,
        }
    }
    
    /// Record a billing event
    pub fn record(&self, tenant_id: &str, event: BillingEvent) {
        let amount = match &event {
            BillingEvent::VectorsStored { count } => {
                (*count as f64 / 1_000_000.0 * self.pricing.vectors_per_million_cents as f64) as u64
            },
            BillingEvent::QueriesExecuted { count } => {
                (*count as f64 / 1_000_000.0 * self.pricing.queries_per_million_cents as f64) as u64
            },
            BillingEvent::StorageUsed { bytes } => {
                (*bytes as f64 / (1024.0 * 1024.0 * 1024.0) * self.pricing.storage_per_gb_cents as f64) as u64
            },
            BillingEvent::TierUpgrade { .. } => 0,
        };
        
        self.records.lock().push(BillingRecord {
            tenant_id: tenant_id.to_string(),
            event,
            timestamp: js_sys::Date::now(),
            amount_cents: amount,
        });
    }
    
    /// Get billing summary for tenant
    pub fn summary(&self, tenant_id: &str) -> u64 {
        self.records.lock()
            .iter()
            .filter(|r| r.tenant_id == tenant_id)
            .map(|r| r.amount_cents)
            .sum()
    }
}

impl Default for BillingTracker {
    fn default() -> Self {
        Self::new(BillingPricing::default())
    }
}

// ============================================================================
// WASM Bindings
// ============================================================================

#[wasm_bindgen]
pub struct MultiTenancyManager {
    tenants: TenantManager,
    billing: BillingTracker,
}

#[wasm_bindgen]
impl MultiTenancyManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            tenants: TenantManager::default(),
            billing: BillingTracker::default(),
        }
    }
    
    /// Create a new tenant
    pub fn create_tenant(&self, id: &str, name: &str, tier: &str) -> Result<JsValue, JsValue> {
        let tier = match tier {
            "free" => TenantTier::Free,
            "starter" => TenantTier::Starter,
            "professional" => TenantTier::Professional,
            "enterprise" => TenantTier::Enterprise,
            _ => TenantTier::Free,
        };
        
        self.tenants.create_tenant(id, name, tier)
            .map(|t| serde_wasm_bindgen::to_value(&t).unwrap())
            .map_err(|e| JsValue::from_str(&e))
    }
    
    /// Get tenant by ID
    pub fn get_tenant(&self, id: &str) -> JsValue {
        match self.tenants.get_tenant(id) {
            Some(t) => serde_wasm_bindgen::to_value(&t).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }
    
    /// Check if operation is allowed
    pub fn check_quota(&self, tenant_id: &str, operation: &str, amount: u64) -> JsValue {
        let result = self.tenants.check_quota(tenant_id, operation, amount);
        serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
    }
    
    /// Record usage
    pub fn record_usage(&self, tenant_id: &str, operation: &str, amount: u64) {
        self.tenants.record_usage(tenant_id, operation, amount);
    }
    
    /// Get usage for tenant
    pub fn get_usage(&self, tenant_id: &str) -> JsValue {
        match self.tenants.get_usage(tenant_id) {
            Some(u) => serde_wasm_bindgen::to_value(&u).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }
    
    /// Get quota for tenant
    pub fn get_quota(&self, tenant_id: &str) -> JsValue {
        match self.tenants.get_quota(tenant_id) {
            Some(q) => serde_wasm_bindgen::to_value(&q).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }
    
    /// Update tenant tier
    pub fn update_tier(&self, tenant_id: &str, tier: &str) -> Result<(), JsValue> {
        let tier = match tier {
            "free" => TenantTier::Free,
            "starter" => TenantTier::Starter,
            "professional" => TenantTier::Professional,
            "enterprise" => TenantTier::Enterprise,
            _ => return Err(JsValue::from_str("Invalid tier")),
        };
        
        self.tenants.update_tier(tenant_id, tier)
            .map_err(|e| JsValue::from_str(&e))
    }
    
    /// Suspend tenant
    pub fn suspend_tenant(&self, tenant_id: &str) -> Result<(), JsValue> {
        self.tenants.suspend_tenant(tenant_id)
            .map_err(|e| JsValue::from_str(&e))
    }
    
    /// List all tenants
    pub fn list_tenants(&self) -> JsValue {
        let tenants = self.tenants.list_tenants();
        serde_wasm_bindgen::to_value(&tenants).unwrap_or(JsValue::NULL)
    }
    
    /// Get billing summary
    pub fn billing_summary(&self, tenant_id: &str) -> u64 {
        self.billing.summary(tenant_id)
    }
}

impl Default for MultiTenancyManager {
    fn default() -> Self {
        Self::new()
    }
}
