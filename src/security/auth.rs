//! Authentication and authorization for WinnowDB
//!
//! Provides API key authentication and role-based access control.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use wasm_bindgen::prelude::*;
use crate::error::WinnowError;

/// User roles for RBAC
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    /// Full access: read, write, delete, admin operations
    Admin,
    /// Read and write access
    ReadWrite,
    /// Read-only access
    ReadOnly,
}

impl Role {
    pub fn can_read(&self) -> bool {
        true // All roles can read
    }
    
    pub fn can_write(&self) -> bool {
        matches!(self, Role::Admin | Role::ReadWrite)
    }
    
    pub fn can_delete(&self) -> bool {
        matches!(self, Role::Admin | Role::ReadWrite)
    }
    
    pub fn can_admin(&self) -> bool {
        matches!(self, Role::Admin)
    }
}

/// Authentication context for a verified user/key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthContext {
    /// Unique user/key identifier
    pub user_id: String,
    /// Assigned roles
    pub roles: Vec<Role>,
    /// Collections this user can access (empty = all)
    pub allowed_collections: HashSet<String>,
    /// When this token/key expires (ms since epoch, 0 = never)
    pub expires_at: u64,
}

impl AuthContext {
    pub fn has_role(&self, role: Role) -> bool {
        self.roles.contains(&role)
    }
    
    pub fn highest_role(&self) -> Role {
        if self.has_role(Role::Admin) {
            Role::Admin
        } else if self.has_role(Role::ReadWrite) {
            Role::ReadWrite
        } else {
            Role::ReadOnly
        }
    }
    
    pub fn can_access_collection(&self, collection: &str) -> bool {
        self.allowed_collections.is_empty() || 
        self.allowed_collections.contains(collection)
    }
    
    pub fn is_expired(&self) -> bool {
        if self.expires_at == 0 {
            return false;
        }
        js_sys::Date::now() as u64 > self.expires_at
    }
    
    /// Check read permission for a collection
    pub fn check_read(&self, collection: &str) -> Result<(), WinnowError> {
        if self.is_expired() {
            return Err(WinnowError::auth_expired());
        }
        if !self.can_access_collection(collection) {
            return Err(WinnowError::access_denied(collection));
        }
        if !self.highest_role().can_read() {
            return Err(WinnowError::insufficient_permissions("read"));
        }
        Ok(())
    }
    
    /// Check write permission for a collection
    pub fn check_write(&self, collection: &str) -> Result<(), WinnowError> {
        if self.is_expired() {
            return Err(WinnowError::auth_expired());
        }
        if !self.can_access_collection(collection) {
            return Err(WinnowError::access_denied(collection));
        }
        if !self.highest_role().can_write() {
            return Err(WinnowError::insufficient_permissions("write"));
        }
        Ok(())
    }
    
    /// Check admin permission
    pub fn check_admin(&self) -> Result<(), WinnowError> {
        if self.is_expired() {
            return Err(WinnowError::auth_expired());
        }
        if !self.highest_role().can_admin() {
            return Err(WinnowError::insufficient_permissions("admin"));
        }
        Ok(())
    }
}

/// API Key authentication provider
#[derive(Debug, Clone)]
pub struct ApiKeyAuth {
    /// Map of API key -> AuthContext
    keys: HashMap<String, AuthContext>,
}

impl ApiKeyAuth {
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
        }
    }
    
    /// Register a new API key
    pub fn register_key(&mut self, key: String, context: AuthContext) {
        self.keys.insert(key, context);
    }
    
    /// Revoke an API key
    pub fn revoke_key(&mut self, key: &str) -> bool {
        self.keys.remove(key).is_some()
    }
    
    /// Verify an API key and return its auth context
    pub fn verify(&self, key: &str) -> Result<AuthContext, WinnowError> {
        match self.keys.get(key) {
            Some(ctx) => {
                if ctx.is_expired() {
                    Err(WinnowError::auth_expired())
                } else {
                    Ok(ctx.clone())
                }
            },
            None => Err(WinnowError::invalid_api_key()),
        }
    }
    
    /// List all registered key IDs (not the keys themselves)
    pub fn list_users(&self) -> Vec<String> {
        self.keys.values().map(|c| c.user_id.clone()).collect()
    }
    
    /// Create an admin key (for bootstrap)
    pub fn create_admin_key(user_id: &str) -> (String, AuthContext) {
        let key = generate_api_key();
        let context = AuthContext {
            user_id: user_id.to_string(),
            roles: vec![Role::Admin],
            allowed_collections: HashSet::new(),
            expires_at: 0, // Never expires
        };
        (key, context)
    }
    
    /// Create a read-only key for specific collections
    pub fn create_readonly_key(user_id: &str, collections: Vec<String>) -> (String, AuthContext) {
        let key = generate_api_key();
        let context = AuthContext {
            user_id: user_id.to_string(),
            roles: vec![Role::ReadOnly],
            allowed_collections: collections.into_iter().collect(),
            expires_at: 0,
        };
        (key, context)
    }
}

impl Default for ApiKeyAuth {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a random API key (32 hex characters)
fn generate_api_key() -> String {
    use js_sys::Math;
    let mut key = String::with_capacity(32);
    for _ in 0..32 {
        let n = (Math::random() * 16.0).floor() as u8;
        let c = if n < 10 { b'0' + n } else { b'a' + (n - 10) };
        key.push(c as char);
    }
    key
}

/// WASM-exposed authentication manager
#[wasm_bindgen]
pub struct AuthManager {
    auth: ApiKeyAuth,
}

#[wasm_bindgen]
impl AuthManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            auth: ApiKeyAuth::new(),
        }
    }
    
    /// Create and register an admin key, returns the key
    pub fn create_admin(&mut self, user_id: &str) -> String {
        let (key, ctx) = ApiKeyAuth::create_admin_key(user_id);
        self.auth.register_key(key.clone(), ctx);
        key
    }
    
    /// Create and register a read-only key
    pub fn create_readonly(&mut self, user_id: &str, collections: JsValue) -> Result<String, JsValue> {
        let collections: Vec<String> = serde_wasm_bindgen::from_value(collections)?;
        let (key, ctx) = ApiKeyAuth::create_readonly_key(user_id, collections);
        self.auth.register_key(key.clone(), ctx);
        Ok(key)
    }
    
    /// Verify an API key, returns user info or error
    pub fn verify(&self, key: &str) -> Result<JsValue, JsValue> {
        let ctx = self.auth.verify(key).map_err(|e| JsValue::from(e))?;
        Ok(serde_wasm_bindgen::to_value(&ctx)?)
    }
    
    /// Revoke a key
    pub fn revoke(&mut self, key: &str) -> bool {
        self.auth.revoke_key(key)
    }
    
    /// List all user IDs
    pub fn list_users(&self) -> JsValue {
        let users = self.auth.list_users();
        serde_wasm_bindgen::to_value(&users).unwrap_or(JsValue::NULL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_role_permissions() {
        assert!(Role::Admin.can_read());
        assert!(Role::Admin.can_write());
        assert!(Role::Admin.can_admin());
        
        assert!(Role::ReadWrite.can_read());
        assert!(Role::ReadWrite.can_write());
        assert!(!Role::ReadWrite.can_admin());
        
        assert!(Role::ReadOnly.can_read());
        assert!(!Role::ReadOnly.can_write());
        assert!(!Role::ReadOnly.can_admin());
    }
    
    #[test]
    fn test_collection_access() {
        let ctx = AuthContext {
            user_id: "test".to_string(),
            roles: vec![Role::ReadOnly],
            allowed_collections: vec!["col1".to_string()].into_iter().collect(),
            expires_at: 0,
        };
        
        assert!(ctx.can_access_collection("col1"));
        assert!(!ctx.can_access_collection("col2"));
    }
}
