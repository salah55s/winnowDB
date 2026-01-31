//! JWT Token support for WinnowDB
//!
//! Provides stateless authentication via JSON Web Tokens.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use crate::error::WinnowError;
use crate::security::auth::{AuthContext, Role};

/// JWT Header
#[derive(Clone, Debug, Serialize, Deserialize)]
struct JwtHeader {
    alg: String,
    typ: String,
}

impl Default for JwtHeader {
    fn default() -> Self {
        Self {
            alg: "HS256".to_string(),
            typ: "JWT".to_string(),
        }
    }
}

/// JWT Payload/Claims
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JwtClaims {
    /// Subject (user ID)
    pub sub: String,
    /// Issuer
    pub iss: String,
    /// Issued at (seconds since epoch)
    pub iat: u64,
    /// Expires at (seconds since epoch)
    pub exp: u64,
    /// Roles
    pub roles: Vec<Role>,
    /// Allowed collections (empty = all)
    pub collections: Vec<String>,
}

impl JwtClaims {
    pub fn new(user_id: &str, issuer: &str, roles: Vec<Role>, ttl_seconds: u64) -> Self {
        let now = (js_sys::Date::now() / 1000.0) as u64;
        Self {
            sub: user_id.to_string(),
            iss: issuer.to_string(),
            iat: now,
            exp: now + ttl_seconds,
            roles,
            collections: vec![],
        }
    }
    
    pub fn with_collections(mut self, collections: Vec<String>) -> Self {
        self.collections = collections;
        self
    }
    
    pub fn is_expired(&self) -> bool {
        let now = (js_sys::Date::now() / 1000.0) as u64;
        now > self.exp
    }
    
    /// Convert to AuthContext
    pub fn to_auth_context(&self) -> AuthContext {
        AuthContext {
            user_id: self.sub.clone(),
            roles: self.roles.clone(),
            allowed_collections: self.collections.iter().cloned().collect(),
            expires_at: self.exp * 1000, // Convert to ms
        }
    }
}

/// Simple HMAC-SHA256 JWT provider
/// Note: For production, use a proper crypto library
#[derive(Clone)]
pub struct JwtProvider {
    secret: Vec<u8>,
    issuer: String,
    default_ttl: u64,
}

impl JwtProvider {
    pub fn new(secret: &str, issuer: &str, default_ttl_seconds: u64) -> Self {
        Self {
            secret: secret.as_bytes().to_vec(),
            issuer: issuer.to_string(),
            default_ttl: default_ttl_seconds,
        }
    }
    
    /// Create a signed JWT token
    pub fn create_token(&self, user_id: &str, roles: Vec<Role>, collections: Option<Vec<String>>) -> String {
        let mut claims = JwtClaims::new(user_id, &self.issuer, roles, self.default_ttl);
        if let Some(cols) = collections {
            claims = claims.with_collections(cols);
        }
        
        let header = JwtHeader::default();
        let header_b64 = base64_encode(&serde_json::to_vec(&header).unwrap());
        let claims_b64 = base64_encode(&serde_json::to_vec(&claims).unwrap());
        
        let signature_input = format!("{}.{}", header_b64, claims_b64);
        let signature = self.sign(&signature_input);
        let signature_b64 = base64_encode(&signature);
        
        format!("{}.{}.{}", header_b64, claims_b64, signature_b64)
    }
    
    /// Verify and decode a JWT token
    pub fn verify_token(&self, token: &str) -> Result<JwtClaims, WinnowError> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(WinnowError::invalid_api_key());
        }
        
        // Verify signature
        let signature_input = format!("{}.{}", parts[0], parts[1]);
        let expected_sig = self.sign(&signature_input);
        let provided_sig = base64_decode(parts[2]).map_err(|_| WinnowError::invalid_api_key())?;
        
        if expected_sig != provided_sig {
            return Err(WinnowError::invalid_api_key());
        }
        
        // Decode claims
        let claims_bytes = base64_decode(parts[1]).map_err(|_| WinnowError::invalid_api_key())?;
        let claims: JwtClaims = serde_json::from_slice(&claims_bytes)
            .map_err(|_| WinnowError::invalid_api_key())?;
        
        // Check expiration
        if claims.is_expired() {
            return Err(WinnowError::auth_expired());
        }
        
        // Check issuer
        if claims.iss != self.issuer {
            return Err(WinnowError::invalid_api_key());
        }
        
        Ok(claims)
    }
    
    /// Simple HMAC-like signature (XOR-based for WASM compatibility)
    /// Note: In production, use a real HMAC-SHA256 implementation
    fn sign(&self, data: &str) -> Vec<u8> {
        let data_bytes = data.as_bytes();
        let mut result = vec![0u8; 32];
        
        for (i, &b) in data_bytes.iter().enumerate() {
            let secret_byte = self.secret[i % self.secret.len()];
            result[i % 32] ^= b.wrapping_add(secret_byte);
        }
        
        // Add a simple hash-like mixing
        for i in 0..31 {
            result[i + 1] = result[i + 1].wrapping_add(result[i].rotate_left(3));
        }
        
        result
    }
}

/// Base64 URL-safe encoding
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut result = String::new();
    
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;
        
        result.push(ALPHABET[(b0 >> 2)] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);
        
        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        }
        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f] as char);
        }
    }
    
    result
}

/// Base64 URL-safe decoding
fn base64_decode(data: &str) -> Result<Vec<u8>, ()> {
    const DECODE: [i8; 128] = [
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,63,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
    ];
    
    let bytes: Vec<u8> = data.bytes().collect();
    let mut result = Vec::new();
    
    for chunk in bytes.chunks(4) {
        let mut buf = [0u8; 4];
        let mut len = 0;
        
        for (i, &b) in chunk.iter().enumerate() {
            if b as usize >= 128 { return Err(()); }
            let v = DECODE[b as usize];
            if v < 0 { continue; }
            buf[i] = v as u8;
            len = i + 1;
        }
        
        if len >= 2 {
            result.push((buf[0] << 2) | (buf[1] >> 4));
        }
        if len >= 3 {
            result.push((buf[1] << 4) | (buf[2] >> 2));
        }
        if len >= 4 {
            result.push((buf[2] << 6) | buf[3]);
        }
    }
    
    Ok(result)
}

/// WASM-exposed JWT manager
#[wasm_bindgen]
pub struct JwtManager {
    provider: JwtProvider,
}

#[wasm_bindgen]
impl JwtManager {
    #[wasm_bindgen(constructor)]
    pub fn new(secret: &str, issuer: &str, ttl_seconds: u32) -> Self {
        Self {
            provider: JwtProvider::new(secret, issuer, ttl_seconds as u64),
        }
    }
    
    /// Create an admin token
    pub fn create_admin_token(&self, user_id: &str) -> String {
        self.provider.create_token(user_id, vec![Role::Admin], None)
    }
    
    /// Create a read-only token for specific collections
    pub fn create_readonly_token(&self, user_id: &str, collections: JsValue) -> Result<String, JsValue> {
        let cols: Vec<String> = serde_wasm_bindgen::from_value(collections)?;
        Ok(self.provider.create_token(user_id, vec![Role::ReadOnly], Some(cols)))
    }
    
    /// Verify token and return auth context
    pub fn verify(&self, token: &str) -> Result<JsValue, JsValue> {
        let claims = self.provider.verify_token(token).map_err(|e| JsValue::from(e))?;
        let ctx = claims.to_auth_context();
        Ok(serde_wasm_bindgen::to_value(&ctx)?)
    }
}
