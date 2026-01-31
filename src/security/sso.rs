//! Enterprise SSO Support for WinnowDB
//! 
//! Handles SAML 2.0, OpenID Connect (OIDC-OAuth2), and LDAP configurations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// ============================================================================
// SSO Provider Types
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SsoType {
    Saml,
    Oidc,
    Ldap,
}

/// SAML 2.0 Configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamlConfig {
    pub idp_metadata_url: String,
    pub sp_entity_id: String,
    pub acs_url: String,
    pub x509_cert: String,
    pub attribute_mapping: HashMap<String, String>, // e.g., "email" -> "urn:oid:..."
}

/// OpenID Connect (OAuth2) Configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OidcConfig {
    pub issuer_url: String,
    pub client_id: String,
    pub client_secret: String,
    pub scopes: Vec<String>,
    pub callback_url: String,
}

/// LDAP Configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LdapConfig {
    pub url: String,
    pub base_dn: String,
    pub bind_dn: Option<String>,
    pub bind_password: Option<String>,
    pub user_filter: String, // e.g., "(uid={})"
    pub group_search_base: Option<String>,
}

/// Generic SSO Provider Configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SsoProviderConfig {
    pub name: String,
    pub provider_type: SsoType,
    pub saml: Option<SamlConfig>,
    pub oidc: Option<OidcConfig>,
    pub ldap: Option<LdapConfig>,
    pub enabled: bool,
    pub auto_create_users: bool,
    pub default_role: String,
}

// ============================================================================
// SSO Identity
// ============================================================================

/// Identity attributes resolved from SSO
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SsoIdentity {
    pub provider_name: String,
    pub user_id: String,
    pub email: Option<String>,
    pub full_name: Option<String>,
    pub groups: Vec<String>,
    pub attributes: HashMap<String, String>,
}

// ============================================================================
// SSO Manager
// ============================================================================

#[wasm_bindgen]
pub struct SsoManager {
    providers: HashMap<String, SsoProviderConfig>,
}

#[wasm_bindgen]
impl SsoManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Add a SAML provider
    pub fn add_saml_provider(
        &mut self, 
        name: &str, 
        idp_metadata_url: &str, 
        sp_entity_id: &str, 
        acs_url: &str,
        cert: &str
    ) {
        let config = SsoProviderConfig {
            name: name.to_string(),
            provider_type: SsoType::Saml,
            saml: Some(SamlConfig {
                idp_metadata_url: idp_metadata_url.to_string(),
                sp_entity_id: sp_entity_id.to_string(),
                acs_url: acs_url.to_string(),
                x509_cert: cert.to_string(),
                attribute_mapping: HashMap::new(),
            }),
            oidc: None,
            ldap: None,
            enabled: true,
            auto_create_users: true,
            default_role: "read_only".to_string(),
        };
        self.providers.insert(name.to_string(), config);
    }

    /// Add an OIDC provider
    pub fn add_oidc_provider(
        &mut self,
        name: &str,
        issuer: &str,
        client_id: &str,
        client_secret: &str,
        callback_url: &str
    ) {
        let config = SsoProviderConfig {
            name: name.to_string(),
            provider_type: SsoType::Oidc,
            saml: None,
            oidc: Some(OidcConfig {
                issuer_url: issuer.to_string(),
                client_id: client_id.to_string(),
                client_secret: client_secret.to_string(),
                scopes: vec!["openid".to_string(), "profile".to_string(), "email".to_string()],
                callback_url: callback_url.to_string(),
            }),
            ldap: None,
            enabled: true,
            auto_create_users: true,
            default_role: "read_only".to_string(),
        };
        self.providers.insert(name.to_string(), config);
    }

    /// Add an LDAP provider
    pub fn add_ldap_provider(
        &mut self,
        name: &str,
        url: &str,
        base_dn: &str
    ) {
        let config = SsoProviderConfig {
            name: name.to_string(),
            provider_type: SsoType::Ldap,
            saml: None,
            oidc: None,
            ldap: Some(LdapConfig {
                url: url.to_string(),
                base_dn: base_dn.to_string(),
                bind_dn: None,
                bind_password: None,
                user_filter: "(uid={})".to_string(),
                group_search_base: None,
            }),
            enabled: true,
            auto_create_users: false, // LDAP usually implies existing directory
            default_role: "read_only".to_string(),
        };
        self.providers.insert(name.to_string(), config);
    }

    /// Get provider configuration (as JSON)
    pub fn get_provider_config(&self, name: &str) -> JsValue {
        if let Some(config) = self.providers.get(name) {
            serde_wasm_bindgen::to_value(config).unwrap_or(JsValue::NULL)
        } else {
            JsValue::NULL
        }
    }
    
    /// List all providers
    pub fn list_providers(&self) -> JsValue {
        let values: Vec<&SsoProviderConfig> = self.providers.values().collect();
        serde_wasm_bindgen::to_value(&values).unwrap_or(JsValue::NULL)
    }
}

impl Default for SsoManager {
    fn default() -> Self {
        Self::new()
    }
}
