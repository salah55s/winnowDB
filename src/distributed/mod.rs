//! Distributed System Module
//! 
//! Provides foundations for Sharding, Replication, and Raft consensus.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// ============================================================================
// Cluster Topology
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeRole {
    Leader,
    Follower,
    Candidate, // For Raft
    Learner,   // Read-only / catch-up
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub address: String,
    pub role: NodeRole,
    pub state: NodeState,
    pub last_heartbeat: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeState {
    Online,
    Offline,
    Joining,
    Leaving,
}

// ============================================================================
// Sharding
// ============================================================================

/// Sharding strategy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ShardingStrategy {
    Hash,       // Consistent hashing
    Range,      // Range-based
    Directory,  // Lookup table
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Shard {
    pub id: u32,
    pub collection: String,
    pub replicas: Vec<String>, // Node IDs
    pub leader: Option<String>, // Node ID
    pub version: u64,
}

// ============================================================================
// Replication (Raft-lite)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogEntry {
    pub term: u64,
    pub index: u64,
    pub command: Vec<u8>, // Serialized Op
}

// ============================================================================
// Distributed Manager
// ============================================================================

#[wasm_bindgen]
pub struct DistributedManager {
    node_id: String,
    nodes: HashMap<String, Node>,
    shards: HashMap<u32, Shard>,
    current_term: u64,
}

#[wasm_bindgen]
impl DistributedManager {
    #[wasm_bindgen(constructor)]
    pub fn new(node_id: &str, address: &str) -> Self {
        let mut nodes = HashMap::new();
        nodes.insert(node_id.to_string(), Node {
            id: node_id.to_string(),
            address: address.to_string(),
            role: NodeRole::Leader, // Start as leader of self
            state: NodeState::Online,
            last_heartbeat: js_sys::Date::now(),
        });
        
        Self {
            node_id: node_id.to_string(),
            nodes,
            shards: HashMap::new(),
            current_term: 0,
        }
    }

    /// Register a peer node
    pub fn add_peer(&mut self, id: &str, address: &str) {
        self.nodes.insert(id.to_string(), Node {
            id: id.to_string(),
            address: address.to_string(),
            role: NodeRole::Follower,
            state: NodeState::Joining,
            last_heartbeat: js_sys::Date::now(),
        });
    }

    /// Create a shard for a collection
    pub fn create_shard(&mut self, shard_id: u32, collection: &str, replication_factor: usize) -> JsValue {
        // Simple default placement strategy
        let replicas: Vec<String> = self.nodes.keys()
            .take(replication_factor)
            .cloned()
            .collect();
            
        let shard = Shard {
            id: shard_id,
            collection: collection.to_string(),
            leader: replicas.first().cloned(),
            replicas,
            version: 1,
        };
        
        self.shards.insert(shard_id, shard.clone());
        serde_wasm_bindgen::to_value(&shard).unwrap_or(JsValue::NULL)
    }

    /// Get cluster state
    pub fn get_topology(&self) -> JsValue {
        let topology = serde_json::json!({
            "self": self.node_id,
            "nodes": self.nodes,
            "shards": self.shards,
            "term": self.current_term,
        });
        serde_wasm_bindgen::to_value(&topology).unwrap_or(JsValue::NULL)
    }
}
