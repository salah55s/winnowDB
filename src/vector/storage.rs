use wasm_bindgen::prelude::*;
use web_sys::AbortSignal;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use bincode;

use std::collections::HashSet;

pub trait VectorIndex: Serialize + DeserializeOwned {
    fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue>;
    fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue>;

    fn to_bytes(&self) -> Result<Vec<u8>, JsValue> {
        bincode::serialize(self).map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    fn from_bytes(data: &[u8]) -> Result<Self, JsValue> where Self: Sized {
        bincode::deserialize(data).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VectorStorage {
    vectors: std::collections::HashMap<u32, Vec<f32>>,
}

impl VectorStorage {
    pub fn new() -> Self {
        Self { vectors: std::collections::HashMap::new() }
    }

    pub fn add(&mut self, id: u32, vector: Vec<f32>) {
        self.vectors.insert(id, vector);
    }

    pub fn remove(&mut self, id: u32) {
        self.vectors.remove(&id);
    }

    pub fn get(&self, id: u32) -> Option<&Vec<f32>> {
        self.vectors.get(&id)
    }

    pub fn clear(&mut self) {
        self.vectors.clear();
    }
}

