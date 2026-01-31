use wasm_bindgen::prelude::*;
use web_sys::AbortSignal;
use serde::{Serialize, Deserialize};
use super::ivf_pq::IndexIVFPQ;
use super::sq::IndexSQ;
use super::bq::IndexBQ;
use super::hnsw::IndexHNSW;
use super::storage::VectorIndex; // Import the trait

// Define the Trait (Object Safe version, without Serde bounds for the trait itself)
// But wait, we want to Serialize the Enum.
// So let's keep the trait simple for "Capabilities".

use std::collections::HashSet;

pub trait VectorIndexTrait {
    fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue>;
    fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue>;
    fn train(&mut self, data: &[f32]) -> Result<(), JsValue>;
    fn delete(&mut self, id: u32) -> Result<(), JsValue>;
    fn clear(&mut self) -> Result<(), JsValue>;
}

// The Enum that holds concrete types
#[derive(Serialize, Deserialize, Clone)]
pub enum VectorIndexEnum {
    Hnsw(IndexHNSW),
    IvfPq(IndexIVFPQ),
    Sq(IndexSQ),
    Bq(IndexBQ),
}

impl VectorIndexTrait for VectorIndexEnum {
    fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), JsValue> {
        match self {
            VectorIndexEnum::Hnsw(idx) => idx.add(id, vector),
            VectorIndexEnum::IvfPq(idx) => idx.add(id, vector),
            VectorIndexEnum::Sq(idx) => idx.add(id, vector),
            VectorIndexEnum::Bq(idx) => idx.add(id, vector),
        }
    }

    fn search(&self, query: Vec<f32>, k: usize, filter: Option<&HashSet<u32>>, signal: Option<&AbortSignal>) -> Result<JsValue, JsValue> {
        match self {
            VectorIndexEnum::Hnsw(idx) => idx.search(query, k, filter, signal),
            VectorIndexEnum::IvfPq(idx) => idx.search(query, k, 20, filter, signal), // Pass nprobe default?
            VectorIndexEnum::Sq(idx) => idx.search(query, k, filter, signal),
            VectorIndexEnum::Bq(idx) => idx.search(query, k, filter, signal),
        }
    }

    fn train(&mut self, data: &[f32]) -> Result<(), JsValue> {
        match self {
            VectorIndexEnum::Hnsw(idx) => idx.train(data),
            VectorIndexEnum::IvfPq(idx) => idx.train(data),
            VectorIndexEnum::Sq(idx) => idx.train(data),
            VectorIndexEnum::Bq(idx) => idx.train(data),
        }
    }

    fn delete(&mut self, id: u32) -> Result<(), JsValue> {
        match self {
            VectorIndexEnum::Hnsw(idx) => idx.delete(id),
            VectorIndexEnum::IvfPq(idx) => idx.delete(id),
            VectorIndexEnum::Sq(idx) => idx.delete(id),
            VectorIndexEnum::Bq(idx) => idx.delete(id),
        }
    }

    fn clear(&mut self) -> Result<(), JsValue> {
        match self {
            VectorIndexEnum::Hnsw(idx) => idx.clear(),
            VectorIndexEnum::IvfPq(idx) => idx.clear(),
            VectorIndexEnum::Sq(idx) => idx.clear(),
            VectorIndexEnum::Bq(idx) => idx.clear(),
        }
    }
}
