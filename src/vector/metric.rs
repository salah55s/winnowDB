use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[wasm_bindgen]
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub enum MetricType {
    L2 = 0,
    InnerProduct = 1,
    Cosine = 2,
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return crate::vector::simd::l2_distance_wasm(a, b);
    }

    let len = a.len();
    if len != b.len() { return f32::MAX; }
    
    let mut sum = 0.0;
    
    // Explicit unrolling for auto-vectorization hint
    let mut i = 0;
    while i + 4 <= len {
        let d0 = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        let d1 = unsafe { *a.get_unchecked(i+1) - *b.get_unchecked(i+1) };
        let d2 = unsafe { *a.get_unchecked(i+2) - *b.get_unchecked(i+2) };
        let d3 = unsafe { *a.get_unchecked(i+3) - *b.get_unchecked(i+3) };
        
        sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        i += 4;
    }
    
    while i < len {
        let d = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        sum += d * d;
        i += 1;
    }
    
    sum.sqrt()
}

pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return crate::vector::simd::dot_product_wasm(a, b);
    }

    let len = a.len();
    if len != b.len() { return 0.0; }
    let mut sum = 0.0;
    for i in 0..len {
        sum += unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) };
    }
    sum
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return crate::vector::simd::cosine_similarity_wasm(a, b);
    }

    let len = a.len();
    if len != b.len() { return 0.0; }
    
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for i in 0..len {
        let v1 = unsafe { *a.get_unchecked(i) };
        let v2 = unsafe { *b.get_unchecked(i) };
        dot += v1 * v2;
        norm_a += v1 * v1;
        norm_b += v2 * v2;
    }
    
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
