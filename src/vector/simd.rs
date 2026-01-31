

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::*;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn l2_distance_wasm(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len(); // Assume equal length checked by caller
    let mut sum_v = unsafe { f32x4_splat(0.0) };
    let mut i = 0;

    // Process 4 floats at a time
    while i + 4 <= len {
        unsafe {
            let v1 = v128_load(a.as_ptr().add(i) as *const v128);
            let v2 = v128_load(b.as_ptr().add(i) as *const v128);
            let diff = f32x4_sub(v1, v2);
            let sq = f32x4_mul(diff, diff);
            sum_v = f32x4_add(sum_v, sq);
        }
        i += 4;
    }

    // Horizontal sum
    let mut sum = unsafe {
        // Extract lanes
        // Note: f32x4_extract_lane requires const index
        // Or simpler: use store to array?
        // wasm_radr_f32x4/etc might exist but extract is safer fallback
        f32x4_extract_lane::<0>(sum_v) +
        f32x4_extract_lane::<1>(sum_v) +
        f32x4_extract_lane::<2>(sum_v) +
        f32x4_extract_lane::<3>(sum_v)
    };

    // Remaining elements
    while i < len {
        let d = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        sum += d * d;
        i += 1;
    }

    sum.sqrt()
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn dot_product_wasm(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum_v = unsafe { f32x4_splat(0.0) };
    let mut i = 0;

    while i + 4 <= len {
        unsafe {
            let v1 = v128_load(a.as_ptr().add(i) as *const v128);
            let v2 = v128_load(b.as_ptr().add(i) as *const v128);
            sum_v = f32x4_add(sum_v, f32x4_mul(v1, v2)); 
        }
        i += 4;
    }

    let mut sum = unsafe {
        f32x4_extract_lane::<0>(sum_v) +
        f32x4_extract_lane::<1>(sum_v) +
        f32x4_extract_lane::<2>(sum_v) +
        f32x4_extract_lane::<3>(sum_v)
    };

    while i < len {
        sum += unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) };
        i += 1;
    }

    sum
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn cosine_similarity_wasm(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    // Calculate Dot, NormA, NormB in one pass
    
    let mut dot_v = unsafe { f32x4_splat(0.0) };
    let mut norm_a_v = unsafe { f32x4_splat(0.0) };
    let mut norm_b_v = unsafe { f32x4_splat(0.0) };
    let mut i = 0;

    while i + 4 <= len {
        unsafe {
            let v1 = v128_load(a.as_ptr().add(i) as *const v128);
            let v2 = v128_load(b.as_ptr().add(i) as *const v128);
            
            dot_v = f32x4_add(dot_v, f32x4_mul(v1, v2));
            norm_a_v = f32x4_add(norm_a_v, f32x4_mul(v1, v1));
            norm_b_v = f32x4_add(norm_b_v, f32x4_mul(v2, v2));
        }
        i += 4;
    }
    
    let mut dot = unsafe { 
        f32x4_extract_lane::<0>(dot_v) + f32x4_extract_lane::<1>(dot_v) +
        f32x4_extract_lane::<2>(dot_v) + f32x4_extract_lane::<3>(dot_v) 
    };
    let mut norm_a = unsafe {
        f32x4_extract_lane::<0>(norm_a_v) + f32x4_extract_lane::<1>(norm_a_v) +
        f32x4_extract_lane::<2>(norm_a_v) + f32x4_extract_lane::<3>(norm_a_v) 
    };
    let mut norm_b = unsafe {
        f32x4_extract_lane::<0>(norm_b_v) + f32x4_extract_lane::<1>(norm_b_v) +
        f32x4_extract_lane::<2>(norm_b_v) + f32x4_extract_lane::<3>(norm_b_v) 
    };
    
    while i < len {
        unsafe {
            let v1 = *a.get_unchecked(i);
            let v2 = *b.get_unchecked(i);
            dot += v1 * v2;
            norm_a += v1 * v1;
            norm_b += v2 * v2;
        }
        i += 1;
    }
    
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
