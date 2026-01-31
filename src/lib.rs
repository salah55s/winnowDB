use wasm_bindgen::prelude::*;

pub mod test_utils;
pub mod storage;
pub mod vector;

pub use storage::*;
pub use vector::storage::VectorIndex;
pub use vector::flat::IndexFlat;
pub use vector::hnsw::IndexHNSW;
pub use vector::ivf::IndexIVF;
pub use vector::pq::IndexPQ;
pub use vector::ivf_pq::IndexIVFPQ;
pub use vector::sq::IndexSQ;
pub use vector::bq::IndexBQ;
pub use vector::metric::MetricType;
pub use vector::webgpu::IndexWebGPU;

pub mod collection;
pub use collection::WinnowCollection;
pub use collection::IndexType;

pub mod resilience;
pub mod observability;
pub mod security;
pub mod filter;
pub mod error;
pub mod tenancy;
pub mod query;
pub mod search;
pub mod compliance;
pub mod distributed;
pub mod tuning;

pub use error::{WinnowError, WinnowErrorKind};
pub use tenancy::MultiTenancyManager;
pub use query::QueryParser;
pub use search::AdvancedSearch;
pub use security::sso::SsoManager;
pub use compliance::ComplianceManager;
pub use distributed::DistributedManager;
pub use tuning::{AutoTuner, IndexComparator};

#[cfg(feature = "parallel")]
#[wasm_bindgen]
pub fn init_thread_pool(num_threads: usize) -> js_sys::Promise {
    wasm_bindgen_rayon::init_thread_pool(num_threads)
}

