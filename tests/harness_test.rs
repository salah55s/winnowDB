use winnow_db::test_utils::TestHarness;
use winnow_db::vector::hnsw::IndexHNSW;
use winnow_db::vector::metric::MetricType;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn test_hnsw_recall() {
    let dim = 16;
    let n_vectors = 1000;
    let n_queries = 10;
    
    // Create Harness
    let harness = TestHarness::new(dim, n_vectors, n_queries, MetricType::L2, |d, m| {
        // HNSW Config: M=16, ef_c=200, ef_s=50
        IndexHNSW::new(d, 16, 200, 50, m)
    });

    // Measure Recall@10
    let recall = harness.measure_recall(10);
    println!("HNSW Recall@10: {}", recall);
    
    // Expect reasonable recall (> 0.9) for HNSW on random data
    assert!(recall > 0.9, "Recall should be high");
    
    // Measure QPS
    let qps = harness.measure_qps(10, 1000.0); // Run for 1s
    println!("HNSW QPS: {}", qps);
    assert!(qps > 0.0);
}
