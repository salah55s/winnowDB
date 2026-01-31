use winnow_db::{IndexFlat, IndexHNSW, MetricType};
use std::time::Instant;
use rand::Rng;

fn generate_vectors(count: usize, dim: usize) -> Vec<(u32, Vec<f32>)> {
    let mut rng = rand::thread_rng();
    (0..count).map(|i| {
        let vector: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
        (i as u32, vector)
    }).collect()
}

#[test]
fn benchmark_and_compare_indices() {
    let vector_count = 1000;
    let dim = 128;
    let k = 10;
    
    println!("Generating {} vectors of dim {}...", vector_count, dim);
    let dataset = generate_vectors(vector_count, dim);
    let query_vector = dataset[0].1.clone(); 

    // 1. Benchmark Flat Index (Ground Truth)
    println!("\n--- Benchmarking Flat Index ---");
    let start = Instant::now();
    let mut flat_index = IndexFlat::new(dim, MetricType::L2); // dim, metric
    for (id, vec) in &dataset {
        flat_index.add(*id, vec.clone()).unwrap(); // unwrap Result<(), JsValue>
    }
    let build_time = start.elapsed();
    println!("Flat Build Time: {:?}", build_time);

    let start = Instant::now();
    let flat_results = flat_index.search_rust(&query_vector, k);
    let flat_results = flat_index.search_rust(&query_vector, k, None);
    let flat_latency = start.elapsed();
    println!("Flat Search Latency: {:?}", flat_latency);
    println!("Flat Top Result Score: {:?}", flat_results[0].1);

    // 2. Benchmark HNSW Index (Approximate)
    println!("\n--- Benchmarking HNSW Index ---");
    let start = Instant::now();
    // new(dim, m, ef_construction, ef_search, metric)
    // Using higher M and ef for random data recall
    let mut hnsw_index = IndexHNSW::new(dim, 24, 200, 100, MetricType::L2);
    for (id, vec) in &dataset {
        hnsw_index.add(*id, vec.clone()).unwrap();
    }
    let build_time = start.elapsed();
    println!("HNSW Build Time: {:?}", build_time);

    let start = Instant::now();
    let hnsw_results = hnsw_index.search_rust(query_vector.clone(), k, None);
    let hnsw_latency = start.elapsed();
    println!("HNSW Search Latency: {:?}", hnsw_latency);
    println!("HNSW Top Result Score: {:?}", hnsw_results[0].1);

    // 3. Calculate Recall
    let flat_ids: Vec<u32> = flat_results.iter().map(|r| r.0).collect();
    let hnsw_ids: Vec<u32> = hnsw_results.iter().map(|r| r.0).collect();
    
    let matches = hnsw_ids.iter().filter(|id| flat_ids.contains(id)).count();
    let recall = matches as f32 / k as f32;
    
    println!("\n--- Comparison Results ---");
    println!("Recall: {:.2}%", recall * 100.0);
    println!("Note: On small random datasets (N=1000), Flat is often faster. HNSW shines at N > 100k.");
    
    assert!(recall >= 0.8, "HNSW recall should be acceptable (>80%)");
    println!("✅ Real Benchmark Test Passed!");
}
