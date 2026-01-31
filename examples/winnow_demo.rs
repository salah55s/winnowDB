use winnow_db::{IndexHNSW, MetricType}; // Use underlying engine directly
use serde_json::json;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

/// Simulate realistic text embeddings (384 dimensions)
fn generate_document_embedding(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let seed = hasher.finish();
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect()
}

fn main() {
    println!("🚀 WinnowDB Real-World Demo (Native Mode)\n");
    println!("Note: Running via IndexHNSW directly to bypass WASM-browser bindings.\n");

    let documents = vec![
        (1, "Introduction to Machine Learning with Python"),
        (2, "Deep Learning Neural Networks Tutorial"),
        (3, "How to Build a REST API with Rust"),
        (4, "Understanding Vector Databases"),
        (5, "Python Data Science Handbook"),
        (6, "Rust Programming Language Guide"),
        (7, "Machine Learning Algorithms Explained"),
        (8, "Building Web Applications with React"),
        (9, "Advanced Python Programming Techniques"),
        (10, "Database Systems and Design Patterns"),
    ];
    
    // 1. CREATE INDEX & STORAGE
    println!("📚 Step 1: Creating collection 'tech_articles'...");
    
    // dim=384, m=16, ef_c=200, ef_s=100, Metric=L2
    let mut index = IndexHNSW::new(384, 16, 200, 100, MetricType::L2);
    let mut payloads: HashMap<u32, serde_json::Value> = HashMap::new();
    
    println!("✅ Collection created with HNSW index (384 dimensions)\n");
    
    // 2. INSERT DOCUMENTS
    println!("📝 Step 2: Inserting {} documents...", documents.len());
    for (id, text) in &documents {
        let embedding = generate_document_embedding(text, 384);
        let payload = json!({
            "title": text,
            "category": if text.contains("Python") { "python" } 
                       else if text.contains("Rust") { "rust" }
                       else if text.contains("Machine") || text.contains("Learning") { "ml" }
                       else { "other" }
        });
        
        // Add to Index (Core Engine)
        index.add(*id, embedding).expect("Failed to add vector");
        // Store Payload (Simulated)
        payloads.insert(*id, payload);
        
        println!("  ✓ Added: {}", text);
    }
    
    println!("\n✅ All documents indexed\n");
    
    // 3. SEMANTIC SEARCH
    println!("🔍 Step 3: Testing semantic search...\n");
    
    let queries = vec![
        ("machine learning python", "Query: Machine Learning + Python"),
        ("rust programming guide", "Query: Rust Programming"),
        ("database design", "Query: Database Design"),
    ];
    
    for (query_text, label) in queries {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("{}", label);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        let query_embedding = generate_document_embedding(query_text, 384);
        
        // Native Search
        let results = index.search_rust(query_embedding, 3, None);
        
        for (rank, (id, score)) in results.iter().enumerate() {
            let doc = documents.iter().find(|(doc_id, _)| doc_id == id).unwrap();
            println!("  {}. [Score: {:.4}] {}", rank + 1, score, doc.1);
        }
        println!();
    }
    
    // 4. FILTERED SEARCH
    println!("🎯 Step 4: Testing filtered search...\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Query: 'programming' (Python category only)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let query_embedding = generate_document_embedding("programming tutorial", 384);
    
    // Manual Filter Logic (Simulating Collection Flow)
    // 1. Get embedding Search Results (get more candidates)
    let candidates = index.search_rust(query_embedding.clone(), 10, None);
    
    let mut filtered_results = Vec::new();
    for (id, score) in candidates {
        if let Some(payload) = payloads.get(&id) {
            if payload["category"] == "python" {
                filtered_results.push((id, score));
            }
        }
        if filtered_results.len() >= 5 { break; }
    }
    
    for (rank, (id, score)) in filtered_results.iter().enumerate() {
        let doc_title = payloads.get(id).unwrap()["title"].as_str().unwrap();
        println!("  {}. [Score: {:.4}] {}", rank + 1, score, doc_title);
    }
    println!();
    
    // 5. PERFORMANCE METRICS
    println!("📊 Step 5: Performance metrics...\n");
    
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let query = generate_document_embedding("test query", 384);
        let _ = index.search_rust(query, 5, None);
    }
    let elapsed = start.elapsed();
    let qps = 100.0 / elapsed.as_secs_f64();
    
    println!("  • 100 searches completed in: {:.2}s", elapsed.as_secs_f64());
    println!("  • Queries per second (QPS): {:.0}", qps);
    println!("  • Average latency: {:.2}ms", elapsed.as_millis() as f64 / 100.0);
    
    // Health check metrics
    let mem = documents.len() * 384 * 4; // Approx vector data
    println!("  • Memory usage (Vectors): ~{} bytes", mem);
    println!("  • Vector count: {}", documents.len());
    println!("  • Status: ok");
    
    println!("\n✨ Demo completed successfully!");
}
