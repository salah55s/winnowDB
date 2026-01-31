#!/usr/bin/env node
/**
 * WinnowDB Semantic Search Example
 * 
 * Demonstrates a simple semantic text search using TF-IDF-like embeddings.
 * In production, use a real embedding model like OpenAI, Cohere, or Hugging Face.
 */

import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Simple bag-of-words embedding (for demonstration)
function createVocabulary(documents) {
    const vocab = new Map();
    let idx = 0;
    for (const doc of documents) {
        const words = doc.text.toLowerCase().split(/\W+/).filter(w => w.length > 2);
        for (const word of words) {
            if (!vocab.has(word)) {
                vocab.set(word, idx++);
            }
        }
    }
    return vocab;
}

function textToEmbedding(text, vocab, dim = 128) {
    const vec = new Float32Array(dim).fill(0);
    const words = text.toLowerCase().split(/\W+/).filter(w => w.length > 2);

    for (const word of words) {
        const idx = vocab.get(word);
        if (idx !== undefined) {
            // Use word index modulo dimension for simple hashing
            vec[idx % dim] += 1;
        }
    }

    // Normalize
    let norm = 0;
    for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    if (norm > 0) {
        for (let i = 0; i < dim; i++) vec[i] /= norm;
    }

    return vec;
}

async function main() {
    console.log('🔍 WinnowDB Semantic Search Example\n');

    // Sample knowledge base
    const documents = [
        { id: 1, title: "Getting Started with JavaScript", text: "JavaScript is a programming language for web development. It runs in browsers and Node.js.", category: "programming" },
        { id: 2, title: "Python for Data Science", text: "Python is popular for machine learning, data analysis, and scientific computing. Libraries like NumPy and Pandas are essential.", category: "programming" },
        { id: 3, title: "Introduction to Rust", text: "Rust is a systems programming language focused on safety and performance. It prevents memory errors at compile time.", category: "programming" },
        { id: 4, title: "Web Development Basics", text: "HTML, CSS, and JavaScript are the foundation of web development. HTML structures content, CSS styles it, and JavaScript adds interactivity.", category: "web" },
        { id: 5, title: "Database Design Principles", text: "Good database design includes normalization, indexing, and query optimization. SQL databases use tables and relationships.", category: "database" },
        { id: 6, title: "Vector Databases Explained", text: "Vector databases store embeddings for semantic search. They use algorithms like HNSW for fast similarity search.", category: "database" },
        { id: 7, title: "Machine Learning Fundamentals", text: "Machine learning trains models on data to make predictions. Neural networks, decision trees, and SVMs are common algorithms.", category: "ml" },
        { id: 8, title: "Natural Language Processing", text: "NLP enables computers to understand human language. Techniques include tokenization, embeddings, and transformers.", category: "ml" },
        { id: 9, title: "Cloud Computing Overview", text: "Cloud computing provides on-demand computing resources. AWS, Azure, and GCP are major providers.", category: "infra" },
        { id: 10, title: "Kubernetes Basics", text: "Kubernetes orchestrates containerized applications. It handles scaling, networking, and deployment automation.", category: "infra" }
    ];

    // Build vocabulary
    const vocab = createVocabulary(documents);
    console.log(`📚 Vocabulary size: ${vocab.size} words\n`);

    // Load WinnowDB
    const wasmPath = join(__dirname, '..', '..', 'pkg', 'winnow_db_bg.wasm');
    const wasmBuffer = await readFile(wasmPath);
    const wasm = await import('../../pkg/winnow_db.js');
    await wasm.default(wasmBuffer);

    const { WinnowCollection } = wasm;
    const DIM = 128;

    // Create collection
    const collection = new WinnowCollection("knowledge_base", {
        dim: DIM,
        index_type: "HNSW",
        m: 16,
        ef_construction: 100,
        max_elements: 1000
    }, null);

    // Helper for payloads
    function getPayload(id) {
        let p = collection.get_payload(id);
        if (typeof p === 'string') try { p = JSON.parse(p); } catch (e) { }
        return p || {};
    }

    // Index documents
    console.log('📥 Indexing documents...');
    for (const doc of documents) {
        const embedding = textToEmbedding(doc.text, vocab, DIM);
        collection.add(doc.id, embedding, null, null, JSON.stringify(doc));
    }
    console.log(`   Indexed ${collection.count()} documents\n`);

    // === Demo queries ===

    async function search(query, k = 5, filter = null) {
        const queryVec = textToEmbedding(query, vocab, DIM);
        const results = collection.search(queryVec, k, filter, null);
        return results.map(([id, score]) => ({
            ...getPayload(id),
            score: score.toFixed(4)
        }));
    }

    // Query 1: Programming languages
    console.log('🔍 Query: "Which programming language is best for web?"');
    let results = await search("programming language web development javascript");
    for (const r of results.slice(0, 3)) {
        console.log(`   [${r.category}] ${r.title} (score: ${r.score})`);
    }

    // Query 2: Machine learning
    console.log('\n🔍 Query: "How do I train machine learning models?"');
    results = await search("machine learning training models neural networks");
    for (const r of results.slice(0, 3)) {
        console.log(`   [${r.category}] ${r.title} (score: ${r.score})`);
    }

    // Query 3: With filter
    console.log('\n🔍 Query: "database" (filtered to database category)');
    results = await search("database storage query", 3, { category: "database" });
    for (const r of results) {
        console.log(`   [${r.category}] ${r.title}`);
    }

    // Query 4: Semantic similarity
    console.log('\n🔍 Query: "vector similarity search algorithms"');
    results = await search("vector similarity search algorithms embeddings");
    for (const r of results.slice(0, 3)) {
        console.log(`   [${r.category}] ${r.title} (score: ${r.score})`);
    }

    collection.free();
    console.log('\n✅ Done!');
}

main().catch(console.error);
