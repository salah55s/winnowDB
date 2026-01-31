#!/usr/bin/env node
/**
 * WinnowDB Performance Benchmark
 * 
 * Tests insert and search performance at various scales.
 */

import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

async function main() {
    console.log('⚡ WinnowDB Performance Benchmark\n');
    console.log('═'.repeat(60));

    // Load WASM
    const wasmPath = join(__dirname, '..', '..', 'pkg', 'winnow_db_bg.wasm');
    const wasmBuffer = await readFile(wasmPath);
    const wasm = await import('../../pkg/winnow_db.js');
    await wasm.default(wasmBuffer);

    const { WinnowCollection } = wasm;

    // Benchmark configurations
    const benchmarks = [
        { vectors: 1000, dim: 128, searches: 1000 },
        { vectors: 5000, dim: 128, searches: 500 },
        { vectors: 10000, dim: 128, searches: 500 },
        { vectors: 10000, dim: 256, searches: 500 },
        { vectors: 20000, dim: 128, searches: 200 },
    ];

    console.log('| Vectors  | Dim  | Insert (vec/s) | Search (qps) | Latency (ms) |');
    console.log('|----------|------|----------------|--------------|--------------|');

    for (const config of benchmarks) {
        const { vectors, dim, searches } = config;

        // Create collection
        const collection = new WinnowCollection("benchmark", {
            dim,
            index_type: "HNSW",
            m: 16,
            ef_construction: 100,
            max_elements: vectors + 100
        }, null);

        // Insert benchmark
        const insertStart = performance.now();
        for (let i = 0; i < vectors; i++) {
            const vec = new Float32Array(dim);
            for (let j = 0; j < dim; j++) {
                vec[j] = Math.random();
            }
            collection.add(i, vec, null, null, null);
        }
        const insertTime = performance.now() - insertStart;
        const insertRate = (vectors / insertTime * 1000).toFixed(0);

        // Search benchmark
        let totalSearchTime = 0;
        for (let i = 0; i < searches; i++) {
            const query = new Float32Array(dim);
            for (let j = 0; j < dim; j++) {
                query[j] = Math.random();
            }
            const start = performance.now();
            collection.search(query, 10, null, null);
            totalSearchTime += performance.now() - start;
        }
        const searchRate = (searches / totalSearchTime * 1000).toFixed(0);
        const avgLatency = (totalSearchTime / searches).toFixed(3);

        console.log(`| ${vectors.toString().padStart(8)} | ${dim.toString().padStart(4)} | ${insertRate.padStart(14)} | ${searchRate.padStart(12)} | ${avgLatency.padStart(12)} |`);

        collection.free();
    }

    console.log('═'.repeat(60));

    // Memory test
    console.log('\n📊 Memory Efficiency Test');
    console.log('-'.repeat(40));

    const memCollection = new WinnowCollection("memory_test", {
        dim: 128,
        index_type: "HNSW",
        m: 16,
        ef_construction: 100,
        max_elements: 11000
    }, null);

    for (let i = 0; i < 10000; i++) {
        const vec = new Float32Array(128);
        for (let j = 0; j < 128; j++) vec[j] = Math.random();
        memCollection.add(i, vec, null, null, JSON.stringify({ id: i, data: "x".repeat(50) }));
    }

    const snapshot = memCollection.snapshot();
    console.log(`Vectors: 10,000 × 128 dims`);
    console.log(`Payloads: ~60 bytes each`);
    console.log(`Snapshot size: ${(snapshot.length / 1024 / 1024).toFixed(2)} MB`);
    console.log(`Per-vector: ${(snapshot.length / 10000).toFixed(0)} bytes`);

    memCollection.free();

    console.log('\n✅ Benchmark complete!');
}

main().catch(console.error);
