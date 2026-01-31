#!/usr/bin/env node
/**
 * WinnowDB Node.js Quickstart Example
 * 
 * Run with: node index.mjs
 */

import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

async function main() {
    console.log('🚀 WinnowDB Quickstart\n');

    // Load WASM module
    const wasmPath = join(__dirname, '..', '..', 'pkg', 'winnow_db_bg.wasm');
    const wasmBuffer = await readFile(wasmPath);
    const wasm = await import('../../pkg/winnow_db.js');
    await wasm.default(wasmBuffer);

    const { WinnowCollection } = wasm;

    // Create a collection
    const collection = new WinnowCollection("movies", {
        dim: 128,                    // Vector dimension
        index_type: "HNSW",          // Hierarchical Navigable Small World
        m: 16,                       // HNSW parameter
        ef_construction: 100,        // Build quality
        max_elements: 10000          // Max vectors
    }, null);

    console.log('📦 Collection created: movies\n');

    // Sample movie data (in production, use embeddings from an ML model)
    const movies = [
        { id: 1, title: "The Matrix", genre: "Sci-Fi", year: 1999 },
        { id: 2, title: "Inception", genre: "Sci-Fi", year: 2010 },
        { id: 3, title: "The Dark Knight", genre: "Action", year: 2008 },
        { id: 4, title: "Pulp Fiction", genre: "Crime", year: 1994 },
        { id: 5, title: "Forrest Gump", genre: "Drama", year: 1994 },
        { id: 6, title: "The Shawshank Redemption", genre: "Drama", year: 1994 },
        { id: 7, title: "Interstellar", genre: "Sci-Fi", year: 2014 },
        { id: 8, title: "Fight Club", genre: "Drama", year: 1999 },
        { id: 9, title: "The Godfather", genre: "Crime", year: 1972 },
        { id: 10, title: "Gladiator", genre: "Action", year: 2000 }
    ];

    // Helper to safely get and parse payload
    function getPayload(id) {
        let payload = collection.get_payload(id);
        if (typeof payload === 'string') {
            try { payload = JSON.parse(payload); } catch (e) { }
        }
        return payload || {};
    }

    // Generate pseudo-embeddings (in production, use a real embedding model)
    function generateEmbedding(movie) {
        const vec = new Float32Array(128);
        // Simple deterministic vectors based on movie properties
        const genreHash = { "Sci-Fi": 0.8, "Action": 0.6, "Crime": 0.4, "Drama": 0.2 };
        const yearNorm = (movie.year - 1970) / 50;

        for (let i = 0; i < 128; i++) {
            vec[i] = (genreHash[movie.genre] || 0.5) * Math.sin(i * 0.1) +
                yearNorm * Math.cos(i * 0.1) +
                (movie.id / 10) * 0.1;
        }
        return vec;
    }

    // Insert movies
    console.log('📥 Inserting movies...');
    for (const movie of movies) {
        const embedding = generateEmbedding(movie);
        collection.add(movie.id, embedding, null, null, JSON.stringify(movie));
    }
    console.log(`   Added ${collection.count()} movies\n`);

    // Search for similar movies
    console.log('🔍 Searching for Sci-Fi movies...');
    const queryMovie = { id: 0, title: "Query", genre: "Sci-Fi", year: 2005 };
    const queryVector = generateEmbedding(queryMovie);

    const results = collection.search(queryVector, 5, null, null);

    console.log('   Top 5 results:');
    for (const [id, distance] of results) {
        const payload = getPayload(id);
        console.log(`   - ${payload.title} (${payload.year}) - Distance: ${distance.toFixed(4)}`);
    }

    // Filter by genre
    console.log('\n🏷️  Filtering by Drama genre...');
    const filter = { genre: "Drama" };
    const filteredResults = collection.search(queryVector, 5, filter, null);

    console.log('   Drama movies:');
    for (const [id, distance] of filteredResults) {
        const payload = getPayload(id);
        console.log(`   - ${payload.title} (${payload.year})`);
    }

    // Update a movie's payload
    console.log('\n✏️  Updating payload...');
    collection.update_payload(1, JSON.stringify({
        ...movies[0],
        rating: 8.7
    }));
    const updated = getPayload(1);
    console.log(`   Updated The Matrix with rating: ${updated.rating}`);

    // Create a snapshot
    console.log('\n💾 Creating snapshot...');
    const snapshot = collection.snapshot();
    console.log(`   Snapshot size: ${(snapshot.length / 1024).toFixed(1)} KB`);

    // Metrics
    console.log('\n📊 Metrics:');
    const metrics = collection.get_metrics();
    console.log(metrics.split('\n').slice(0, 5).map(l => `   ${l}`).join('\n'));

    // Cleanup
    collection.free();

    console.log('\n✅ Done!');
}

main().catch(console.error);
