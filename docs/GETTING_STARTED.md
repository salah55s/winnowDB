# Getting Started with WinnowDB

This guide will walk you through installing WinnowDB and running your first vector search.

## Prerequisites

- Node.js 16+ or any modern browser
- npm or yarn

## Installation

```bash
npm install winnow-db
```

Or with yarn:

```bash
yarn add winnow-db
```

## Your First Vector Search

### 1. Initialize the WASM Module

```javascript
import init, { WinnowCollection } from 'winnow-db';

// Initialize WASM (required once)
await init();
```

### 2. Create a Collection

```javascript
const config = {
  dim: 768,           // Vector dimension (must match your embeddings)
  index_type: 0,      // 0=HNSW, 1=IVFPQ, 2=SQ, 3=BQ
  m: 16,              // HNSW connections per node
  ef_construction: 200 // HNSW build quality
};

const db = new WinnowCollection("my_vectors", config);
```

### 3. Add Vectors

```javascript
// Add a single vector
const vector = new Float32Array(768).fill(0.1);
const payload = JSON.stringify({ title: "My Document", category: "tech" });

await db.add(
  1,        // ID (unique integer)
  vector,   // Vector (Float32Array)
  null,     // Sparse indices (optional)
  payload   // Metadata JSON (optional)
);

// Add multiple vectors
for (let i = 2; i <= 1000; i++) {
  const v = new Float32Array(768).map(() => Math.random());
  await db.add(i, v, null, JSON.stringify({ title: `Doc ${i}` }));
}
```

### 4. Search

```javascript
const query = new Float32Array(768).fill(0.1);
const results = await db.search(query, 10); // Top 10

console.log(results);
// Output: [[1, 0.0], [42, 0.15], [99, 0.23], ...]
// Format: [id, distance]
```

### 5. Search with Filters

```javascript
const filter = {
  "$and": [
    { "category": { "$eq": "tech" } },
    { "price": { "$lt": 100 } }
  ]
};

const results = await db.search(query, 10, filter);
```

### 6. Persist Your Data

```javascript
// Save to binary
const snapshot = db.snapshot();
localStorage.setItem('my_db', btoa(String.fromCharCode(...snapshot)));

// Load from binary
const data = Uint8Array.from(atob(localStorage.getItem('my_db')), c => c.charCodeAt(0));
const loaded = WinnowCollection.load_snapshot(data, null);
```

## Next Steps

- [API Reference](./API_REFERENCE.md) - All methods documented
- [Performance Tuning](./PERFORMANCE.md) - Optimize for your use case
- [Architecture](./ARCHITECTURE.md) - How it works under the hood
- [Examples](../examples/) - Working code samples

## Common Patterns

### Embedding Text with Transformers.js

```javascript
import { pipeline } from '@xenova/transformers';

const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

async function embed(text) {
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return new Float32Array(output.data);
}

const vector = await embed("Hello world");
await db.add(1, vector, null, JSON.stringify({ text: "Hello world" }));
```

### Using with React

```jsx
import { useEffect, useState } from 'react';
import init, { WinnowCollection } from 'winnow-db';

function useVectorDB() {
  const [db, setDb] = useState(null);
  
  useEffect(() => {
    (async () => {
      await init();
      const collection = new WinnowCollection("my_app", { dim: 384, index_type: 0 });
      setDb(collection);
    })();
  }, []);
  
  return db;
}
```

## Troubleshooting

### "Index not trained"

IVF-PQ indices require training before use:

```javascript
// Collect representative vectors first
const trainingData = new Float32Array(1000 * 768);
// Fill with your data...

await db.train(trainingData);
```

### "Dimension mismatch"

All vectors must match the `dim` in your config:

```javascript
// Wrong: dim=768 but vector is 512
const bad = new Float32Array(512);
await db.add(1, bad); // Error!

// Correct
const good = new Float32Array(768);
await db.add(1, good); // Works
```

### "Out of memory"

Reduce memory usage:

```javascript
// Use IVF-PQ instead of HNSW for large datasets
const config = { dim: 768, index_type: 1 };

// Or compact periodically
await db.compact();
```
