# WinnowDB Node.js Quickstart

A complete example showing how to use WinnowDB in Node.js.

## Features Demonstrated

- Collection creation with HNSW index
- Vector insertion with payloads
- Similarity search
- Filtered search
- Payload updates
- Snapshots for persistence
- Metrics monitoring

## Run

```bash
cd examples/node-quickstart
node index.mjs
```

## Expected Output

```
🚀 WinnowDB Quickstart

📦 Collection created: movies

📥 Inserting movies...
   Added 10 movies

🔍 Searching for Sci-Fi movies...
   Top 5 results:
   - Inception (2010) - Distance: 0.1234
   - The Matrix (1999) - Distance: 0.2345
   ...

🏷️  Filtering by Drama genre...
   ...

💾 Creating snapshot...
   Snapshot size: 15.2 KB

✅ Done!
```

## Code Walkthrough

### 1. Initialize WASM

```javascript
const wasmBuffer = await readFile('pkg/winnow_db_bg.wasm');
const wasm = await import('./pkg/winnow_db.js');
await wasm.default(wasmBuffer);
```

### 2. Create Collection

```javascript
const collection = new WinnowCollection("movies", {
    dim: 128,
    index_type: "HNSW",
    m: 16,
    ef_construction: 100,
    max_elements: 10000
}, null);
```

### 3. Add Vectors

```javascript
collection.add(id, embedding, null, null, JSON.stringify(payload));
```

### 4. Search

```javascript
const results = collection.search(queryVector, 5, filter, null);
```

### 5. Cleanup

```javascript
collection.free();
```
