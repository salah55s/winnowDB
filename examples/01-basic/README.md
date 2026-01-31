# Basic Example

Simple vector insert and search demo.

## Run

```bash
cd examples/01-basic
npx serve .
```

Then open <http://localhost:3000>

## Features

- Initialize WinnowDB with HSNW index
- Add 1000 random vectors
- Search for nearest neighbors
- Display latency metrics

## Code Highlights

```javascript
// Create collection
const db = new WinnowCollection("basic_example", {
  dim: 128,
  index_type: 0, // HNSW
  m: 16,
  ef_construction: 100
});

// Add vector
await db.add(id, vector, null, null, payload);

// Search
const results = await db.search(query, 10);
// Returns: [[id, distance], ...]
```
