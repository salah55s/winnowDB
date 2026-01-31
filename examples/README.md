# WinnowDB Examples

Simple, copy-paste examples to get started with WinnowDB.

## Quick Start

### Node.js

```bash
# Run the quickstart
node examples/node-quickstart/index.mjs
```

### Browser

```bash
# Serve files (any static server works)
python3 -m http.server 8000

# Open: http://localhost:8000/examples/01-basic/index.html
```

## Examples List

| Example | Description | Run |
|---------|-------------|-----|
| **node-quickstart** | Movies search with payloads | `node examples/node-quickstart/index.mjs` |
| **semantic-search** | Knowledge base semantic search | `node examples/semantic-search/index.mjs` |
| **benchmark** | Performance testing | `node examples/benchmark/index.mjs` |
| **01-basic** | Browser demo with buttons | Open `examples/01-basic/index.html` |

## Minimal Code

```javascript
import init, { WinnowCollection } from '@winnow-db/winnow-db';

await init();

const db = new WinnowCollection("test", {
    dim: 128,
    index_type: "HNSW",
    max_elements: 1000
}, null);

// Add
db.add(1, new Float32Array(128).fill(0.5), null, null, '{"hello": "world"}');

// Search
const results = db.search(new Float32Array(128).fill(0.5), 5, null, null);
console.log(results); // [[1, 0]]

db.free();
```

## With Real Embeddings

```javascript
// Using OpenAI
import OpenAI from 'openai';
const openai = new OpenAI();

async function embed(text) {
    const res = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: text
    });
    return new Float32Array(res.data[0].embedding);
}

// Or use HuggingFace Transformers.js (browser-compatible)
import { pipeline } from '@xenova/transformers';
const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

async function embed(text) {
    const result = await embedder(text, { pooling: 'mean', normalize: true });
    return new Float32Array(result.data);
}
```
