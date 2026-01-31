# Semantic Search Example

A practical example showing how to build a knowledge base search with WinnowDB.

## Overview

This example demonstrates:

- Building a simple text embedding function (TF-IDF-style)
- Indexing documents with metadata
- Semantic search queries
- Filtered search by category

## Run

```bash
cd examples/semantic-search
node index.mjs
```

## In Production

Replace the simple `textToEmbedding` function with a real embedding model:

```javascript
// Using OpenAI
import OpenAI from 'openai';
const openai = new OpenAI();

async function getEmbedding(text) {
    const response = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: text
    });
    return new Float32Array(response.data[0].embedding);
}

// Using HuggingFace Transformers.js
import { pipeline } from '@xenova/transformers';
const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

async function getEmbedding(text) {
    const result = await embedder(text, { pooling: 'mean', normalize: true });
    return new Float32Array(result.data);
}
```
