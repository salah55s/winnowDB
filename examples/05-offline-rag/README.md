# Offline RAG with WinnowDB

Run a complete Retrieval-Augmented Generation system 100% offline in the browser.

## Architecture

```
┌─────────────┐     ┌────────────┐     ┌─────────────┐
│  Documents  │ ──► │  WinnowDB  │ ──► │  LLM (Local)│
│  (Chunks)   │     │  (Vectors) │     │  (Answers)  │
└─────────────┘     └────────────┘     └─────────────┘
```

## Stack

- **Embeddings**: [Transformers.js](https://huggingface.co/docs/transformers.js) (all-MiniLM-L6-v2)
- **Vector DB**: WinnowDB
- **LLM**: [Web LLM](https://webllm.mlc.ai/) or [LLaMA.cpp WASM](https://github.com/nicbou/llama-cpp-wasm)

## Quick Start

```javascript
import init, { WinnowCollection } from 'winnow-db';
import { pipeline } from '@xenova/transformers';

// 1. Initialize
await init();
const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
const db = new WinnowCollection("rag", { dim: 384, index_type: 0 });

// 2. Index documents
async function indexDocument(id, text) {
  const chunks = chunkText(text, 512); // Split into 512 char chunks
  for (let i = 0; i < chunks.length; i++) {
    const embedding = await embed(chunks[i]);
    await db.add(id * 1000 + i, embedding, null, null, JSON.stringify({ text: chunks[i] }));
  }
}

// 3. Retrieve context
async function retrieve(query, k = 5) {
  const queryEmbed = await embed(query);
  const results = await db.search(queryEmbed, k);
  return results.map(([id, _]) => getPayload(id).text);
}

// 4. Generate answer
async function ask(question) {
  const context = await retrieve(question);
  const prompt = `Context:\n${context.join('\n\n')}\n\nQuestion: ${question}\nAnswer:`;
  return llm.generate(prompt);
}

// Helper
async function embed(text) {
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return new Float32Array(output.data);
}
```

## Demo

```bash
cd examples/05-offline-rag
npm install
npm start
```

## Performance

| Operation | Latency |
|-----------|---------|
| Embed query | 50ms |
| Search 100k chunks | 5ms |
| LLM generation | 2-10s |
| **Total RAG** | **~3s** |

*Tested on M1 MacBook with Llama-2-7B*

## Tips

1. **Chunk size**: 256-512 chars works best for most documents
2. **Overlap**: Use 10-20% overlap between chunks
3. **Reranking**: Add cross-encoder for better relevance
4. **Caching**: Cache embeddings for repeated queries
