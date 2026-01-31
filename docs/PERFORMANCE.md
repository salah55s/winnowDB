# Performance Tuning

Optimize WinnowDB for your specific use case.

## Index Selection Guide

| Use Case | Index | Config |
|----------|-------|--------|
| General (< 1M vectors) | HNSW | `m=16, ef_c=200` |
| High recall critical | HNSW | `m=32, ef_c=400` |
| Large dataset (10M+) | IVF-PQ | `n_centroids=1024` |
| Memory constrained | SQ8 | - |
| Ultra-fast, low precision | BQ | - |

## HNSW Tuning

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `m` | 16 | Higher = better recall, more memory |
| `ef_construction` | 200 | Higher = slower build, better graph |
| `ef_search` | 100 | Higher = slower search, better recall |

### Recommended Configs

```javascript
// High recall (98%+)
{ m: 32, ef_construction: 400, ef_search: 200 }

// Balanced
{ m: 16, ef_construction: 200, ef_search: 100 }

// Fast search
{ m: 8, ef_construction: 100, ef_search: 50 }
```

## IVF-PQ Tuning

### Training Requirements

- Minimum: 1,000 vectors (10 × n_centroids)
- Recommended: 10,000+ vectors

```javascript
// Training data should be representative
const trainingData = new Float32Array(10000 * dim);
await collection.train(trainingData);
```

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_centroids` | 256 | More = better recall, slower search |
| `n_sub` | 8 | More = better precision, more memory |

## Memory Usage

### Calculation

```
HNSW:
  Per vector: 4 × dim + 8 × m × layers
  1M × 768d × m=16: ~4GB

IVF-PQ:
  Per vector: n_sub bytes + overhead
  1M × 768d × n_sub=8: ~10MB + centroids

SQ8:
  Per vector: dim bytes
  1M × 768d: ~768MB

BQ:
  Per vector: dim/8 bytes
  1M × 768d: ~96MB
```

### Reducing Memory

1. Use IVF-PQ or SQ8 for large datasets
2. Call `compact()` periodically
3. Evict payloads with memory budget

## Latency Optimization

### Search Latency

| Bottleneck | Fix |
|------------|-----|
| High ef_search | Reduce ef_search |
| Large k | Reduce k, paginate |
| Complex filters | Use IVF with coarse filter |
| WebGPU init | Reuse IndexWebGPU |

### Insert Latency

| Bottleneck | Fix |
|------------|-----|
| WAL flush | Batch inserts |
| HNSW reconnect | Lower m |
| Memory limit | Increase budget |

## Benchmarks

### Standard Datasets

| Dataset | Vectors | Dim | Index | Recall@10 | QPS | Latency |
|---------|---------|-----|-------|-----------|-----|---------|
| SIFT-1M | 1M | 128 | HNSW | 95.2% | 2,500 | 0.4ms |
| SIFT-1M | 1M | 128 | IVF-PQ | 91.8% | 4,200 | 0.2ms |
| GloVe | 1.2M | 300 | HNSW | 93.8% | 1,800 | 0.6ms |
| OpenAI | 100k | 1536 | HNSW | 94.5% | 800 | 1.2ms |

### Comparison

| Library | Engine | Recall@10 | QPS |
|---------|--------|-----------|-----|
| WinnowDB (WASM) | HNSW | 95.2% | 2,500 |
| hnswlib (Native) | HNSW | 95.8% | 3,200 |
| Faiss (Native) | IVF-HNSW | 96.1% | 2,800 |
| usearch (WASM) | HNSW | 94.5% | 2,100 |

## Best Practices

1. **Batch inserts** - Insert 100+ vectors at once
2. **Warm-up** - Run 10 searches before benchmarking
3. **Compact regularly** - Call `compact()` after bulk deletes
4. **Profile first** - Use `health_check()` to identify bottlenecks
