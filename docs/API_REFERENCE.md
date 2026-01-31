# API Reference

Complete documentation for all public WinnowDB methods.

## WinnowCollection

The main class for managing vector collections.

### Constructor

```javascript
new WinnowCollection(name: string, config: CollectionConfig, wal_handle?: JsValue)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `string` | ✅ | Collection name |
| `config` | `CollectionConfig` | ✅ | Configuration object |
| `wal_handle` | `JsValue` | ❌ | OPFS file handle for WAL |

**CollectionConfig:**

```typescript
interface CollectionConfig {
  dim: number;           // Vector dimension (required)
  index_type: number;    // 0=HNSW, 1=IVFPQ, 2=SQ, 3=BQ
  m?: number;            // HNSW: connections per node (default: 16)
  ef_construction?: number; // HNSW: build quality (default: 200)
  ef_search?: number;    // HNSW: search quality (default: 100)
  n_centroids?: number;  // IVFPQ: clusters (default: 256)
  n_sub?: number;        // IVFPQ: subquantizers (default: 8)
  metric?: number;       // 0=L2, 1=InnerProduct, 2=Cosine
}
```

---

### add()

Insert a vector with optional metadata.

```javascript
await collection.add(id, vector, sparse_indices?, sparse_values?, payload?)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `number` | ✅ | Unique vector ID |
| `vector` | `Float32Array` | ✅ | Dense vector |
| `sparse_indices` | `Uint32Array` | ❌ | Non-zero indices for sparse |
| `sparse_values` | `Float32Array` | ❌ | Values for sparse indices |
| `payload` | `string` | ❌ | JSON metadata |

**Returns:** `void`

**Errors:**

- `DIMENSION_MISMATCH`: Vector length ≠ config.dim
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `MEMORY_LIMIT_EXCEEDED`: Over budget

**Example:**

```javascript
const vector = new Float32Array([0.1, 0.2, 0.3, ...]);
await collection.add(1, vector, null, null, '{"category": "tech"}');
```

---

### search()

Find nearest neighbors.

```javascript
const results = await collection.search(query, k, filter?, signal?)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | `Float32Array` | ✅ | Query vector |
| `k` | `number` | ✅ | Number of results |
| `filter` | `object` | ❌ | Metadata filter |
| `signal` | `AbortSignal` | ❌ | Cancellation signal |

**Returns:** `Array<[number, number]>` - `[[id, distance], ...]`

**Filter Operators:**

```javascript
{ "field": { "$eq": value } }    // Equals
{ "field": { "$ne": value } }    // Not equals
{ "field": { "$gt": value } }    // Greater than
{ "field": { "$lt": value } }    // Less than
{ "field": { "$in": [...] } }    // In array
{ "$and": [...] }                // Logical AND
{ "$or": [...] }                 // Logical OR
```

**Example:**

```javascript
const results = await collection.search(
  query,
  10,
  { "category": { "$eq": "tech" }, "price": { "$lt": 100 } }
);
```

---

### search_hybrid()

Combine dense and sparse search with RRF.

```javascript
const results = await collection.search_hybrid(
  dense_query, sparse_indices, sparse_values, k, dense_weight, filter?
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dense_query` | `Float32Array` | ✅ | Dense vector |
| `sparse_indices` | `Uint32Array` | ✅ | Sparse indices |
| `sparse_values` | `Float32Array` | ✅ | Sparse values |
| `k` | `number` | ✅ | Number of results |
| `dense_weight` | `number` | ✅ | Weight 0-1 (1 = all dense) |
| `filter` | `object` | ❌ | Metadata filter |

**Returns:** `Array<[number, number]>`

---

### search_with_timeout()

Search with automatic cancellation.

```javascript
const results = await collection.search_with_timeout(query, k, filter?, timeout_ms)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `timeout_ms` | `number` | ✅ | Max duration in milliseconds |

**Returns:** `Array<[number, number]>`

**Errors:**

- Rejects if timeout exceeded

---

### delete()

Remove a vector by ID.

```javascript
await collection.delete(id)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `number` | ✅ | Vector ID to delete |

**Returns:** `void`

---

### clear()

Remove all vectors.

```javascript
await collection.clear()
```

---

### count()

Get total vector count.

```javascript
const n = collection.count()
```

**Returns:** `number`

---

### compact()

Rebuild index to reclaim memory.

```javascript
await collection.compact()
```

Removes deleted vectors and re-trains quantization indices.

---

### snapshot()

Export collection to binary.

```javascript
const data = collection.snapshot()
```

**Returns:** `Uint8Array`

---

### load_snapshot()

Restore collection from binary.

```javascript
const collection = WinnowCollection.load_snapshot(data, wal_handle?)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | `Uint8Array` | ✅ | Snapshot binary |
| `wal_handle` | `JsValue` | ❌ | OPFS handle for new WAL |

**Returns:** `WinnowCollection`

---

### health_check()

Get system health status.

```javascript
const health = collection.health_check()
```

**Returns:**

```json
{
  "status": "Healthy",
  "memory_usage": 102400,
  "active_writes": 0,
  "circuit_open": false
}
```

---

### get_metrics()

Get Prometheus-format metrics.

```javascript
const metrics = collection.get_metrics()
```

**Returns:** `string` - Prometheus exposition format

---

## IndexWebGPU

GPU-accelerated brute force search (experimental).

### new()

```javascript
const gpu = await IndexWebGPU.new(dim, initial_capacity)
```

### init()

Upload vectors to GPU.

```javascript
gpu.init(flat_vectors, count)
```

### search()

```javascript
const results = await gpu.search(query, k, metric)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `metric` | `number` | 0=L2, 1=IP, 2=Cosine |

### dispose()

Release GPU resources.

```javascript
gpu.dispose()
```

---

## Error Codes

| Code | Message | Solution |
|------|---------|----------|
| `DIMENSION_MISMATCH` | Vector dimension wrong | Check config.dim |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry |
| `MEMORY_LIMIT_EXCEEDED` | Over memory budget | Call compact() |
| `INDEX_NOT_TRAINED` | IVF-PQ needs training | Call train() first |
| `CIRCUIT_OPEN` | System under stress | Wait for recovery |
