# Architecture

Deep dive into how WinnowDB works under the hood.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     WinnowCollection                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Index     │  │   Sparse    │  │    VectorStorage    │  │
│  │  (HNSW/PQ)  │  │   Index     │  │   (Raw Vectors)     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                    │              │
│         └────────────────┴────────────────────┘              │
│                          │                                   │
│  ┌───────────────────────┴───────────────────────────────┐  │
│  │                    Persistence                         │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────┐   │  │
│  │  │   WAL    │  │  Snapshots   │  │  PayloadStore  │   │  │
│  │  │  (OPFS)  │  │   (Binary)   │  │    (OPFS)      │   │  │
│  │  └──────────┘  └──────────────┘  └────────────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Index Types

### HNSW (Hierarchical Navigable Small World)

Best for: High recall, fast search, moderate datasets (1M vectors)

```
Layer 3:    [A] ─────────────────────────── [B]
             │                               │
Layer 2:    [A] ──── [C] ──── [D] ───────── [B]
             │        │        │             │
Layer 1:    [A]─[E]─[C]─[F]─[D]─[G]─[H]─[B]─[I]
             │   │   │   │   │   │   │   │   │
Layer 0:    [A][E][C][F][D][G][H][B][I][J][K][L]
                        (all nodes)
```

**How it works:**

1. Build: Insert nodes starting at top layer, connect to M nearest
2. Search: Descend from top, greedily follow best neighbors
3. Return: Final search at layer 0 with ef_search candidates

**Parameters:**

- `m`: Connections per node (higher = better recall, more memory)
- `ef_construction`: Build quality (higher = slower build, better graph)
- `ef_search`: Search quality (higher = slower search, better recall)

**Complexity:**

- Build: O(n log n)
- Search: O(log n)
- Memory: O(n × m)

---

### IVF-PQ (Inverted File with Product Quantization)

Best for: Large datasets (10M+ vectors), memory constrained

```
Training:
┌────────────────────────────────────────────────┐
│ Vectors → K-Means → 256 Centroids (Coarse)     │
│                                                 │
│ Vectors → Split into 8 subvectors               │
│        → K-Means per subvector → 256 codes each │
└────────────────────────────────────────────────┘

Storage:
┌─────────────────────────────────────────────────┐
│ Vector [1024 floats] → [8 bytes] (32x smaller!) │
│                                                  │
│ Coarse centroid + 8 PQ codes per vector          │
└─────────────────────────────────────────────────┘
```

**How it works:**

1. Train: Cluster vectors into centroids, learn PQ codebooks
2. Assign: Each vector → nearest centroid + PQ codes
3. Search: Find nearest centroids, scan assigned vectors using PQ

**Parameters:**

- `n_centroids`: Coarse clusters (more = better recall, slower search)
- `n_sub`: Subquantizers (more = better precision, more memory)

**Complexity:**

- Build: O(n × iterations)
- Search: O(n_centroids + n/n_centroids)
- Memory: O(n × n_sub) bytes

---

### SQ8 (Scalar Quantization)

Best for: Memory efficiency with good recall

```
Original:  [0.123, 0.456, 0.789, 0.012]  (16 bytes)
Quantized: [31, 117, 201, 3]             (4 bytes)
           ↑    ↑    ↑    ↑
           Maps float range to 0-255
```

**Compression:** 4x memory reduction

---

### BQ (Binary Quantization)

Best for: Ultra-fast, low precision use cases

```
Original:  [0.1, -0.2, 0.3, -0.4]  (16 bytes)
Binary:    [1, 0, 1, 0]            (0.5 bytes)
           ↑  ↑  ↑  ↑
           sign(value)
```

**Compression:** 32x reduction
**Search:** XOR + popcount (extremely fast)

---

## Hybrid Search & RRF

Combines dense (semantic) and sparse (keyword) search:

```
Dense Results:   [A, B, C, D, E]  (semantic similarity)
Sparse Results:  [B, A, F, G, H]  (keyword matching)
                    ↓
RRF Formula:     score(d) = Σ 1/(k + rank(d))
                    ↓
Final:           [B, A, C, D, F]  (fused ranking)
```

The `dense_weight` parameter controls the balance (0.7 = 70% dense, 30% sparse).

---

## Persistence

### Write-Ahead Log (WAL)

```
WAL Entry Format:
┌────────────────────────────────────────┐
│ [Length: 4B] [Payload: N] [CRC32: 4B]  │
└────────────────────────────────────────┘

Entry Types:
- Insert { id, vector, sparse, payload }
- Delete { id }
- Clear
```

**Recovery process:**

1. Read WAL from start
2. Validate each entry (length + CRC32)
3. Replay operations into fresh index
4. Truncate any corrupted tail

### Snapshots

Binary serialization of entire collection state:

- Index structure (nodes, edges, centroids)
- Raw vectors (VectorStorage)
- Metadata (payloads, active IDs)

**Format:** bincode serialization (~90% of in-memory size)

---

## Memory Management

```
┌────────────────────────────────────────────┐
│              Memory Budget                 │
├────────────────────────────────────────────┤
│  Vectors:      ████████████░░░  75%        │
│  Index:        ██░░░░░░░░░░░░░  15%        │
│  Payloads:     █░░░░░░░░░░░░░░  10%        │
├────────────────────────────────────────────┤
│  Budget: 500MB  Used: 450MB  Free: 50MB    │
└────────────────────────────────────────────┘
```

When over budget:

1. Evict payloads to OPFS (LRU)
2. If still over, reject new writes

---

## Concurrency Model

```rust
WinnowCollection {
    index: Arc<RwLock<VectorIndexEnum>>,
    sparse_index: Arc<RwLock<Option<IndexSparse>>>,
    payloads: Arc<RwLock<HashMap<u32, Payload>>>,
    active_ids: Arc<RwLock<HashSet<u32>>>,
    vectors: Arc<RwLock<VectorStorage>>,
    wal: Arc<RwLock<Option<OpWal>>>,
}
```

- **Reads:** Multiple concurrent readers
- **Writes:** Single writer (blocks readers briefly)
- **Snapshots:** Global read lock (consistent point-in-time)

---

## WebGPU Acceleration

For brute-force search on GPU:

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    var dist: f32 = 0.0;
    for (var i: u32 = 0u; i < dim; i++) {
        let diff = vectors[idx * dim + i] - query[i];
        dist += diff * diff;
    }
    results[idx] = sqrt(dist);
}
```

Dispatches 64 threads per workgroup, computes all distances in parallel.
