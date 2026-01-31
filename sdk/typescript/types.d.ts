/**
 * WinnowDB TypeScript SDK
 * 
 * High-level TypeScript bindings for the WinnowDB vector database.
 */

// ============================================================================
// Core Types
// ============================================================================

export interface CollectionConfig {
    name: string;
    dim: number;
    indexType?: 'Flat' | 'HNSW' | 'IVFPQ' | 'SQ' | 'BQ';
    metric?: 'Cosine' | 'Euclidean' | 'DotProduct';
    hnswM?: number;
    hnswEfConstruction?: number;
}

export interface VectorPoint {
    id: number;
    vector: Float32Array | number[];
    payload?: Record<string, unknown>;
    sparse?: SparseVector;
}

export interface SparseVector {
    indices: Uint32Array | number[];
    values: Float32Array | number[];
}

export interface SearchResult {
    id: number;
    score: number;
    payload?: Record<string, unknown>;
}

export interface SearchOptions {
    filter?: Filter;
    includePayload?: boolean;
    includeVector?: boolean;
    scoreThreshold?: number;
    offset?: number;
    timeout?: number;
}

// ============================================================================
// Filter Types (MongoDB-style)
// ============================================================================

export type FilterValue = string | number | boolean | null | FilterValue[];

export type FilterOperator =
    | { $eq: FilterValue }
    | { $ne: FilterValue }
    | { $gt: number }
    | { $gte: number }
    | { $lt: number }
    | { $lte: number }
    | { $in: FilterValue[] }
    | { $nin: FilterValue[] }
    | { $contains: string }
    | { $startsWith: string }
    | { $endsWith: string }
    | { $exists: boolean };

export type FilterCondition = Record<string, FilterValue | FilterOperator>;

export interface Filter {
    $and?: Filter[];
    $or?: Filter[];
    $not?: Filter;
    [field: string]: FilterValue | FilterOperator | Filter[] | Filter | undefined;
}

// ============================================================================
// Multi-Tenancy Types
// ============================================================================

export type TenantTier = 'free' | 'starter' | 'professional' | 'enterprise';
export type TenantStatus = 'active' | 'suspended' | 'deleted' | 'trial';

export interface Tenant {
    id: string;
    name: string;
    tier: TenantTier;
    status: TenantStatus;
    createdAt: number;
    metadata: Record<string, string>;
}

export interface ResourceQuota {
    maxCollections: number;
    maxVectorsPerCollection: number;
    maxTotalVectors: number;
    maxStorageBytes: number;
    maxQpm: number;
    maxConcurrentOps: number;
    maxDimensions: number;
}

export interface ResourceUsage {
    collections: number;
    totalVectors: number;
    storageBytes: number;
    queriesThisMinute: number;
    currentConcurrentOps: number;
}

export interface QuotaCheckResult {
    allowed: boolean;
    reason?: string;
    usagePercent: number;
}

// ============================================================================
// Advanced Search Types
// ============================================================================

export type MultiVectorStrategy =
    | 'union'
    | 'intersection'
    | 'average'
    | 'max'
    | 'min'
    | { weighted: number[] };

export interface MultiVectorQuery {
    vectors: (Float32Array | number[])[];
    strategy: MultiVectorStrategy;
    k: number;
}

export interface BatchQuery {
    id: string;
    vector: Float32Array | number[];
    k: number;
    filter?: Filter;
}

export interface BatchSearchResult {
    queryId: string;
    results: SearchResult[];
    latencyMs: number;
    error?: string;
}

export interface GroupedSearchRequest {
    vector: Float32Array | number[];
    groupBy: string;
    groupsLimit: number;
    groupSize: number;
}

export interface SearchGroup {
    groupValue: string;
    hits: SearchResult[];
    bestScore: number;
}

export interface RecommendRequest {
    positiveIds: number[];
    negativeIds: number[];
    k: number;
}

export interface ScrollRequest {
    filter?: Filter;
    limit: number;
    offset?: string;
    withPayload?: boolean;
    withVector?: boolean;
}

export interface ScrollResult {
    points: VectorPoint[];
    nextOffset?: string;
    hasMore: boolean;
}

// ============================================================================
// Query Language Types
// ============================================================================

export type SearchType = 'dense' | 'sparse' | 'hybrid';

export interface Query {
    operation: 'search' | 'get' | 'count' | 'scroll' | 'aggregate';
    collection: string;
    vector?: Float32Array | number[];
    sparse?: SparseVector;
    k?: number;
    filter?: Filter;
    searchType?: SearchType;
    options?: QueryOptions;
}

export interface QueryOptions {
    timeout?: number;
    consistency?: 'eventual' | 'strong';
}

// ============================================================================
// SDK Client Interface
// ============================================================================

export interface WinnowDBClient {
    // Collection operations
    createCollection(config: CollectionConfig): Promise<void>;
    deleteCollection(name: string): Promise<void>;
    listCollections(): Promise<string[]>;

    // Vector operations
    insert(collection: string, points: VectorPoint[]): Promise<number[]>;
    upsert(collection: string, points: VectorPoint[]): Promise<number[]>;
    delete(collection: string, ids: number[]): Promise<void>;
    get(collection: string, ids: number[]): Promise<VectorPoint[]>;

    // Search operations
    search(collection: string, vector: number[], k: number, options?: SearchOptions): Promise<SearchResult[]>;
    searchSparse(collection: string, sparse: SparseVector, k: number, options?: SearchOptions): Promise<SearchResult[]>;
    searchHybrid(collection: string, dense: number[], sparse: SparseVector, k: number, alpha: number, options?: SearchOptions): Promise<SearchResult[]>;

    // Advanced search
    searchMultiVector(collection: string, query: MultiVectorQuery, options?: SearchOptions): Promise<SearchResult[]>;
    searchBatch(collection: string, queries: BatchQuery[]): Promise<BatchSearchResult[]>;
    searchGrouped(collection: string, request: GroupedSearchRequest, options?: SearchOptions): Promise<SearchGroup[]>;
    recommend(collection: string, request: RecommendRequest): Promise<SearchResult[]>;
    scroll(collection: string, request: ScrollRequest): Promise<ScrollResult>;

    // Count
    count(collection: string, filter?: Filter): Promise<number>;

    // Persistence
    snapshot(): Promise<Uint8Array>;
    loadSnapshot(data: Uint8Array): Promise<void>;
}

// ============================================================================
// SDK Builder
// ============================================================================

export interface WinnowDBConfig {
    /** Tenant ID for multi-tenant mode */
    tenantId?: string;

    /** API key for authentication */
    apiKey?: string;

    /** JWT token for authentication */
    jwtToken?: string;

    /** Default timeout in milliseconds */
    timeout?: number;

    /** Enable debug logging */
    debug?: boolean;
}

/**
 * Create a WinnowDB client instance.
 * 
 * @example
 * ```typescript
 * import { createClient } from '@winnowdb/sdk';
 * 
 * const client = await createClient({
 *   tenantId: 'my-tenant',
 *   apiKey: 'my-api-key',
 * });
 * 
 * // Create a collection
 * await client.createCollection({
 *   name: 'embeddings',
 *   dim: 384,
 *   indexType: 'HNSW',
 * });
 * 
 * // Insert vectors
 * await client.insert('embeddings', [
 *   { id: 1, vector: new Float32Array([0.1, 0.2, 0.3, ...]), payload: { text: 'hello' } },
 * ]);
 * 
 * // Search
 * const results = await client.search('embeddings', queryVector, 10, {
 *   filter: { category: 'tech' },
 * });
 * ```
 */
export declare function createClient(config?: WinnowDBConfig): Promise<WinnowDBClient>;

// ============================================================================
// Query Builder
// ============================================================================

export declare class QueryBuilder {
    constructor(collection: string);

    /** Set the query vector */
    vector(v: number[]): this;

    /** Set sparse vector */
    sparse(indices: number[], values: number[]): this;

    /** Enable hybrid search */
    hybrid(dense: number[], sparseIndices: number[], sparseValues: number[]): this;

    /** Set number of results */
    topK(k: number): this;

    /** Add a filter */
    filter(f: Filter): this;

    /** Filter where field equals value */
    whereEq(field: string, value: FilterValue): this;

    /** Filter where field is greater than value */
    whereGt(field: string, value: number): this;

    /** Filter where field is in list */
    whereIn(field: string, values: FilterValue[]): this;

    /** Include payload in results */
    includePayload(include: boolean): this;

    /** Include vector in results */
    includeVector(include: boolean): this;

    /** Set minimum score threshold */
    scoreThreshold(threshold: number): this;

    /** Set result offset */
    offset(off: number): this;

    /** Set timeout */
    timeout(ms: number): this;

    /** Build and return the query */
    build(): Query;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Normalize a vector to unit length.
 */
export declare function normalizeVector(vector: number[]): Float32Array;

/**
 * Compute cosine similarity between two vectors.
 */
export declare function cosineSimilarity(a: number[], b: number[]): number;

/**
 * Compute Euclidean distance between two vectors.
 */
export declare function euclideanDistance(a: number[], b: number[]): number;
