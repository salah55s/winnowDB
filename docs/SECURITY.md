# Security

WinnowDB security features and best practices.

## Threat Model

WinnowDB runs entirely in the browser. Threats include:

- **XSS**: Malicious scripts accessing vector data
- **Data Exfiltration**: Leaking embeddings to attackers
- **DoS**: Exhausting memory/CPU
- **Persistence Tampering**: Corrupting WAL/snapshots

## Built-in Protections

### Rate Limiting

Token bucket algorithm (100 RPS default):

```javascript
// Requests beyond limit return error
try {
  await collection.search(query, 10);
} catch (e) {
  if (e.includes("Rate limit")) {
    await sleep(1000);
    // Retry
  }
}
```

### Circuit Breaker

Automatic failure protection:

```javascript
const health = collection.health_check();
if (health.circuit_open) {
  // System under stress, wait before retrying
}
```

### Memory Limits

Configurable memory budget with eviction:

- Over budget → Evict payloads to OPFS
- Still over → Reject new writes

### Query Timeouts

Prevent runaway searches:

```javascript
// Auto-cancel after 500ms
const results = await collection.search_with_timeout(query, 10, null, 500);
```

## Client-Side Encryption

### At-Rest Encryption

Encrypt snapshots before storage:

```javascript
async function encryptSnapshot(snapshot, password) {
  const key = await deriveKey(password);
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const encrypted = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    key,
    snapshot
  );
  return { iv, data: new Uint8Array(encrypted) };
}

async function deriveKey(password) {
  const enc = new TextEncoder();
  const keyMaterial = await crypto.subtle.importKey(
    'raw', enc.encode(password), 'PBKDF2', false, ['deriveKey']
  );
  return crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt: enc.encode('winnowdb'), iterations: 100000, hash: 'SHA-256' },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}
```

### Zero-Knowledge Storage

For sensitive embeddings:

1. Encrypt vectors before adding
2. Store encrypted in WinnowDB
3. Decrypt on retrieval

Note: Encrypted vectors cannot be searched semantically!

## Best Practices

### 1. Validate Input

```javascript
function validateVector(v, expectedDim) {
  if (!(v instanceof Float32Array)) throw new Error('Must be Float32Array');
  if (v.length !== expectedDim) throw new Error(`Expected ${expectedDim} dims`);
  if (v.some(x => !Number.isFinite(x))) throw new Error('Invalid values');
}
```

### 2. Sanitize Payloads

```javascript
function sanitizePayload(payload) {
  // Remove sensitive fields
  const { password, token, secret, ...safe } = payload;
  return JSON.stringify(safe);
}
```

### 3. Isolate Collections

Use separate collections for different sensitivity levels.

### 4. Regular Backups

```javascript
// Daily encrypted backup
const snapshot = collection.snapshot();
const encrypted = await encryptSnapshot(snapshot, BACKUP_PASSWORD);
await saveToOpfs('backup.enc', encrypted);
```

### 5. Monitor Health

```javascript
setInterval(() => {
  const health = collection.health_check();
  if (health.memory_usage > MAX_MEMORY * 0.9) {
    console.warn('Memory high, consider compacting');
  }
}, 60000);
```

## Data Handling

### GDPR Compliance

```javascript
// Right to deletion
async function deleteUserData(userId) {
  const idsToDelete = await findIdsByUser(userId);
  for (const id of idsToDelete) {
    await collection.delete(id);
  }
  await collection.compact(); // Actually remove data
}

// Data export
async function exportUserData(userId) {
  const userVectors = await findVectorsByUser(userId);
  return JSON.stringify(userVectors);
}
```

### Data Retention

Implement TTL via metadata:

```javascript
const payload = { created: Date.now(), ttl: 86400000 }; // 24h
await collection.add(id, vector, null, JSON.stringify(payload));

// Cleanup job
const expired = await findExpired();
for (const id of expired) {
  await collection.delete(id);
}
```
