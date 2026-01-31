/**
 * WinnowDB Production Configuration
 * 
 * Best practices for production deployments.
 */

import init, { WinnowCollection } from 'winnow-db';

// ============================================
// 1. INITIALIZATION
// ============================================

async function createProductionDB() {
    await init();

    const config = {
        dim: 768,
        index_type: 0,       // HNSW for general use
        m: 16,               // Balanced connections
        ef_construction: 200, // Good build quality
        ef_search: 100,      // Good search quality
        metric: 0,           // L2 distance
    };

    // Get WAL handle from OPFS
    const opfsRoot = await navigator.storage.getDirectory();
    const walFile = await opfsRoot.getFileHandle('wal.bin', { create: true });
    const walHandle = await walFile.createSyncAccessHandle();

    return new WinnowCollection("production", config, walHandle);
}

// ============================================
// 2. ERROR HANDLING
// ============================================

async function safeAdd(db, id, vector, payload) {
    try {
        await db.add(id, vector, null, null, JSON.stringify(payload));
        return { success: true };
    } catch (e) {
        const error = e.toString();

        if (error.includes("Rate limit")) {
            // Wait and retry
            await sleep(1000);
            return safeAdd(db, id, vector, payload);
        }

        if (error.includes("Memory limit")) {
            // Compact and retry
            await db.compact();
            return safeAdd(db, id, vector, payload);
        }

        return { success: false, error };
    }
}

async function safeSearch(db, query, k, filter, timeoutMs = 5000) {
    try {
        return await db.search_with_timeout(query, k, filter, timeoutMs);
    } catch (e) {
        console.error('Search failed:', e);
        return [];
    }
}

// ============================================
// 3. HEALTH MONITORING
// ============================================

function startHealthMonitor(db, intervalMs = 60000) {
    return setInterval(() => {
        const health = db.health_check();
        const data = JSON.parse(health);

        console.log(`[Health] Status: ${data.status}, Memory: ${formatBytes(data.memory_usage)}, Writes: ${data.active_writes}`);

        if (data.circuit_open) {
            console.warn('[Alert] Circuit breaker is OPEN - system under stress');
        }

        if (data.memory_usage > 400 * 1024 * 1024) { // 400MB
            console.warn('[Alert] Memory usage high, consider compacting');
        }
    }, intervalMs);
}

// ============================================
// 4. BACKUP & RESTORE
// ============================================

async function backup(db) {
    const snapshot = db.snapshot();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

    // Save to OPFS
    const opfsRoot = await navigator.storage.getDirectory();
    const backupFile = await opfsRoot.getFileHandle(`backup-${timestamp}.bin`, { create: true });
    const writable = await backupFile.createWritable();
    await writable.write(snapshot);
    await writable.close();

    console.log(`[Backup] Created backup: backup-${timestamp}.bin (${formatBytes(snapshot.length)})`);
    return `backup-${timestamp}.bin`;
}

async function restore(filename) {
    const opfsRoot = await navigator.storage.getDirectory();
    const file = await opfsRoot.getFileHandle(filename);
    const fileData = await file.getFile();
    const data = new Uint8Array(await fileData.arrayBuffer());

    return WinnowCollection.load_snapshot(data, null);
}

// ============================================
// 5. METRICS EXPORT
// ============================================

function exportMetrics(db) {
    // Prometheus format
    const metrics = db.get_metrics();

    // Send to your monitoring system
    // fetch('/metrics', { method: 'POST', body: metrics });

    return metrics;
}

// ============================================
// 6. GRACEFUL SHUTDOWN
// ============================================

async function shutdown(db, intervalId) {
    console.log('[Shutdown] Starting graceful shutdown...');

    // Stop health monitor
    clearInterval(intervalId);

    // Final backup
    await backup(db);

    // Compact to save space
    await db.compact();

    console.log('[Shutdown] Complete');
}

// ============================================
// HELPERS
// ============================================

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

// ============================================
// USAGE
// ============================================

async function main() {
    // Initialize
    const db = await createProductionDB();
    const healthMonitor = startHealthMonitor(db);

    // Use
    await safeAdd(db, 1, new Float32Array(768).fill(0.1), { title: 'Doc 1' });
    const results = await safeSearch(db, new Float32Array(768).fill(0.1), 10, null);

    // Shutdown
    window.addEventListener('beforeunload', () => shutdown(db, healthMonitor));
}

export { createProductionDB, safeAdd, safeSearch, backup, restore, exportMetrics };
