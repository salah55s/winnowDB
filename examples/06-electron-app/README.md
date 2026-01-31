# Electron App: Semantic Notes

Desktop note-taking app with semantic search powered by WinnowDB.

## Features

- ✅ Create, edit, delete notes
- ✅ Semantic search (find by meaning, not keywords)
- ✅ Offline-first (no internet required)
- ✅ Cross-platform (Windows, Mac, Linux)

## Quick Start

```bash
cd examples/06-electron-app
npm install
npm start
```

## Project Structure

```
06-electron-app/
├── main.js        # Electron main process
├── preload.js     # Bridge to renderer
├── renderer.js    # UI logic + WinnowDB
├── index.html     # App UI
└── package.json
```

## Key Code

### Initialize WinnowDB

```javascript
// renderer.js
import init, { WinnowCollection } from 'winnow-db';

await init();
const db = new WinnowCollection("notes", { dim: 384, index_type: 0 });
```

### Add Note

```javascript
async function addNote(title, content) {
  const id = Date.now();
  const embedding = await embed(title + ' ' + content);
  await db.add(id, embedding, null, null, JSON.stringify({ title, content }));
  
  // Persist
  const snapshot = db.snapshot();
  await fs.writeFile('notes.db', snapshot);
}
```

### Search Notes

```javascript
async function searchNotes(query) {
  const embedding = await embed(query);
  const results = await db.search(embedding, 10);
  return results.map(([id, score]) => ({ ...getNote(id), score }));
}
```

## Build for Distribution

```bash
npm run build:mac    # macOS .dmg
npm run build:win    # Windows .exe
npm run build:linux  # Linux .AppImage
```

## Dependencies

```json
{
  "dependencies": {
    "winnow-db": "^0.1.0",
    "@xenova/transformers": "^2.0.0"
  },
  "devDependencies": {
    "electron": "^28.0.0",
    "electron-builder": "^24.0.0"
  }
}
```
