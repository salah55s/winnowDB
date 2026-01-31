# Browser Extension: Semantic History Search

Chrome extension that indexes your browsing history and enables semantic search.

## Features

- Automatically indexes visited pages
- Search history with natural language
- Runs 100% locally (no server)

## Installation

1. Build WinnowDB: `npm run build`
2. Copy `pkg/` to this directory
3. Go to `chrome://extensions`
4. Enable "Developer mode"
5. Click "Load unpacked" and select this folder

## How it Works

```javascript
// Background script indexes history
chrome.history.onVisited.addListener(async (result) => {
  const embedding = await embed(result.title + ' ' + result.url);
  await db.add(result.id, embedding, null, null, JSON.stringify(result));
});

// Popup searches
const results = await db.search(queryEmbedding, 20);
```

## Files

- `manifest.json` - Extension manifest
- `popup.html` - Search UI
- `background.js` - Indexing service worker
