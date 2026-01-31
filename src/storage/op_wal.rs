use super::FileSystemSyncAccessHandle;
use wasm_bindgen::prelude::*;
use js_sys::Uint8Array;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use std::io::{self, Cursor, Read};
use crc32fast::Hasher;

// Entry Header: [Length: 4 bytes] (Little Endian)
// Entry Tail:   [CRC32: 4 bytes]  (Little Endian)
// Layout: [Len] [Payload...] [CRC]

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WalEntry {
    Insert { 
        id: u32, 
        vector: Vec<f32>, 
        sparse_indices: Option<Vec<u32>>,
        sparse_values: Option<Vec<f32>>,
        payload: Option<String> 
    },
    Delete { id: u32 },
    Clear, 
}

pub struct OpWal {
    handle: FileSystemSyncAccessHandle,
    current_offset: u64,
}

impl OpWal {
    pub fn new(handle: FileSystemSyncAccessHandle) -> io::Result<Self> {
        // Seek to end? No, we need to recover first to find the valid end.
        // For 'new' we assume the caller will call 'recover' immediately.
        Ok(OpWal {
            handle,
            current_offset: 0,
        })
    }

    /// Appends an entry to the log.
    /// Format: [Length u32] [Payload] [CRC32]
    /// We checksum (Length bytes + Payload bytes).
    pub fn append(&mut self, entry: &WalEntry) -> io::Result<()> {
        let payload = bincode::serialize(entry)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        
        let len = payload.len() as u32;
        
        let mut full_frame = Vec::with_capacity(4 + payload.len() + 4);
        full_frame.extend_from_slice(&len.to_le_bytes());
        full_frame.extend_from_slice(&payload);
        
        // Calculate CRC
        let mut hasher = Hasher::new();
        hasher.update(&len.to_le_bytes()); // Include length in CRC for robustness
        hasher.update(&payload);
        let crc = hasher.finalize();
        
        full_frame.extend_from_slice(&crc.to_le_bytes());
        
        // Write to disk
        let written = self.write_at(self.current_offset, &full_frame)?;
        if written != full_frame.len() {
             return Err(io::Error::new(io::ErrorKind::Other, "Incomplete WAL write"));
        }
        
        self.handle.flush().map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;
        
        self.current_offset += full_frame.len() as u64;
        
        Ok(())
    }

    /// Reads all valid entries from the log.
    /// If a corrupted entry is found (tail), it truncates the file at that point.
    pub fn recover(&mut self) -> io::Result<Vec<WalEntry>> {
        let size = self.handle.get_size()
             .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))? as u64;
             
        let mut entries = Vec::new();
        let mut offset = 0;
        
        // Buffer for reading length (4 bytes)
        let mut head_buf = [0u8; 4];
        
        loop {
            if offset + 8 > size { // Need at least Length(4) + CRC(4) even for empty payload?
                // Actually payload min size in bincode is likely > 0.
                if offset < size {
                    // Leftover bytes < header size -> Corruption/Fragmentation
                    // Truncate here.
                    self.truncate(offset)?;
                }
                break;
            }
            
            // 1. Read Length
            if let Err(_) = self.read_exact_at(offset, &mut head_buf) {
                 // EOF or error
                 break;
            }
            let len = u32::from_le_bytes(head_buf);
            let frame_size = 4 + len as u64 + 4;
            
            if offset + frame_size > size {
                // Incomplete frame at end
                self.truncate(offset)?;
                break;
            }
            
            // 2. Read Payload + CRC
            // optimization: read payload and crc together
            let mut body_buf = vec![0u8; (len as usize) + 4];
            if let Err(_) = self.read_exact_at(offset + 4, &mut body_buf) {
                // Should not happen given check above
                break;
            }
            
            // 3. Verify CRC
            let payload = &body_buf[..len as usize];
            let stored_crc_bytes = &body_buf[len as usize..];
            let stored_crc = u32::from_le_bytes(stored_crc_bytes.try_into().unwrap());
            
            let mut hasher = Hasher::new();
            hasher.update(&head_buf); // Length
            hasher.update(payload);
            let computed_crc = hasher.finalize();
            
            if computed_crc != stored_crc {
                // Corruption detected!
                web_sys::console::warn_1(&JsValue::from_str(&format!("WAL Corruption at offset {}. Truncating.", offset)));
                self.truncate(offset)?;
                break;
            }
            
            // 4. Deserialize
            match bincode::deserialize::<WalEntry>(payload) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                     // Valid CRC but invalid data? rare. 
                     // Treat as corruption.
                     web_sys::console::error_1(&JsValue::from_str(&format!("WAL Deser Error: {:?}", e)));
                     self.truncate(offset)?;
                     break;
                }
            }
            
            // Advance
            offset += frame_size;
        }
        
        self.current_offset = offset;
        Ok(entries)
    }
    
    // Helpers
    
    pub fn reset(&mut self) -> io::Result<()> {
        self.truncate(0)?;
        self.current_offset = 0;
        Ok(())
    }

    fn truncate(&mut self, size: u64) -> io::Result<()> {
        self.handle.truncate(size as f64)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;
        Ok(())
    }
    
    fn write_at(&self, offset: u64, data: &[u8]) -> io::Result<usize> {
        // Access existing handle write
        // We'll trust our previous pattern:
        // Needs mutable slice for Uint8Array view... wait, view takes &mut [u8].
        // But we have &[u8]. We need to copy to a temp vec or cast if unsafe.
        // Unsafe cast is okay if we don't mutate JS side while holding it.
        // Or just clone.
        let mut buf = data.to_vec();
        let view = unsafe { Uint8Array::view(&mut buf) };
        let options = js_sys::Object::new();
        js_sys::Reflect::set(&options, &JsValue::from_str("at"), &JsValue::from_f64(offset as f64)).unwrap();
        
        let written = self.handle.write_with_options(&view, &options)
             .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;
             
        Ok(written as usize) // Assumes f64 fits in usize
    }
    
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let view = unsafe { Uint8Array::view(buf) };
        let options = js_sys::Object::new();
        js_sys::Reflect::set(&options, &JsValue::from_str("at"), &JsValue::from_f64(offset as f64)).unwrap();
        
        let read = self.handle.read_with_options(&view, &options)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;
            
        if read != buf.len() as f64 {
             return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Short read"));
        }
        Ok(())
    }
}
