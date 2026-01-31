use super::FileSystemSyncAccessHandle;
use wasm_bindgen::prelude::*;
use js_sys::Uint8Array;
use std::io::{self, Write};
use std::collections::HashMap;
use crc32fast::Hasher;

// WAL Frame: [Header 16 bytes] [Data 4096 bytes]
// Header:
// - Magic: 4 bytes (WAL1)
// - Page ID: 4 bytes
// - CRC32: 4 bytes
// - Seq: 4 bytes

const PAGE_SIZE: usize = 4096;
const HEADER_SIZE: usize = 16;
const MAGIC_BYTES: [u8; 4] = *b"WAL1";

#[derive(Debug)]
pub struct WalFrame {
    pub page_id: u32,
    pub data: Vec<u8>,
}

pub struct WalManager {
    handle: FileSystemSyncAccessHandle,
    // Map page_id -> offset in WAL file
    // If a page is in WAL multiple times, we map to the latest offset.
    index: HashMap<u32, u64>,
    current_offset: u64,
}

impl WalManager {
    pub fn new(handle: FileSystemSyncAccessHandle) -> io::Result<Self> {
        let mut wal = Self {
            handle,
            index: HashMap::new(),
            current_offset: 0,
        };
        wal.recover()?;
        Ok(wal)
    }

    fn recover(&mut self) -> io::Result<()> {
        // Scan the WAL file frame by frame.
        // Valid frames update the index.
        // Invalid frames (torn writes) stop the scan.
        
        let size = self.handle.get_size()
             .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))? as u64;

        let mut offset = 0;
        let mut buf = vec![0u8; HEADER_SIZE + PAGE_SIZE];

        while offset + (HEADER_SIZE + PAGE_SIZE) as u64 <= size {
            // Read Frame
            // Using read_with_options directly or we need a helper.
            // Let's implement a read_exact equivalent for the handle wrapper if possible,
            // or just use manual JS calls here for simplicity as we don't have a Read impl on Handle directly (OpfsReader has it).
            // Actually, we can reuse OpfsReader logic via a helper or just manual coding.
            
            // Manual read for now to keep WalManager independent of OpfsReader struct if needed.
            
            let view = unsafe { Uint8Array::view(&mut buf) };
            let options = js_sys::Object::new();
            js_sys::Reflect::set(&options, &JsValue::from_str("at"), &JsValue::from_f64(offset as f64)).unwrap();
            
            let bytes_read = self.handle.read_with_options(&view, &options)
                 .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;
            
            if bytes_read != (HEADER_SIZE + PAGE_SIZE) as f64 {
                // Incomplete frame, stop.
                break;
            }

            // Verify Magic
            if buf[0..4] != MAGIC_BYTES {
                 // Corrupt or different version? Stop.
                 break;
            }

            // Read Page ID
            let page_id = u32::from_le_bytes(buf[4..8].try_into().unwrap());
            
            // Read CRC
            let expected_crc = u32::from_le_bytes(buf[8..12].try_into().unwrap());
            
            // Verify
            let mut hasher = Hasher::new();
            hasher.update(&buf[HEADER_SIZE..]); // Data
            hasher.update(&page_id.to_le_bytes()); // Mix metadata
            if hasher.finalize() != expected_crc {
                // Corrupt frame, stop recovery here
                break;
            }
            
            // Index it
            self.index.insert(page_id, offset);
            
            offset += (HEADER_SIZE + PAGE_SIZE) as u64;
        }

        self.current_offset = offset;
        // Truncate any tearing at the end?
        // Ideally yes, but SyncAccessHandle truncate might be tricky.
        // For now, we just append from known good state.
        
        Ok(())
    }
    
    pub fn write_page(&mut self, page_id: u32, data: &[u8]) -> io::Result<()> {
        if data.len() != PAGE_SIZE {
             return Err(io::Error::new(io::ErrorKind::InvalidInput, "Data must be 4KB"));
        }
        
        let mut crc_hasher = Hasher::new();
        crc_hasher.update(data);
        crc_hasher.update(&page_id.to_le_bytes());
        let crc = crc_hasher.finalize();
        
        let mut frame = Vec::with_capacity(HEADER_SIZE + PAGE_SIZE);
        frame.extend_from_slice(&MAGIC_BYTES);
        frame.extend_from_slice(&page_id.to_le_bytes());
        frame.extend_from_slice(&crc.to_le_bytes()); // Real CRC
        frame.extend_from_slice(&[0u8; 4]); // Seq placeholder
        frame.extend_from_slice(data);
        
        let view = unsafe { Uint8Array::view(&mut frame) };
        let options = js_sys::Object::new();
        js_sys::Reflect::set(&options, &JsValue::from_str("at"), &JsValue::from_f64(self.current_offset as f64)).unwrap();
        
        let written = self.handle.write_with_options(&view, &options)
             .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;
             
        if written != (HEADER_SIZE + PAGE_SIZE) as f64 {
             return Err(io::Error::new(io::ErrorKind::Other, "Incomplete WAL write"));
        }
        
        self.handle.flush().map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;
        
        self.index.insert(page_id, self.current_offset);
        self.current_offset += (HEADER_SIZE + PAGE_SIZE) as u64;
        
        Ok(())
    }
    
    pub fn read_page(&self, page_id: u32) -> Option<Vec<u8>> {
        let offset = self.index.get(&page_id)?;
        
        // Read data part
        let data_offset = offset + HEADER_SIZE as u64;
        let mut buf = vec![0u8; PAGE_SIZE];
        let view = unsafe { Uint8Array::view(&mut buf) };
        let options = js_sys::Object::new();
        js_sys::Reflect::set(&options, &JsValue::from_str("at"), &JsValue::from_f64(data_offset as f64)).unwrap();
        
        let read = self.handle.read_with_options(&view, &options).ok()?;
        if read != PAGE_SIZE as f64 { return None; }
        
        Some(buf)
    }
    
    pub fn checkpoint(&mut self, main_handle: &FileSystemSyncAccessHandle) -> io::Result<()> {
        // Move all pages from WAL to Main
        // We iterate over the index (latest versions of pages)
        
        for (page_id, offset) in &self.index {
             // Read from WAL
             let mut buf = vec![0u8; PAGE_SIZE];
             let view = unsafe { Uint8Array::view(&mut buf) };
             let read_opts = js_sys::Object::new();
             js_sys::Reflect::set(&read_opts, &JsValue::from_str("at"), &JsValue::from_f64((offset + HEADER_SIZE as u64) as f64)).unwrap();
             
             let bytes = self.handle.read_with_options(&view, &read_opts)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Checkpoint read failed: {:?}", e)))?;
             
             if bytes != PAGE_SIZE as f64 {
                 return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Incomplete page in WAL during checkpoint"));
             }
             
             // Write to Main
             let main_offset = (*page_id as u64) * (PAGE_SIZE as u64);
             let write_opts = js_sys::Object::new();
             js_sys::Reflect::set(&write_opts, &JsValue::from_str("at"), &JsValue::from_f64(main_offset as f64)).unwrap();
             main_handle.write_with_options(&view, &write_opts)
                 .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Checkpoint write failed: {:?}", e)))?;
        }
        
        main_handle.flush().map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;
        
        // Reset WAL
        // self.handle.truncate(0) ... missing bindgen
        // For now, we just reset our internal tracking and overwrite. 
        // Real WAL requires truncate or file deletion.
        // Assuming we can overwrite from 0.
        
        self.current_offset = 0;
        self.index.clear();
        self.handle.flush().unwrap();
        
        Ok(())
    }
}
