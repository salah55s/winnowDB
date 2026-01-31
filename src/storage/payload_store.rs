use super::FileSystemSyncAccessHandle;
use wasm_bindgen::prelude::*;
use js_sys::Uint8Array;
use std::io::{self, Cursor, Read, Write, Seek, SeekFrom};
use crc32fast::Hasher;

pub struct PayloadStore {
    handle: FileSystemSyncAccessHandle,
    current_offset: u64,
}

impl PayloadStore {
    pub fn new(handle: FileSystemSyncAccessHandle) -> io::Result<Self> {
        // Init offset to end of file
        let size = handle.get_size()
             .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))? as u64;
             
        Ok(PayloadStore {
            handle,
            current_offset: size,
        })
    }

    /// Saves a payload to disk. Returns (offset, length).
    /// Format: [Length u32] [Data...] [CRC32]
    pub fn save(&mut self, data: &[u8]) -> io::Result<(u64, u32)> {
        let len = data.len() as u32;
        let start_offset = self.current_offset;
        
        let mut full_frame = Vec::with_capacity(4 + data.len() + 4);
        full_frame.extend_from_slice(&len.to_le_bytes());
        full_frame.extend_from_slice(data);
        
        // CRC
        let mut hasher = Hasher::new();
        hasher.update(&len.to_le_bytes());
        hasher.update(data);
        let crc = hasher.finalize();
        
        full_frame.extend_from_slice(&crc.to_le_bytes());
        
        // Write
        let written = self.write_at(start_offset, &full_frame)?;
        if written != full_frame.len() {
             return Err(io::Error::new(io::ErrorKind::Other, "Incomplete Payload write"));
        }
        
        self.handle.flush().map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;
        
        self.current_offset += full_frame.len() as u64;
        
        Ok((start_offset, full_frame.len() as u32))
    }

    /// Reads payload from disk given offset and total frame length.
    pub fn load(&self, offset: u64, length: u32) -> io::Result<Vec<u8>> {
        // Read exact frame
        let mut buf = vec![0u8; length as usize];
        self.read_exact_at(offset, &mut buf)?;
        
        // Verify structure
        if buf.len() < 8 { return Err(io::Error::new(io::ErrorKind::InvalidData, "Frame too short")); }
        
        let len_bytes = &buf[0..4];
        let data_len = u32::from_le_bytes(len_bytes.try_into().unwrap());
        
        if (data_len as usize) + 8 != length as usize {
             return Err(io::Error::new(io::ErrorKind::InvalidData, "Length mismatch"));
        }
        
        let data = &buf[4..(4 + data_len as usize)];
        let stored_crc_bytes = &buf[(4 + data_len as usize)..];
        let stored_crc = u32::from_le_bytes(stored_crc_bytes.try_into().unwrap());
        
        // Verify CRC
        let mut hasher = Hasher::new();
        hasher.update(len_bytes);
        hasher.update(data);
        let computed = hasher.finalize();
        
        if computed != stored_crc {
             return Err(io::Error::new(io::ErrorKind::InvalidData, "CRC Mismatch"));
        }
        
        Ok(data.to_vec())
    }
    
    // Helpers (duplicated from OpWal - could reuse trait but this is faster for now)
    fn write_at(&self, offset: u64, data: &[u8]) -> io::Result<usize> {
        let mut buf = data.to_vec();
        let view = unsafe { Uint8Array::view(&mut buf) };
        let options = js_sys::Object::new();
        js_sys::Reflect::set(&options, &JsValue::from_str("at"), &JsValue::from_f64(offset as f64)).unwrap();
        
        let written = self.handle.write_with_options(&view, &options)
             .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;
             
        Ok(written as usize)
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
