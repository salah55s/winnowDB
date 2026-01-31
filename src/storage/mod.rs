use std::io::{self, Read, Seek, SeekFrom};
use wasm_bindgen::prelude::*;
use js_sys::Uint8Array;

pub mod paged;
pub mod wal;
pub mod op_wal;
pub mod payload_store;
pub mod vault;
pub mod snapshot;
pub mod backup;
pub use paged::PagedReader;
pub use wal::WalManager;
pub use op_wal::OpWal;
pub use payload_store::PayloadStore;
pub use vault::{VaultReader, VaultWriter};
pub use snapshot::{SnapshotHeader, SnapshotError, wrap_snapshot, unwrap_snapshot, SNAPSHOT_VERSION};
pub use backup::{BackupScheduler, BackupManager, BackupConfig, BackupBackend};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = FileSystemSyncAccessHandle)]
    pub type FileSystemSyncAccessHandle;

    #[wasm_bindgen(method, catch, js_name = read)]
    pub fn read_with_options(
        this: &FileSystemSyncAccessHandle,
        buffer: &Uint8Array,
        options: &JsValue,
    ) -> Result<f64, JsValue>;

    #[wasm_bindgen(method, catch, js_name = close)]
    pub fn close(this: &FileSystemSyncAccessHandle) -> Result<(), JsValue>;
    
    #[wasm_bindgen(method, catch, js_name = getSize)]
    pub fn get_size(this: &FileSystemSyncAccessHandle) -> Result<f64, JsValue>;
    
    #[wasm_bindgen(method, catch, js_name = flush)]
    pub fn flush(this: &FileSystemSyncAccessHandle) -> Result<(), JsValue>;

    #[wasm_bindgen(method, catch, js_name = write)]
    pub fn write_with_options(
        this: &FileSystemSyncAccessHandle,
        buffer: &Uint8Array,
        options: &JsValue,
    ) -> Result<f64, JsValue>;

    #[wasm_bindgen(method, catch, js_name = truncate)]
    pub fn truncate(this: &FileSystemSyncAccessHandle, new_size: f64) -> Result<(), JsValue>;
}

pub struct OpfsReader {
    handle: FileSystemSyncAccessHandle,
    pos: u64,
    size: u64,
}

impl OpfsReader {
    pub fn new(handle: FileSystemSyncAccessHandle) -> io::Result<Self> {
        let size = handle.get_size()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))? as u64;
            
        Ok(Self {
            handle,
            pos: 0,
            size,
        })
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

impl Read for OpfsReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.size {
            return Ok(0);
        }

        // Create a view into the rust buffer
        // SAFETY: We must ensure the JS side reads SYNCHRONOUSLY and does not detach the buffer.
        // FileSystemSyncAccessHandle.read is synchronous.
        let view = unsafe { Uint8Array::view(buf) };
        
        // Prepare options: { at: self.pos }
        let options = js_sys::Object::new();
        
        js_sys::Reflect::set(
            &options, 
            &JsValue::from_str("at"), 
            &JsValue::from_f64(self.pos as f64)
        ).map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;

        let bytes_read = self.handle.read_with_options(&view, &options)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;

        let bytes_read_usize = bytes_read as usize;
        self.pos += bytes_read_usize as u64;
        
        Ok(bytes_read_usize)
    }
}

impl Seek for OpfsReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(p) => p as i64,
            SeekFrom::End(p) => self.size as i64 + p,
            SeekFrom::Current(p) => self.pos as i64 + p,
        };

        if new_pos < 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Invalid seek to negative position"));
        }

        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

// Ensure handle is closed when dropped
impl Drop for OpfsReader {
    fn drop(&mut self) {
        let _ = self.handle.close();
    }
}
