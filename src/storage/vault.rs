use wasm_bindgen::prelude::*;
use std::io::{self, Read, Seek, SeekFrom};
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use pbkdf2::pbkdf2_hmac;
use sha2::Sha256;
use js_sys::Uint8Array;
use crate::storage::OpfsReader;
use crate::storage::FileSystemSyncAccessHandle;

const BLOCK_SIZE: usize = 4096;
const METADATA_SIZE: usize = 64; // IV (12) + TAG (16) + Padding (36)
const PHYSICAL_BLOCK_SIZE: usize = BLOCK_SIZE + METADATA_SIZE;
const SALT_SIZE: usize = 16;

pub struct VaultReader {
    reader: OpfsReader,
    key: [u8; 32],
    logical_pos: u64,
    logical_size: u64,
}

impl VaultReader {
    pub fn new(reader: OpfsReader, password: &str) -> io::Result<Self> {
        let mut salt = [0u8; SALT_SIZE];
        let mut r = reader;
        
        // Read salt from beginning of file
        r.seek(SeekFrom::Start(0))?;
        r.read_exact(&mut salt)?;

        // Derive key
        let mut key = [0u8; 32];
        pbkdf2_hmac::<Sha256>(password.as_bytes(), &salt, 100_000, &mut key);

        let physical_size = r.size();
        let logical_size = if physical_size < SALT_SIZE as u64 {
            0
        } else {
            ((physical_size - SALT_SIZE as u64) / PHYSICAL_BLOCK_SIZE as u64) * BLOCK_SIZE as u64
        };

        Ok(Self {
            reader: r,
            key,
            logical_pos: 0,
            logical_size,
        })
    }
}

impl Read for VaultReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.logical_pos >= self.logical_size {
            return Ok(0);
        }

        let block_index = self.logical_pos / BLOCK_SIZE as u64;
        let offset_in_block = (self.logical_pos % BLOCK_SIZE as u64) as usize;
        let physical_pos = SALT_SIZE as u64 + (block_index * PHYSICAL_BLOCK_SIZE as u64);

        // Read the entire physical block
        let mut physical_block = [0u8; PHYSICAL_BLOCK_SIZE];
        self.reader.seek(SeekFrom::Start(physical_pos))?;
        let bytes_read = self.reader.read(&mut physical_block)?;

        if bytes_read < METADATA_SIZE {
            return Ok(0);
        }

        // Decrypt
        let cipher = Aes256Gcm::new_from_slice(&self.key)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            
        let nonce = Nonce::from_slice(&physical_block[0..12]);
        let ciphertext = &physical_block[METADATA_SIZE..bytes_read];
        // Note: The tag is expected by aes-gcm at the end of the ciphertext usually, 
        // but it depends on the implementation. libaes-gcm standard is [12 IV][Ciphertext][16 Tag].
        // Let's adjust our layout: [12 IV][16 Tag][4096 Ciphertext] -> Actually lib expects [Ciphertext][Tag].
        
        // Let's use: [12 IV][16 Tag][4096 Data]
        let tag = &physical_block[12..28];
        let mut payload = ciphertext.to_vec();
        payload.extend_from_slice(tag);

        let decrypted = cipher.decrypt(nonce, payload.as_ref())
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Decryption failed: {}", e)))?;

        // Copy needed portion to output buffer
        let to_copy = std::cmp::min(buf.len(), BLOCK_SIZE - offset_in_block);
        let end = offset_in_block + to_copy;
        buf[..to_copy].copy_from_slice(&decrypted[offset_in_block..end]);

        self.logical_pos += to_copy as u64;
        Ok(to_copy)
    }
}

impl Seek for VaultReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(p) => p as i64,
            SeekFrom::End(p) => self.logical_size as i64 + p,
            SeekFrom::Current(p) => self.logical_pos as i64 + p,
        };

        if new_pos < 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Invalid seek"));
        }

        self.logical_pos = new_pos as u64;
        Ok(self.logical_pos)
    }
}

#[wasm_bindgen]
pub struct VaultWriter {
    handle: FileSystemSyncAccessHandle,
    key: [u8; 32],
}

#[wasm_bindgen]
impl VaultWriter {
    pub fn open(handle: FileSystemSyncAccessHandle, password: &str, salt: Vec<u8>) -> Result<VaultWriter, JsValue> {
        let mut s = [0u8; 16];
        if salt.len() != 16 { return Err(JsValue::from_str("Salt must be 16 bytes")); }
        s.copy_from_slice(&salt);
        
        Self::new(handle, password, &s)
            .map_err(|e| JsValue::from_str(&format!("VaultWriter Init Error: {:?}", e)))
    }

    pub fn write_chunk(&mut self, offset: f64, data: Uint8Array) -> Result<usize, JsValue> {
        let buf = data.to_vec();
        self.write_block(offset as u64, &buf)
            .map_err(|e| JsValue::from_str(&format!("VaultWriter Write Error: {:?}", e)))
    }
}

impl VaultWriter {
    pub fn new(handle: FileSystemSyncAccessHandle, password: &str, salt: &[u8; 16]) -> io::Result<Self> {
        // Derive key
        let mut key = [0u8; 32];
        pbkdf2_hmac::<Sha256>(password.as_bytes(), salt, 100_000, &mut key);

        // Write salt to beginning if file is new
        if handle.get_size().unwrap_or(0.0) == 0.0 {
            let view = unsafe { Uint8Array::view(salt) };
            let options = js_sys::Object::new();
            js_sys::Reflect::set(&options, &"at".into(), &0.0.into()).unwrap();
            handle.write_with_options(&view, &options).map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;
        }

        Ok(Self { handle, key })
    }

    pub fn write_block(&mut self, logical_offset: u64, data: &[u8]) -> io::Result<usize> {
        if data.len() != BLOCK_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Data must be exactly 4KB"));
        }

        let block_index = logical_offset / BLOCK_SIZE as u64;
        let physical_pos = SALT_SIZE as u64 + (block_index * PHYSICAL_BLOCK_SIZE as u64);

        // Generate IV
        let mut iv = [0u8; 12];
        getrandom::getrandom(&mut iv).map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        
        // Encrypt
        let cipher = Aes256Gcm::new_from_slice(&self.key)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        let nonce = Nonce::from_slice(&iv);
        
        let encrypted = cipher.encrypt(nonce, data)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        // Layout: [12 IV][16 TAG (at end of 'encrypted')][4096 Ciphertext]
        // Wait, 'encrypted' from AesGcm is [Ciphertext][Tag].
        // Our VaultReader expects [12 IV][16 Tag][4096 Ciphertext].
        
        let mut physical_block = Vec::with_capacity(PHYSICAL_BLOCK_SIZE);
        physical_block.extend_from_slice(&iv);
        
        let ciphertext = &encrypted[0..BLOCK_SIZE];
        let tag = &encrypted[BLOCK_SIZE..];
        
        physical_block.extend_from_slice(tag);
        physical_block.extend_from_slice(ciphertext);
        
        // Padding to 64 bytes overhead
        while physical_block.len() < PHYSICAL_BLOCK_SIZE {
            physical_block.push(0);
        }

        let view = unsafe { Uint8Array::view(&physical_block) };
        let options = js_sys::Object::new();
        js_sys::Reflect::set(&options, &"at".into(), &(physical_pos as f64).into()).unwrap();
        
        self.handle.write_with_options(&view, &options)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;

        Ok(BLOCK_SIZE)
    }
}
