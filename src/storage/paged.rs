use std::io::{self, Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use lru::LruCache;

const PAGE_SIZE: usize = 4096;
const CACHE_CAPACITY: usize = 1000; // 4MB cache

pub struct PagedReader {
    inner: Box<dyn ReadSeek>,
    cache: LruCache<u64, Vec<u8>>,
    pos: u64,
    size: u64,
}

pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

impl PagedReader {
    pub fn new(mut inner: Box<dyn ReadSeek>) -> io::Result<Self> {
        let size = inner.seek(SeekFrom::End(0))?;
        inner.seek(SeekFrom::Start(0))?;
        
        Ok(Self {
            inner,
            cache: LruCache::new(NonZeroUsize::new(CACHE_CAPACITY).unwrap()),
            pos: 0,
            size,
        })
    }

    fn read_page(&mut self, page_idx: u64) -> io::Result<&Vec<u8>> {
        if !self.cache.contains(&page_idx) {
            let page_start = page_idx * PAGE_SIZE as u64;
            self.inner.seek(SeekFrom::Start(page_start))?;
            
            let mut buffer = vec![0u8; PAGE_SIZE];
            let read = self.inner.read(&mut buffer)?;
            buffer.truncate(read);
            
            self.cache.put(page_idx, buffer);
        }
        
        Ok(self.cache.get(&page_idx).unwrap())
    }
}

impl Read for PagedReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.size {
            return Ok(0);
        }

        let mut bytes_read = 0;
        let mut buf_offset = 0;
        
        while buf_offset < buf.len() && self.pos < self.size {
            let page_idx = self.pos / PAGE_SIZE as u64;
            let page_offset = (self.pos % PAGE_SIZE as u64) as usize;
            
            let page = self.read_page(page_idx)?;
            
            if page_offset >= page.len() {
                break; // End of file inside a page boundary (should be handled by size check but just safety)
            }
            
            let available = page.len() - page_offset;
            let to_read = std::cmp::min(buf.len() - buf_offset, available);
            
            buf[buf_offset..buf_offset + to_read].copy_from_slice(&page[page_offset..page_offset + to_read]);
            
            self.pos += to_read as u64;
            buf_offset += to_read;
            bytes_read += to_read;
        }

        Ok(bytes_read)
    }
}

impl Seek for PagedReader {
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
