use std::fs::File;
use std::path::Path;

use drift_core::{DriftError, Result};
use memmap2::Mmap;

const F64_SIZE: usize = std::mem::size_of::<f64>();

pub struct MappedF64 {
    mmap: Mmap,
    len: usize,
}

impl MappedF64 {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(|e| DriftError::MemoryMap {
            path: path_ref.display().to_string(),
            source: e,
        })?;
        let metadata = file.metadata().map_err(|e| DriftError::MemoryMap {
            path: path_ref.display().to_string(),
            source: e,
        })?;
        let byte_len = metadata.len() as usize;
        if byte_len == 0 {
            return Err(DriftError::MemoryMap {
                path: path_ref.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "empty memory-mapped file",
                ),
            });
        }
        if byte_len % F64_SIZE != 0 {
            return Err(DriftError::MemoryMap {
                path: path_ref.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "file size is not a multiple of f64",
                ),
            });
        }
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| DriftError::MemoryMap {
            path: path_ref.display().to_string(),
            source: e,
        })?;
        Ok(Self {
            mmap,
            len: byte_len / F64_SIZE,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[f64] {
        let ptr = self.mmap.as_ptr() as *const f64;
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }
}

pub fn load_f64_slice(path: impl AsRef<Path>) -> Result<MappedF64> {
    MappedF64::open(path)
}
