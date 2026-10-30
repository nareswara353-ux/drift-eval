use drift_core::{DriftError, Result};

pub struct ZeroCopySlice<'a> {
    data: &'a [f64],
    offset: usize,
    length: usize,
}

impl<'a> ZeroCopySlice<'a> {
    pub fn new(data: &'a [f64]) -> Self {
        Self {
            data,
            offset: 0,
            length: data.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn as_slice(&self) -> &'a [f64] {
        self.data
            .get(self.offset..self.offset + self.length)
            .unwrap_or(&[])
    }

    pub fn subview(&self, start: usize, len: usize) -> Result<Self> {
        let end = start
            .checked_add(len)
            .ok_or_else(|| DriftError::Invariant("zero-copy subview overflow".to_string()))?;
        if end > self.length {
            return Err(DriftError::DimensionMismatch {
                expected: self.length,
                actual: end,
            });
        }
        Ok(Self {
            data: self.data,
            offset: self.offset + start,
            length: len,
        })
    }

    pub fn chunks(&self, chunk_size: usize) -> Result<Vec<ZeroCopySlice<'a>>> {
        if chunk_size == 0 {
            return Err(DriftError::InvalidChunkSize(0));
        }
        let mut chunks = Vec::with_capacity(self.length / chunk_size + 1);
        let mut start = 0usize;
        while start < self.length {
            let len = (self.length - start).min(chunk_size);
            chunks.push(self.subview(start, len)?);
            start += len;
        }
        Ok(chunks)
    }
}

pub fn borrow_slice(data: &[f64]) -> ZeroCopySlice<'_> {
    ZeroCopySlice::new(data)
}
