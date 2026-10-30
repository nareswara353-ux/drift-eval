pub mod chunking;
pub mod executor;
pub mod memmap_loader;
pub mod zero_copy;

pub use chunking::{chunk_indices, ChunkPlan};
pub use executor::{Executor, ExecutorConfig};
pub use memmap_loader::{load_f64_slice, MappedF64};
pub use zero_copy::{borrow_slice, ZeroCopySlice};
