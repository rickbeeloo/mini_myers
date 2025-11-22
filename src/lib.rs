pub mod constant {
    pub const IUPAC_MASKS: usize = 16;
    pub const INVALID_IUPAC: u8 = 255;
}

pub mod backend;
mod iupac;
pub mod search;
pub mod search_old;
pub mod tqueries;

// Re-export commonly used items at the crate root
pub use backend::{I16x16Backend, I32x8Backend, I64x4Backend, SimdBackend, U16, U32, U64};
pub use search::Searcher;
pub use tqueries::TQueries;
