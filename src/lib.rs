//! # mini_myers
//!
//! SIMD implementation of the Myers bitvector algorithm specifically to test
//! whether short queries (<=64 nucleotides) are present in a longer DNA sequence with at most `k` edits.
//!
//! ## Features
//!
//! - **Batch processing**: Process 8 or 16 queries simultaneously depending on their length
//! - **Flexible backends**: Choose between `U32` (8 lanes) or `U64` (4 lanes) for different query lengths
//! - **Multiple modes**: Scan for minimum costs or find all match positions
//!
//! ## When to use
//!
//! - Short queries (≤32 nucleotides with U32 backend, ≤64 with U64)
//! - Multiple queries to search (best performance with multiples of 8)
//! - Need edit distance or match positions
//! - Supports DNA IUPAC codes (A, C, G, T, N, etc)
//!
//! ## Example
//!
//! ```rust
//! use mini_myers::{Searcher, TQueries};
//! use mini_myers::backend::{U32, U64};
//!
//! // Create a searcher with U32 backend
//! let mut searcher = Searcher::<U32>::new();
//! let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
//! let encoded = TQueries::<U32>::new(&queries, false);
//! let target = b"CCCTCGCCCCCCATGCCCCC";
//!
//! // Scan mode: get boolean results indicating if each query matches (within k edits)
//! let results = searcher.scan(&encoded, target, 0, None);
//! assert!(results[0]); // "ATG" matches
//! assert!(!results[1]); // "TTG" doesn't match
//!
//!
//! // Use U64 backend for longer queries (up to 64 nucleotides)
//! let mut searcher64 = Searcher::<U64>::new();
//! let encoded = TQueries::<U64>::new(&queries, false);
//! let results = searcher64.scan(&encoded, target, 4, None);
//! ```
//!
//! ## Performance
//!
//! For 32 queries of length 24 in a 100K DNA string with k=4:
//! - mini_myers: ~23 µs/query
//! - See benchmarks for detailed comparisons

// These traits are needed for both versions (cmp_eq/simd_eq and cmp_gt/simd_gt)

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
pub use backend::{I32x8Backend, I64x4Backend, SimdBackend, U32, U64};
pub use search::Searcher;
pub use tqueries::TQueries;
