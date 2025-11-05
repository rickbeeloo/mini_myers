//! # mini_myers
//!
//! SIMD implementation of the Myers bitvector algorithm specifically to test
//! whether short queries (<=32 nucleotides) are present in a longer DNA sequence with at most `k` edits.
//!
//! ## Features
//!
//! - **SIMD-accelerated**: Uses the `wide` crate for stable SIMD processing of multiple queries.
//! - **Batch processing**: Process up to 32 queries simultaneously
//!
//! ## When to use
//!
//! - Short queries (≤32 nucleotides)
//! - Multiple queries to search (best performance with multiples of 8)
//! - Only need edit distance, not positions
//! - Supports DNA IUPAC codes (A, C, G, T, N, etc)
//!
//! ## Example
//!
//! ```rust
//! use mini_myers::{TQueries, mini_search};
//!
//! // Prepare queries
//! let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
//! let transposed = TQueries::new(&queries);
//!
//! // Search in target sequence
//! let target = b"CCCTCGCCCCCCATGCCCCC";
//! let result = mini_search(&transposed, target, 4, None);
//!
//! // Result: [0, 1] means ATG has 0 edits, TTG has 1 edit
//! assert_eq!(result, vec![0.0, 1.0]);
//! ```
//!
//! ## Performance
//!
//! For 32 queries of length 24 in a 100K DNA string with k=4:
//! - mini_myers: ~23 µs/query
//! - See benchmarks for detailed comparisons

// These traits are needed for both versions (cmp_eq/simd_eq and cmp_gt/simd_gt)

pub mod constant {
    pub const SIMD_LANES: usize = 8;
    pub const IUPAC_MASKS: usize = 16;
    pub const INVALID_IUPAC: u8 = 255;
}

mod iupac;
pub mod search;
pub mod tqueries;

// Re-export commonly used items at the crate root
pub use search::{mini_search, mini_search_with_positions, MatchInfo};
pub use tqueries::TQueries;
