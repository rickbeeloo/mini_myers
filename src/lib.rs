//! # mini_myers
//!
//! SIMD implementation of the Myers bitvector algorithm specifically to test
//! whether short queries (<=32 nucleotides) are present in a longer DNA sequence with at most `k` edits.
//!
//! ## Features
//!
//! - **SIMD-accelerated**: Uses portable SIMD for parallel processing of multiple queries, number of lanes depends on the length of the queries.
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
//! let result = mini_search(&transposed, target, 4);
//!
//! // Result: [0, 1] means ATG has 0 edits, TTG has 1 edit
//! assert_eq!(result, vec![0, 1]);
//! ```
//!
//! ## Performance
//!
//! For 32 queries of length 24 in a 100K DNA string with k=4:
//! - mini_myers: ~23 µs/query
//! - See benchmarks for detailed comparisons

#![feature(portable_simd)]
use core::simd::Simd;
use std::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use std::simd::{LaneCount, SupportedLaneCount};

const IUPAC_MASKS: usize = 16;

/// Transposed query representation for efficient SIMD batch processing.
///
/// This structure stores queries in a transposed format.
/// It also precomputes the `peq` (position-equivalent) bitvectors for each nucleotide type.
///
/// # Examples
///
/// ```rust
/// use mini_myers::TQueries;
///
/// let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
/// let transposed = TQueries::new(&queries);
/// assert_eq!(transposed.n_queries, 2);
/// assert_eq!(transposed.query_length, 3);
/// ```
#[derive(Debug, Clone)]
pub struct TQueries {
    /// SIMD vectors representing transposed queries, one vector per position
    pub vectors: Vec<Simd<u8, 32>>,
    /// Length of each query (all queries must have the same length)
    pub query_length: usize,
    /// Number of queries (must be ≤32)
    pub n_queries: usize,
    /// Precomputed peq bitvectors keyed by IUPAC mask (0..=15)
    pub peq_masks: [Vec<u32>; IUPAC_MASKS],
}

impl TQueries {
    /// Creates a new `TQueries` structure from a slice of query sequences.
    ///
    /// This method transposes the queries and precomputes the peq bitvectors
    /// for the Myers algorithm. All queries must have the same length, and the number
    /// of queries must not exceed 32 (but can be less)
    ///
    /// # Arguments
    ///
    /// * `queries` - A slice of byte vectors, where each vector represents one query sequence.
    ///   Each query must contain valid IUPAC nucleotide codes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mini_myers::TQueries;
    ///
    /// let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
    /// let transposed = TQueries::new(&queries);
    /// ```
    pub fn new(queries: &[Vec<u8>]) -> Self {
        assert!(!queries.is_empty(), "No queries provided");
        let query_length = queries[0].len();
        assert!(
            queries.iter().all(|q| q.len() == query_length),
            "All queries must have the same length"
        );

        let n_queries = queries.len();
        assert!(n_queries <= 32, "Number of queries exceeds 32");

        assert!(
            query_length > 0 && query_length <= 32,
            "Query length must be 1..=32"
        );

        let mut vector_data: Vec<[u8; 32]> = vec![[0u8; 32]; query_length];
        let mut peq_masks: [Vec<u32>; IUPAC_MASKS] = std::array::from_fn(|_| vec![0u32; n_queries]);

        for (qi, q) in queries.iter().enumerate() {
            for (pos, &raw_c) in q.iter().enumerate() {
                let encoded = get_encoded(raw_c);
                assert!(
                    encoded != INVALID_IUPAC,
                    "Query at index {} contains invalid IUPAC character: {:?}",
                    qi,
                    raw_c as char
                );

                vector_data[pos][qi] = raw_c.to_ascii_uppercase();
                let bit = 1u32 << pos;
                if encoded != 0 {
                    for (mask_idx, mask_vec) in peq_masks.iter_mut().enumerate().skip(1) {
                        let mask = mask_idx as u8;
                        if (encoded & mask) != 0 {
                            mask_vec[qi] |= bit;
                        }
                    }
                }
            }
        }

        let vectors = vector_data.into_iter().map(Simd::from_array).collect();

        Self {
            vectors,
            query_length,
            n_queries,
            peq_masks,
        }
    }
}

/// Searches for all queries in the target sequence using the Myers algorithm.
/// Returning the minimum edits found for the query in the entire target, or
/// `-1` if below the provided maximum edit distance `k`.
///
/// The number of used SIMD lanes depends on the length of the queries:
/// - Queries ≤16 nucleotides: uses 16 lanes
/// - Queries >16 nucleotides: uses 8 lanes
///
/// # Arguments
///
/// * `transposed` - A `TQueries` structure containing the preprocessed queries
/// * `target` - The target DNA sequence to search in (as a byte slice)
/// * `k` - Maximum edit distance threshold.
///
/// # Returns
///
/// A vector of `i32` values, one per query, containing:
/// - The minimum edit distance found (0 to k) if a match within threshold is found
/// - `-1` if no match within the threshold k was found
///
///
///
/// # Examples
///
/// ```rust
/// use mini_myers::{TQueries, mini_search};
///
/// let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
/// let transposed = TQueries::new(&queries);
/// let target = b"CCCTCGCCCCCCATGCCCCC";
///
/// // Search with k=4 (allow up to 4 edits)
/// let result = mini_search(&transposed, target, 4);
/// // Result: [0, 1] - ATG found with 0 edits, TTG found with 1 edit
///
/// ```
#[inline(always)]
pub fn mini_search(transposed: &TQueries, target: &[u8], k: u8) -> Vec<i32> {
    let nq = transposed.n_queries;
    if nq == 0 {
        return Vec::new();
    }
    let m = transposed.query_length;
    assert!(m > 0 && m <= 32, "query length must be 1..=32");

    // Use i32 throughout (like backup.rs), but with more lanes for short queries
    if m <= 16 {
        search_generic::<16>(transposed, target, k)
    } else {
        search_generic::<8>(transposed, target, k)
    }
}

/// Build peqs vectors from precomputed masks.
#[inline(always)]
fn build_peqs_vectors<const LANES: usize>(
    peq_masks: &[Vec<u32>; IUPAC_MASKS],
    nq: usize,
    vectors_in_block: usize,
) -> [Vec<Simd<i32, LANES>>; IUPAC_MASKS]
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut peqs: [Vec<Simd<i32, LANES>>; IUPAC_MASKS] =
        std::array::from_fn(|_| Vec::with_capacity(vectors_in_block));

    for v in 0..vectors_in_block {
        let base = v * LANES;
        for mask_idx in 0..IUPAC_MASKS {
            let mut lane = [0i32; LANES];
            let mask_vec = &peq_masks[mask_idx];
            for (lane_idx, lane_slot) in lane.iter_mut().enumerate() {
                let qi = base + lane_idx;
                if qi < nq {
                    *lane_slot = mask_vec[qi] as i32;
                }
            }
            peqs[mask_idx].push(Simd::from_array(lane));
        }
    }

    peqs
}

#[inline(always)]
fn search_generic<const LANES: usize>(transposed: &TQueries, target: &[u8], k: u8) -> Vec<i32>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    // todo: would be nice to also extract this to transposedQueries so we only
    // have to build the peq table once
    let nq = transposed.n_queries;
    let m = transposed.query_length;

    let peq_masks = &transposed.peq_masks;

    let vectors_in_block = nq.div_ceil(LANES);
    //println!("Using {} vectors in block", vectors_in_block);

    let peqs = build_peqs_vectors(peq_masks, nq, vectors_in_block);

    let all_ones = Simd::<i32, LANES>::splat(!0i32);
    let zero_v = Simd::<i32, LANES>::splat(0i32);
    let one_v = Simd::<i32, LANES>::splat(1i32);

    let mut pv_vec: Vec<Simd<i32, LANES>> = vec![all_ones; vectors_in_block];
    let mut mv_vec: Vec<Simd<i32, LANES>> = vec![zero_v; vectors_in_block];
    let mut scores_vec: Vec<Simd<i32, LANES>> =
        vec![Simd::<i32, LANES>::splat(m as i32); vectors_in_block];

    let max_possible = (m + target.len()) as i32;
    let mut min_scores_vec: Vec<Simd<i32, LANES>> =
        vec![Simd::<i32, LANES>::splat(max_possible); vectors_in_block];

    let high_bit: u32 = 1u32 << (m - 1);
    let mask_vec = Simd::<i32, LANES>::splat(high_bit as i32);

    let target_len = target.len();
    for t_idx in 0..target_len {
        let tb = unsafe { target.get_unchecked(t_idx) };
        let encoded = get_encoded(*tb);
        if encoded == INVALID_IUPAC {
            panic!("Target contains invalid IUPAC character: {:?}", *tb as char);
        }
        let peq_slice = unsafe { peqs.get_unchecked(encoded as usize) };
        for v in 0..vectors_in_block {
            // Regular Myers
            let eq = unsafe { peq_slice.get_unchecked(v) };
            let pv = pv_vec[v];
            let mv = mv_vec[v];
            let xv = eq | mv;
            let xh = (((eq & pv) + pv) ^ pv) | eq;
            let ph = mv | !(xh | pv);
            let mh = pv & xh;
            // Track edit distance cost
            let ph_bit = (ph & mask_vec).simd_ne(zero_v).to_int() & one_v;
            let mh_bit = (mh & mask_vec).simd_ne(zero_v).to_int() & one_v;
            let ph_shift = ph << 1;
            let new_pv = (mh << 1) | !(xv | ph_shift);
            let new_mv = ph_shift & xv;

            unsafe {
                *pv_vec.get_unchecked_mut(v) = new_pv;
                *mv_vec.get_unchecked_mut(v) = new_mv;
                let new_score = *scores_vec.get_unchecked(v) + (ph_bit - mh_bit);
                *scores_vec.get_unchecked_mut(v) = new_score;
                *min_scores_vec.get_unchecked_mut(v) =
                    min_scores_vec.get_unchecked(v).simd_min(new_score);
            }
        }
    }
    let k_v = Simd::<i32, LANES>::splat(k as i32);
    let neg1_v = Simd::<i32, LANES>::splat(-1);
    let mut result = vec![-1i32; nq];

    for v in 0..vectors_in_block {
        let min_score = unsafe { *min_scores_vec.get_unchecked(v) };
        // we check for leq k, all positions got init to max edits as t.len() + m, which then become -1 here
        let selected = min_score.simd_le(k_v).select(min_score, neg1_v);
        let base = v * LANES;
        let end = (base + LANES).min(nq);
        result[base..end].copy_from_slice(&selected.to_array()[..end - base]);
    }
    //println!("Time taken: {:?}", end_time.duration_since(start_time));
    result
}

const INVALID_IUPAC: u8 = 255;

#[inline(always)]
fn get_encoded(c: u8) -> u8 {
    IUPAC_CODE[(c & 0x1F) as usize]
}

// Based on sassy: https://github.com/RagnarGrootKoerkamp/sassy/blob/master/src/profiles/iupac.rs#L258
#[rustfmt::skip]
const IUPAC_CODE: [u8; 32] = {
    let mut t = [INVALID_IUPAC; 32];
    const A: u8 = 1 << 0;
    const C: u8 = 1 << 1;
    const T: u8 = 1 << 2;
    const G: u8 = 1 << 3;

    t[b'A' as usize & 0x1F] = A;
    t[b'C' as usize & 0x1F] = C;
    t[b'T' as usize & 0x1F] = T;
    t[b'U' as usize & 0x1F] = T;
    t[b'G' as usize & 0x1F] = G;
    t[b'N' as usize & 0x1F] = A | C | T | G;

    t[b'R' as usize & 0x1F] = A | G;
    t[b'Y' as usize & 0x1F] = C | T;
    t[b'S' as usize & 0x1F] = G | C;
    t[b'W' as usize & 0x1F] = A | T;
    t[b'K' as usize & 0x1F] = G | T;
    t[b'M' as usize & 0x1F] = A | C;
    t[b'B' as usize & 0x1F] = C | G | T;
    t[b'D' as usize & 0x1F] = A | G | T;
    t[b'H' as usize & 0x1F] = A | C | T;
    t[b'V' as usize & 0x1F] = A | C | G;

    t[b'X' as usize & 0x1F] = 0;

    t
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iupac_query_matches_standard_target() {
        let queries = vec![b"n".to_vec()];
        let transposed = TQueries::new(&queries);
        assert_eq!(mini_search(&transposed, b"A", 0), vec![0]);
        assert_eq!(mini_search(&transposed, b"a", 0), vec![0]);
    }

    #[test]
    fn test_iupac_target_matches_standard_query() {
        let queries = vec![b"A".to_vec()];
        let transposed = TQueries::new(&queries);
        assert_eq!(mini_search(&transposed, b"N", 0), vec![0]);
        assert_eq!(mini_search(&transposed, b"n", 0), vec![0]);
    }

    #[test]
    fn test_iupac_mismatch_requires_edit() {
        let queries = vec![b"R".to_vec()];
        let transposed = TQueries::new(&queries);
        assert_eq!(mini_search(&transposed, b"C", 0), vec![-1]);
        assert_eq!(mini_search(&transposed, b"C", 1), vec![1]);
        assert_eq!(mini_search(&transposed, b"c", 1), vec![1]);
    }

    #[test]
    #[should_panic(expected = "invalid IUPAC character")]
    fn test_invalid_query_panics() {
        let queries = vec![b"AZ".to_vec()];
        let _ = TQueries::new(&queries);
    }

    #[test]
    fn test_mini_search() {
        let q = b"ATG".to_vec();
        let queries = vec![q];
        let t = b"CCCCCCCCCATGCCCCC";
        let transposed = TQueries::new(&queries);
        let result = mini_search(&transposed, t, 4);
        assert_eq!(result, vec![0]); // Lowest edits, 0 for q at idx 0
    }

    #[test]
    fn test_double_search() {
        let q1 = b"ATG".to_vec();
        let q2 = b"TTG".to_vec();
        let queries = vec![q1, q2];
        let t = b"CCCTTGCCCCCCATGCCCCC";
        let transposed = TQueries::new(&queries);
        let result = mini_search(&transposed, t, 4);
        assert_eq!(result, vec![0, 0]);
    }

    #[test]
    fn test_edit() {
        let q1 = b"ATCAGA";
        let queries = vec![q1.to_vec()];
        let t = b"ATCTGA"; // 1 edit
        let transposed = TQueries::new(&queries);
        let result = mini_search(&transposed, t, 4);
        assert_eq!(result, vec![1]);
        let t = b"GTCTGA"; // 2 edits
        let result = mini_search(&transposed, t, 4);
        assert_eq!(result, vec![2]);
        let t = b"GTTGA"; // 3 edits (1 del)
        let result = mini_search(&transposed, t, 4);
        assert_eq!(result, vec![3]);
        // Match should not be recovered when k == 1
        let result = mini_search(&transposed, t, 1);
        assert_eq!(result, vec![-1]);
    }

    #[test]
    fn test_lowest_edits_returned() {
        let q1 = b"GGGCATCGATGAC";
        let queries = vec![q1.to_vec()];
        let t = b"CCCCCCCGGGCATCGATGACCCCCCCCCCCCCCCGGGCTTCGATGAC";
        //                                                     ^^^^^^^^^^^^^ has one mutation
        let transposed = TQueries::new(&queries);
        let result = mini_search(&transposed, t, 4);
        assert_eq!(result, vec![0]);
    }

    #[test]
    #[should_panic(expected = "All queries must have the same length")]
    fn test_error_unequal_lengths() {
        let q1 = b"GGGCATCGATGAC";
        let q2 = b"AAA";
        let t = b"CCC";
        let queries = vec![q1.to_vec(), q2.to_vec()];
        let transposed = TQueries::new(&queries);
        let result = mini_search(&transposed, t, 4);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn read_example() {
        let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
        let transposed = TQueries::new(&queries);
        let target = b"CCCTCGCCCCCCATGCCCCC";
        let result = mini_search(&transposed, target, 4);
        println!("Result: {:?}", result);
    }
}
