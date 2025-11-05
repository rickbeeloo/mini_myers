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
use crate::constant::*;
use wide::{i32x8, u8x32};
#[cfg(not(feature = "latest_wide"))]
use wide_v07 as wide;
#[cfg(feature = "latest_wide")]
use wide_v08 as wide;

#[derive(Debug, Clone)]
pub struct TQueries {
    /// SIMD vectors representing transposed queries, one vector per position
    pub vectors: Vec<u8x32>,
    /// Length of each query (all queries must have the same length)
    pub query_length: usize,
    /// Number of queries (must be ≤32)
    pub n_queries: usize,
    /// Precomputed peq bitvectors keyed by IUPAC mask (0..=15)
    pub peq_masks: [Vec<u32>; IUPAC_MASKS],
    /// Precomputed peq bitvectors for each IUPAC mask
    pub peqs: [Vec<i32x8>; IUPAC_MASKS],
}

/// Build peqs vectors from precomputed masks.
#[inline(always)]
fn build_peqs_vectors(
    peq_masks: &[Vec<u32>; IUPAC_MASKS],
    nq: usize,
    vectors_in_block: usize,
) -> [Vec<i32x8>; IUPAC_MASKS] {
    let mut peqs: [Vec<i32x8>; IUPAC_MASKS] =
        std::array::from_fn(|_| Vec::with_capacity(vectors_in_block));

    for v in 0..vectors_in_block {
        let base = v * SIMD_LANES;
        for mask_idx in 0..IUPAC_MASKS {
            let mut lane = [0i32; SIMD_LANES];
            let mask_vec = &peq_masks[mask_idx];
            for (lane_idx, lane_slot) in lane.iter_mut().enumerate() {
                let qi = base + lane_idx;
                if qi < nq {
                    *lane_slot = mask_vec[qi] as i32;
                }
            }
            peqs[mask_idx].push(i32x8::new(lane));
        }
    }

    peqs
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
                let encoded = crate::iupac::get_encoded(raw_c);
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

        let vectors = vector_data.into_iter().map(u8x32::new).collect();

        let nq = n_queries;
        let vectors_in_block = nq.div_ceil(SIMD_LANES);
        let peqs = build_peqs_vectors(&peq_masks, nq, vectors_in_block);

        Self {
            vectors,
            query_length,
            n_queries,
            peq_masks,
            peqs,
        }
    }
}
