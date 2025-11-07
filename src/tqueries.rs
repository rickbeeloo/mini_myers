/// Transposed query representation for efficient SIMD batch processing.
///
/// This structure stores queries in a transposed format.
/// It also precomputes the `peq` (position-equivalent) bitvectors for each nucleotide type.
///
/// # Examples
///
/// ```rust
/// use mini_myers::backend::U32;
/// use mini_myers::TQueries;
///
/// let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
/// let transposed = TQueries::<U32>::new(&queries);
/// assert_eq!(transposed.n_queries, 2);
/// assert_eq!(transposed.query_length, 3);
/// ```
use crate::backend::SimdBackend;
use crate::constant::{INVALID_IUPAC, IUPAC_MASKS};
use std::marker::PhantomData;
use wide::u8x32;

#[derive(Debug, Clone)]
pub struct TQueries<B: SimdBackend> {
    /// SIMD vectors representing transposed queries, one vector per position per block
    /// Layout: vectors[position][block_idx] where each block contains up to 32 queries
    pub vectors: Vec<Vec<u8x32>>,
    /// Length of each query (all queries must have the same length)
    pub query_length: usize,
    /// Number of queries
    pub n_queries: usize,
    /// Number of blocks of 32 queries needed
    pub n_blocks: usize,
    /// Precomputed peq bitvectors keyed by IUPAC mask (0..=15)
    /// Layout: peq_masks[iupac_mask][query_idx]
    pub peq_masks: [Vec<u64>; IUPAC_MASKS],
    /// Precomputed peq bitvectors for each IUPAC mask
    /// Layout: peqs[iupac_mask][simd_vector_idx] where each SIMD vector contains `B::LANES` lanes
    pub peqs: [Vec<B::Simd>; IUPAC_MASKS],
    _marker: PhantomData<B>,
}

/// Build peqs vectors from precomputed masks.
#[inline(always)]
fn build_peqs_vectors<B: SimdBackend>(
    peq_masks: &[Vec<u64>; IUPAC_MASKS],
    nq: usize,
    vectors_in_block: usize,
) -> [Vec<B::Simd>; IUPAC_MASKS] {
    let mut peqs: [Vec<B::Simd>; IUPAC_MASKS] =
        std::array::from_fn(|_| Vec::with_capacity(vectors_in_block));

    for v in 0..vectors_in_block {
        let base = v * B::LANES;
        for mask_idx in 0..IUPAC_MASKS {
            let mut lane = B::LaneArray::default();
            let lane_slice = lane.as_mut();
            let mask_vec = &peq_masks[mask_idx];
            for lane_idx in 0..B::LANES {
                let qi = base + lane_idx;
                if qi < nq {
                    lane_slice[lane_idx] = B::mask_word_to_scalar(mask_vec[qi]);
                }
            }
            peqs[mask_idx].push(B::from_array(lane));
        }
    }

    peqs
}

impl<B: SimdBackend> TQueries<B> {
    /// Creates a new `TQueries` structure from a slice of query sequences.
    ///
    /// This method transposes the queries and precomputes the peq bitvectors
    /// for the Myers algorithm. All queries must have the same length (bounded by the backend limb size).
    ///
    /// # Arguments
    ///
    /// * `queries` - A slice of byte vectors, where each vector represents one query sequence.
    ///   Each query must contain valid IUPAC nucleotide codes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mini_myers::backend::U32;
    /// use mini_myers::TQueries;
    ///
    /// let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
    /// let transposed = TQueries::<U32>::new(&queries);
    /// ```
    pub fn new(queries: &[Vec<u8>]) -> Self {
        assert!(!queries.is_empty(), "No queries provided");
        let query_length = queries[0].len();
        assert!(
            queries.iter().all(|q| q.len() == query_length),
            "All queries must have the same length"
        );

        let n_queries = queries.len();
        assert!(query_length > 0, "Query length must be greater than zero");
        assert!(
            query_length <= B::LIMB_BITS,
            "Query length must be <= {} for this backend",
            B::LIMB_BITS
        );

        // Calculate how many blocks of 32 queries we need
        let n_blocks = n_queries.div_ceil(32);

        // Initialize vector data: [position][block][query_in_block]
        let mut vector_data: Vec<Vec<[u8; 32]>> = vec![vec![[0u8; 32]; n_blocks]; query_length];
        let mut peq_masks: [Vec<u64>; IUPAC_MASKS] = std::array::from_fn(|_| vec![0u64; n_queries]);

        for (qi, q) in queries.iter().enumerate() {
            let block_idx = qi / 32;
            let idx_in_block = qi % 32;

            for (pos, &raw_c) in q.iter().enumerate() {
                let encoded = crate::iupac::get_encoded(raw_c);
                assert!(
                    encoded != INVALID_IUPAC,
                    "Query at index {} contains invalid IUPAC character: {:?}",
                    qi,
                    raw_c as char
                );

                vector_data[pos][block_idx][idx_in_block] = raw_c.to_ascii_uppercase();
                let bit = 1u64 << pos;
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

        // Convert to SIMD vectors
        let vectors: Vec<Vec<u8x32>> = vector_data
            .into_iter()
            .map(|pos_blocks| pos_blocks.into_iter().map(u8x32::new).collect())
            .collect();

        let vectors_in_block = n_queries.div_ceil(B::LANES);
        let peqs = build_peqs_vectors::<B>(&peq_masks, n_queries, vectors_in_block);

        Self {
            vectors,
            query_length,
            n_queries,
            n_blocks,
            peq_masks,
            peqs,
            _marker: PhantomData,
        }
    }
}
