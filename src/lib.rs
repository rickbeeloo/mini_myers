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

#[cfg(not(feature = "latest_wide"))]
use wide_v07 as wide;

#[cfg(feature = "latest_wide")]
use wide_v08 as wide;

// These traits are needed for both versions (cmp_eq/simd_eq and cmp_gt/simd_gt)
use wide::{i32x8, u8x32, CmpEq, CmpGt};

const SIMD_LANES: usize = 8;

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

/// Match information including position and cost for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchInfo {
    /// Index of the query (0-based)
    pub query_idx: usize,
    /// Minimum edit distance found (0 to k)
    pub cost: i32,
    /// Position in target where best match ends (0-based), or -1 if no match
    pub pos: i32,
}

/// Match information including position and fractional cost when using overhang penalties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchInfoOverhang {
    /// Index of the query (0-based)
    pub query_idx: usize,
    /// Minimum edit distance found with overhang penalty alpha applied
    pub cost: f32,
    /// Position in target where best match ends (0-based)
    pub pos: i32,
}

/// Searches for all queries in the target sequence using the Myers algorithm.
/// Returning the minimum edits found for the query in the entire target, or
/// `-1` if below the provided maximum edit distance `k`.
///
/// The SIMD backend currently relies on `wide::i32x8` (8 lanes) for batched processing.
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
    search_simd(transposed, target, k)
}

/// Searches for all queries in the target sequence using the Myers algorithm,
/// returning detailed match information including positions.
///
/// This function tracks ALL positions in the target where each query matches
/// within the threshold k. There is a small performance overhead (~2-5%) compared to
/// `mini_search` due to additional position tracking.
///
/// # Arguments
///
/// * `transposed` - A `TQueries` structure containing the preprocessed queries
/// * `target` - The target DNA sequence to search in (as a byte slice)
/// * `k` - Maximum edit distance threshold.
///
/// # Returns
///
/// A vector of `MatchInfo` structs containing all matches found:
/// - `query_idx`: The index of the query (0-based)
/// - `cost`: The edit distance at this position (0 to k)
/// - `pos`: The position in target where match ends (0-based)
///
/// Multiple entries may exist per query if it matches at multiple positions.
/// Results are ordered by position (earlier positions first).
///
/// # Examples
///
/// ```rust
/// use mini_myers::{TQueries, mini_search_with_positions};
///
/// let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
/// let transposed = TQueries::new(&queries);
/// let target = b"CCCTCGCCCCCCATGCCCCC";
/// let mut results = Vec::new();
///
/// let result = mini_search_with_positions(&transposed, target, 4, &mut results);
/// // Matches are now in results
/// ```
#[inline(always)]
pub fn mini_search_with_positions(
    transposed: &TQueries,
    target: &[u8],
    k: u8,
    results: &mut Vec<MatchInfo>,
) {
    let nq = transposed.n_queries;
    if nq == 0 {
        return;
    }
    let m = transposed.query_length;
    assert!(m > 0 && m <= 32, "query length must be 1..=32");
    search_simd_with_positions(transposed, target, k, results)
}

/// Searches for all queries using an overhang penalty `alpha` at the text boundaries.
///
/// Overhangs represent query characters that fall outside the target boundaries. They incur
/// a reduced cost of `alpha` per character (default `alpha = 1.0` restores the standard behaviour).
/// Returns the minimum edit distance for each query as a floating point value. A result of `-1.0`
/// indicates no match was found within the provided `k` threshold.
#[inline(always)]
pub fn mini_search_with_overhang(
    transposed: &TQueries,
    target: &[u8],
    k: u8,
    alpha: f32,
) -> Vec<f32> {
    let nq = transposed.n_queries;
    if nq == 0 {
        return Vec::new();
    }
    let m = transposed.query_length;
    assert!(m > 0 && m <= 32, "query length must be 1..=32");
    let alpha = alpha.clamp(0.0, 1.0);
    if alpha >= 0.999_999 {
        return search_simd(transposed, target, k)
            .into_iter()
            .map(|v| v as f32)
            .collect();
    }
    search_simd_with_overhang(transposed, target, k, alpha)
}

/// Searches for all queries using an overhang penalty `alpha`, returning all matches with positions.
#[inline(always)]
pub fn mini_search_with_positions_overhang(
    transposed: &TQueries,
    target: &[u8],
    k: u8,
    alpha: f32,
    results: &mut Vec<MatchInfoOverhang>,
) {
    let nq = transposed.n_queries;
    if nq == 0 {
        return;
    }
    let m = transposed.query_length;
    assert!(m > 0 && m <= 32, "query length must be 1..=32");
    let alpha = alpha.clamp(0.0, 1.0);
    if alpha >= 0.999_999 {
        let mut tmp = Vec::new();
        search_simd_with_positions(transposed, target, k, &mut tmp);
        results.extend(tmp.into_iter().map(|info| MatchInfoOverhang {
            query_idx: info.query_idx,
            cost: info.cost as f32,
            pos: info.pos,
        }));
        return;
    }
    search_simd_with_positions_overhang(transposed, target, k, alpha, results)
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

#[derive(Copy, Clone)]
struct OverhangColumnCtx {
    all_ones: i32x8,
    zero_v: i32x8,
    one_v: i32x8,
    mask_vec: i32x8,
    scale_shift: u32,
}

#[inline(always)]
fn apply_overhang_column(
    eq: i32x8,
    adjust_vec: i32x8,
    pv: &mut i32x8,
    mv: &mut i32x8,
    score: &mut i32x8,
    ctx: OverhangColumnCtx,
) -> i32x8 {
    let all_ones = ctx.all_ones;
    let zero_v = ctx.zero_v;
    let one_v = ctx.one_v;
    let mask_vec = ctx.mask_vec;

    let pv_val = *pv;
    let mv_val = *mv;
    let xv = eq | mv_val;
    let xh = (((eq & pv_val) + pv_val) ^ pv_val) | eq;
    let ph = mv_val | (all_ones ^ (xh | pv_val));
    let mh = pv_val & xh;

    #[cfg(not(feature = "latest_wide"))]
    let ph_bit_mask = (ph & mask_vec).cmp_eq(zero_v);
    #[cfg(feature = "latest_wide")]
    let ph_bit_mask = (ph & mask_vec).simd_eq(zero_v);

    let ph_bit = (all_ones ^ ph_bit_mask) & one_v;

    #[cfg(not(feature = "latest_wide"))]
    let mh_bit_mask = (mh & mask_vec).cmp_eq(zero_v);
    #[cfg(feature = "latest_wide")]
    let mh_bit_mask = (mh & mask_vec).simd_eq(zero_v);

    let mh_bit = (all_ones ^ mh_bit_mask) & one_v;
    let ph_shift = ph << 1;
    let new_pv = (mh << 1) | (all_ones ^ (xv | ph_shift));
    let new_mv = ph_shift & xv;

    *pv = new_pv;
    *mv = new_mv;

    let new_score = *score + (ph_bit - mh_bit);
    *score = new_score;

    let new_score_scaled = new_score << ctx.scale_shift;
    new_score_scaled - adjust_vec
}

#[inline(always)]
fn search_simd(transposed: &TQueries, target: &[u8], k: u8) -> Vec<i32> {
    // todo: would be nice to also extract this to transposedQueries so we only
    // have to build the peq table once
    let nq = transposed.n_queries;
    let m = transposed.query_length;

    let vectors_in_block = nq.div_ceil(SIMD_LANES);

    let all_ones = i32x8::splat(!0);
    let zero_v = i32x8::splat(0);
    let one_v = i32x8::splat(1);

    let mut pv_vec: Vec<i32x8> = vec![all_ones; vectors_in_block];
    let mut mv_vec: Vec<i32x8> = vec![zero_v; vectors_in_block];
    let mut scores_vec: Vec<i32x8> = vec![i32x8::splat(m as i32); vectors_in_block];

    let max_possible = (m + target.len()) as i32;
    let mut min_scores_vec: Vec<i32x8> = vec![i32x8::splat(max_possible); vectors_in_block];

    let high_bit: u32 = 1u32 << (m - 1);
    let mask_vec = i32x8::splat(high_bit as i32);

    for &tb in target {
        let encoded = get_encoded(tb);
        if encoded == INVALID_IUPAC {
            panic!("Target contains invalid IUPAC character: {:?}", tb as char);
        }
        let peq_slice = unsafe { transposed.peqs.get_unchecked(encoded as usize) };
        for v in 0..vectors_in_block {
            // Regular Myers
            let eq = unsafe { *peq_slice.get_unchecked(v) };
            let pv = pv_vec[v];
            let mv = mv_vec[v];
            let xv = eq | mv;
            let xh = (((eq & pv) + pv) ^ pv) | eq;
            let ph = mv | (all_ones ^ (xh | pv));
            let mh = pv & xh;

            // Track edit distance cost
            #[cfg(not(feature = "latest_wide"))]
            let ph_bit_mask = (ph & mask_vec).cmp_eq(zero_v); // wide 0.7, faster it seems
            #[cfg(feature = "latest_wide")]
            let ph_bit_mask = (ph & mask_vec).simd_eq(zero_v); // performance regression in wide 0.8.1

            let ph_bit = (all_ones ^ ph_bit_mask) & one_v;

            #[cfg(not(feature = "latest_wide"))]
            let mh_bit_mask = (mh & mask_vec).cmp_eq(zero_v); // wide 0.7, faster it seems
            #[cfg(feature = "latest_wide")]
            let mh_bit_mask = (mh & mask_vec).simd_eq(zero_v); // performance regression in wide 0.8.1

            let mh_bit = (all_ones ^ mh_bit_mask) & one_v;
            let ph_shift = ph << 1;
            let new_pv = (mh << 1) | (all_ones ^ (xv | ph_shift));
            let new_mv = ph_shift & xv;

            unsafe {
                *pv_vec.get_unchecked_mut(v) = new_pv;
                *mv_vec.get_unchecked_mut(v) = new_mv;
                let new_score = *scores_vec.get_unchecked(v) + (ph_bit - mh_bit);
                *scores_vec.get_unchecked_mut(v) = new_score;
                *min_scores_vec.get_unchecked_mut(v) =
                    min_scores_vec.get_unchecked(v).min(new_score);
            }
        }
    }
    let k_v = i32x8::splat(k as i32);
    let neg1_v = i32x8::splat(-1);
    let mut result = vec![-1i32; nq];

    for v in 0..vectors_in_block {
        let min_score = unsafe { *min_scores_vec.get_unchecked(v) };

        #[cfg(not(feature = "latest_wide"))]
        let mask = all_ones ^ min_score.cmp_gt(k_v); // wide 0.7, faster it seems
        #[cfg(feature = "latest_wide")]
        let mask = all_ones ^ min_score.simd_gt(k_v); // performance regression in wide 0.8.1

        let selected = mask.blend(min_score, neg1_v);
        let base = v * SIMD_LANES;
        let end = (base + SIMD_LANES).min(nq);
        let selected_arr = selected.to_array();
        result[base..end].copy_from_slice(&selected_arr[..end - base]);
    }
    result
}

#[inline(always)]
fn search_simd_with_overhang(transposed: &TQueries, target: &[u8], k: u8, alpha: f32) -> Vec<f32> {
    const SCALE_SHIFT: u32 = 8;
    const SCALE: i32 = 1 << SCALE_SHIFT; // 256

    let nq = transposed.n_queries;
    let m = transposed.query_length;
    let vectors_in_block = nq.div_ceil(SIMD_LANES);

    let alpha_scaled = ((alpha * (SCALE as f32)).round() as i32).clamp(0, SCALE);
    let extra_penalty_scaled = SCALE - alpha_scaled;
    let k_scaled = (k as i32) << SCALE_SHIFT;
    let inv_scale = 1.0f32 / (SCALE as f32);

    let all_ones = i32x8::splat(!0);
    let zero_v = i32x8::splat(0);
    let one_v = i32x8::splat(1);
    let zero_eq_vec = i32x8::splat(0);

    let mut pv_vec: Vec<i32x8> = vec![all_ones; vectors_in_block];
    let mut mv_vec: Vec<i32x8> = vec![zero_v; vectors_in_block];
    let mut scores_vec: Vec<i32x8> = vec![i32x8::splat(m as i32); vectors_in_block];

    let initial_adjusted = (m as i32) * alpha_scaled;
    let mut min_adjusted_vec: Vec<i32x8> = vec![i32x8::splat(initial_adjusted); vectors_in_block];

    let high_bit: u32 = 1u32 << (m - 1);
    let mask_vec = i32x8::splat(high_bit as i32);
    let k_scaled_vec = i32x8::splat(k_scaled);
    let ctx = OverhangColumnCtx {
        all_ones,
        zero_v,
        one_v,
        mask_vec,
        scale_shift: SCALE_SHIFT,
    };

    for (idx, &tb) in target.iter().enumerate() {
        let encoded = get_encoded(tb);
        if encoded == INVALID_IUPAC {
            panic!("Target contains invalid IUPAC character: {:?}", tb as char);
        }
        let overhang = m.saturating_sub(idx + 1) as i32;
        let adjust_vec = i32x8::splat(overhang * extra_penalty_scaled);
        let peq_slice = unsafe { transposed.peqs.get_unchecked(encoded as usize) };

        for v in 0..vectors_in_block {
            let eq = unsafe { *peq_slice.get_unchecked(v) };
            let adjusted = apply_overhang_column(
                eq,
                adjust_vec,
                unsafe { pv_vec.get_unchecked_mut(v) },
                unsafe { mv_vec.get_unchecked_mut(v) },
                unsafe { scores_vec.get_unchecked_mut(v) },
                ctx,
            );
            unsafe {
                let current_min = *min_adjusted_vec.get_unchecked(v);
                *min_adjusted_vec.get_unchecked_mut(v) = current_min.min(adjusted);
            }
        }
    }

    if alpha_scaled < SCALE {
        for trailing in 1..=m {
            let adjust_vec = i32x8::splat((trailing as i32) * extra_penalty_scaled);
            for v in 0..vectors_in_block {
                let adjusted = apply_overhang_column(
                    zero_eq_vec,
                    adjust_vec,
                    unsafe { pv_vec.get_unchecked_mut(v) },
                    unsafe { mv_vec.get_unchecked_mut(v) },
                    unsafe { scores_vec.get_unchecked_mut(v) },
                    ctx,
                );
                unsafe {
                    let current_min = *min_adjusted_vec.get_unchecked(v);
                    *min_adjusted_vec.get_unchecked_mut(v) = current_min.min(adjusted);
                }
            }
        }
    }

    let mut result = vec![-1.0f32; nq];
    let neg_mask = i32x8::splat(i32::MAX);

    for v in 0..vectors_in_block {
        let min_adj = unsafe { *min_adjusted_vec.get_unchecked(v) };

        #[cfg(not(feature = "latest_wide"))]
        let mask = all_ones ^ min_adj.cmp_gt(k_scaled_vec);
        #[cfg(feature = "latest_wide")]
        let mask = all_ones ^ min_adj.simd_gt(k_scaled_vec);

        let selected = mask.blend(min_adj, neg_mask);
        let base = v * SIMD_LANES;
        let end = (base + SIMD_LANES).min(nq);
        let selected_arr = selected.to_array();
        for (lane_idx, &val) in selected_arr[..(end - base)].iter().enumerate() {
            if val != i32::MAX {
                result[base + lane_idx] = (val as f32) * inv_scale;
            }
        }
    }
    result
}

/// SIMD search with position tracking - optimized for minimal overhead.
/// Tracks ALL positions where score <= k by incrementally collecting matches.
#[inline(always)]
fn search_simd_with_positions(
    transposed: &TQueries,
    target: &[u8],
    k: u8,
    results: &mut Vec<MatchInfo>,
) {
    let all_ones = i32x8::splat(!0);
    let zero_v = i32x8::splat(0);
    let one_v = i32x8::splat(1);

    let nq = transposed.n_queries;
    let m = transposed.query_length;

    let vectors_in_block = nq.div_ceil(SIMD_LANES);

    let mut pv_vec: Vec<i32x8> = vec![all_ones; vectors_in_block];
    let mut mv_vec: Vec<i32x8> = vec![zero_v; vectors_in_block];
    let mut scores_vec: Vec<i32x8> = vec![i32x8::splat(m as i32); vectors_in_block];

    let high_bit: u32 = 1u32 << (m - 1);
    let mask_vec = i32x8::splat(high_bit as i32);
    let k_v = i32x8::splat(k as i32);

    for (pos_counter, &tb) in target.iter().enumerate() {
        let encoded = get_encoded(tb);
        if encoded == INVALID_IUPAC {
            panic!("Target contains invalid IUPAC character: {:?}", tb as char);
        }
        let peq_slice = unsafe { transposed.peqs.get_unchecked(encoded as usize) };
        for v in 0..vectors_in_block {
            // Regular Myers
            let eq = unsafe { *peq_slice.get_unchecked(v) };
            let pv = pv_vec[v];
            let mv = mv_vec[v];
            let xv = eq | mv;
            let xh = (((eq & pv) + pv) ^ pv) | eq;
            let ph = mv | (all_ones ^ (xh | pv));
            let mh = pv & xh;

            // Track edit distance cost
            #[cfg(not(feature = "latest_wide"))]
            let ph_bit_mask = (ph & mask_vec).cmp_eq(zero_v);
            #[cfg(feature = "latest_wide")]
            let ph_bit_mask = (ph & mask_vec).simd_eq(zero_v);

            let ph_bit = (all_ones ^ ph_bit_mask) & one_v;

            #[cfg(not(feature = "latest_wide"))]
            let mh_bit_mask = (mh & mask_vec).cmp_eq(zero_v);
            #[cfg(feature = "latest_wide")]
            let mh_bit_mask = (mh & mask_vec).simd_eq(zero_v);

            let mh_bit = (all_ones ^ mh_bit_mask) & one_v;
            let ph_shift = ph << 1;
            let new_pv = (mh << 1) | (all_ones ^ (xv | ph_shift));
            let new_mv = ph_shift & xv;

            let new_score = unsafe { *scores_vec.get_unchecked(v) } + (ph_bit - mh_bit);

            unsafe {
                *pv_vec.get_unchecked_mut(v) = new_pv;
                *mv_vec.get_unchecked_mut(v) = new_mv;
                *scores_vec.get_unchecked_mut(v) = new_score;
            }

            // Check which queries in this SIMD lane have score <= k
            #[cfg(not(feature = "latest_wide"))]
            let match_mask = all_ones ^ new_score.cmp_gt(k_v);
            #[cfg(feature = "latest_wide")]
            let match_mask = all_ones ^ new_score.simd_gt(k_v);

            // Only extract scores if there are any matches in this vector
            // this prevents wasting the to_array() call when we don't have any
            // matches in the first place
            let match_bits = unsafe { std::mem::transmute::<i32x8, [i32; 8]>(match_mask) };
            let has_matches = match_bits.iter().any(|&b| b != 0);

            if has_matches {
                let score_arr = new_score.to_array();
                let base = v * SIMD_LANES;
                let end = (base + SIMD_LANES).min(nq);

                for (i, &score) in score_arr[..(end - base)].iter().enumerate() {
                    if score <= k as i32 {
                        results.push(MatchInfo {
                            query_idx: base + i,
                            cost: score,
                            pos: pos_counter as i32,
                        });
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn search_simd_with_positions_overhang(
    transposed: &TQueries,
    target: &[u8],
    k: u8,
    alpha: f32,
    results: &mut Vec<MatchInfoOverhang>,
) {
    const SCALE_SHIFT: u32 = 8;
    const SCALE: i32 = 1 << SCALE_SHIFT;

    let all_ones = i32x8::splat(!0);
    let zero_v = i32x8::splat(0);
    let one_v = i32x8::splat(1);

    let nq = transposed.n_queries;
    let m = transposed.query_length;
    let vectors_in_block = nq.div_ceil(SIMD_LANES);

    let alpha_scaled = ((alpha * (SCALE as f32)).round() as i32).clamp(0, SCALE);
    let extra_penalty_scaled = SCALE - alpha_scaled;
    let k_scaled = (k as i32) << SCALE_SHIFT;
    let k_scaled_vec = i32x8::splat(k_scaled);
    let inv_scale = 1.0f32 / (SCALE as f32);

    let mut pv_vec: Vec<i32x8> = vec![all_ones; vectors_in_block];
    let mut mv_vec: Vec<i32x8> = vec![zero_v; vectors_in_block];
    let mut scores_vec: Vec<i32x8> = vec![i32x8::splat(m as i32); vectors_in_block];

    let high_bit: u32 = 1u32 << (m - 1);
    let mask_vec = i32x8::splat(high_bit as i32);
    let zero_eq_vec = i32x8::splat(0);
    let ctx = OverhangColumnCtx {
        all_ones,
        zero_v,
        one_v,
        mask_vec,
        scale_shift: SCALE_SHIFT,
    };

    for (idx, &tb) in target.iter().enumerate() {
        let encoded = get_encoded(tb);
        if encoded == INVALID_IUPAC {
            panic!("Target contains invalid IUPAC character: {:?}", tb as char);
        }
        let overhang = m.saturating_sub(idx + 1) as i32;
        let adjust_vec = i32x8::splat(overhang * extra_penalty_scaled);
        let peq_slice = unsafe { transposed.peqs.get_unchecked(encoded as usize) };
        let pos = idx as i32;

        for v in 0..vectors_in_block {
            let eq = unsafe { *peq_slice.get_unchecked(v) };
            let adjusted = apply_overhang_column(
                eq,
                adjust_vec,
                unsafe { pv_vec.get_unchecked_mut(v) },
                unsafe { mv_vec.get_unchecked_mut(v) },
                unsafe { scores_vec.get_unchecked_mut(v) },
                ctx,
            );

            #[cfg(not(feature = "latest_wide"))]
            let match_mask = all_ones ^ adjusted.cmp_gt(k_scaled_vec);
            #[cfg(feature = "latest_wide")]
            let match_mask = all_ones ^ adjusted.simd_gt(k_scaled_vec);

            let match_bits = unsafe { std::mem::transmute::<i32x8, [i32; 8]>(match_mask) };
            if match_bits.iter().any(|&b| b != 0) {
                let adjusted_arr = adjusted.to_array();
                let base = v * SIMD_LANES;
                let end = (base + SIMD_LANES).min(nq);

                for (i, &score_scaled) in adjusted_arr[..(end - base)].iter().enumerate() {
                    if score_scaled <= k_scaled {
                        results.push(MatchInfoOverhang {
                            query_idx: base + i,
                            cost: score_scaled as f32 * inv_scale,
                            pos,
                        });
                    }
                }
            }
        }
    }

    if alpha_scaled < SCALE {
        for trailing in 1..=m {
            let adjust_vec = i32x8::splat((trailing as i32) * extra_penalty_scaled);
            let pos = (target.len() + trailing - 1) as i32;

            for v in 0..vectors_in_block {
                let adjusted = apply_overhang_column(
                    zero_eq_vec,
                    adjust_vec,
                    unsafe { pv_vec.get_unchecked_mut(v) },
                    unsafe { mv_vec.get_unchecked_mut(v) },
                    unsafe { scores_vec.get_unchecked_mut(v) },
                    ctx,
                );

                #[cfg(not(feature = "latest_wide"))]
                let match_mask = all_ones ^ adjusted.cmp_gt(k_scaled_vec);
                #[cfg(feature = "latest_wide")]
                let match_mask = all_ones ^ adjusted.simd_gt(k_scaled_vec);

                let match_bits = unsafe { std::mem::transmute::<i32x8, [i32; 8]>(match_mask) };
                if match_bits.iter().any(|&b| b != 0) {
                    let adjusted_arr = adjusted.to_array();
                    let base = v * SIMD_LANES;
                    let end = (base + SIMD_LANES).min(nq);

                    for (i, &score_scaled) in adjusted_arr[..(end - base)].iter().enumerate() {
                        if score_scaled <= k_scaled {
                            results.push(MatchInfoOverhang {
                                query_idx: base + i,
                                cost: score_scaled as f32 * inv_scale,
                                pos,
                            });
                        }
                    }
                }
            }
        }
    }
}

const INVALID_IUPAC: u8 = 255;

#[inline(always)]
fn get_encoded(c: u8) -> u8 {
    IUPAC_CODE[(c & 0x1F) as usize]
}

// Based on sassy: https://github.com/RagnarGrootKoerkamp/sassy/blob/master/src/profiles/iupac.rs#L258
// todo: add some tests for this table
#[rustfmt::skip]
const IUPAC_CODE: [u8; 32] = {
    // Every char *not* being in the table will be set to invalid IUPAC (255 u8 value)
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

    #[test]
    fn test_mini_search_with_positions() {
        let q = b"ATG".to_vec();
        let queries = vec![q];
        let t = b"CCCCCCCCCATGCCCCC";
        //             012345678901234567
        //                      ATG at pos 9-11 (ends at 11)
        let transposed = TQueries::new(&queries);
        let mut result = Vec::new();
        mini_search_with_positions(&transposed, t, 4, &mut result);

        // Should find at least one match
        assert!(!result.is_empty());
        // Find the exact match (cost 0)
        let exact_matches: Vec<_> = result.iter().filter(|m| m.cost == 0).collect();
        assert!(!exact_matches.is_empty());
        assert_eq!(exact_matches[0].query_idx, 0);
        assert_eq!(exact_matches[0].pos, 11); // Position where match ends
    }

    #[test]
    fn test_positions_multiple_occurrences() {
        let queries = vec![b"ATG".to_vec()];
        let t = b"ATGCCCATGCCCATG";
        //        012345678901234
        // ATG at positions 0-2 (ends at 2), 6-8 (ends at 8), 12-14 (ends at 14)
        let transposed = TQueries::new(&queries);
        let mut result = Vec::new();
        mini_search_with_positions(&transposed, t, 0, &mut result);

        // Should find all 3 exact matches
        let exact_matches: Vec<_> = result
            .iter()
            .filter(|m| m.cost == 0 && m.query_idx == 0)
            .collect();
        assert_eq!(exact_matches.len(), 3);
        assert_eq!(exact_matches[0].pos, 2);
        assert_eq!(exact_matches[1].pos, 8);
        assert_eq!(exact_matches[2].pos, 14);
    }

    #[test]
    fn test_positions_match_costs() {
        let q1 = b"ATG".to_vec();
        let q2 = b"TTG".to_vec();
        let queries = vec![q1, q2];
        let t = b"CCCTTGCCCCCCATGCCCCC";
        let transposed = TQueries::new(&queries);

        let result_basic = mini_search(&transposed, t, 4);
        let mut result_pos = Vec::new();
        mini_search_with_positions(&transposed, t, 4, &mut result_pos);

        // mini_search returns minimum cost per query
        // mini_search_with_positions returns all matches
        // Check that the minimum cost in positions matches mini_search result
        for (query_idx, _) in queries.iter().enumerate() {
            let min_cost_in_positions = result_pos
                .iter()
                .filter(|m| m.query_idx == query_idx)
                .map(|m| m.cost)
                .min()
                .unwrap_or(-1);
            assert_eq!(result_basic[query_idx], min_cost_in_positions);
        }
    }

    #[test]
    fn test_positions_no_match() {
        let q = b"AAAAA".to_vec();
        let queries = vec![q];
        let t = b"CCCCCCCCCCCCC";
        let transposed = TQueries::new(&queries);
        let mut result = Vec::new();
        mini_search_with_positions(&transposed, t, 1, &mut result);

        // Should have no matches for query 0 with cost <= 1
        let matches_q0: Vec<_> = result.iter().filter(|m| m.query_idx == 0).collect();
        assert_eq!(matches_q0.len(), 0);
    }

    #[test]
    fn test_positions_multiple_queries() {
        let queries = vec![
            b"AAAA".to_vec(),
            b"TTTT".to_vec(),
            b"GGGG".to_vec(),
            b"CCCC".to_vec(),
        ];
        let t = b"AAGGGGTTTTCCCC";
        let transposed = TQueries::new(&queries);
        let mut result = Vec::new();
        mini_search_with_positions(&transposed, t, 2, &mut result);

        // All queries should have at least one match with k=2
        for query_idx in 0..4 {
            let query_matches: Vec<_> =
                result.iter().filter(|m| m.query_idx == query_idx).collect();
            assert!(!query_matches.is_empty());
            // All matches should be within threshold
            for m in &query_matches {
                assert!(m.cost >= 0 && m.cost <= 2);
                assert!(m.pos >= 0);
            }
        }
    }
    #[test]
    fn test_overhang_half_penalty() {
        let queries = vec![b"AC".to_vec()];
        let transposed = TQueries::new(&queries);
        let t = b"A";
        let result = mini_search_with_overhang(&transposed, t, 1, 0.5);
        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - 0.5).abs() < 1e-6,
            "expected 0.5, got {}",
            result[0]
        );
    }

    #[test]
    fn test_overhang_matches_standard_when_alpha_one() {
        let queries = vec![b"ATG".to_vec()];
        let transposed = TQueries::new(&queries);
        let t = b"CCCCCCCCCATGCCCCC";
        let standard = mini_search(&transposed, t, 4);
        let overhang = mini_search_with_overhang(&transposed, t, 4, 1.0);
        assert_eq!(standard.len(), overhang.len());
        for (i, &val) in standard.iter().enumerate() {
            assert!((overhang[i] - val as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn test_positions_overhang() {
        let queries = vec![b"AC".to_vec()];
        let transposed = TQueries::new(&queries);
        let t = b"A";
        let mut results = Vec::new();
        mini_search_with_positions_overhang(&transposed, t, 1, 0.5, &mut results);
        assert!(!results.is_empty());
        let min_cost_entry = results
            .iter()
            .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap())
            .unwrap();
        assert_eq!(min_cost_entry.query_idx, 0);
        assert!((min_cost_entry.cost - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_longer_overhang_left() {
        let queries = vec![b"AAACCC".to_vec()];
        let transposed = TQueries::new(&queries);
        let t = b"CCCGGGGGGGG";
        let mut results = Vec::new();
        mini_search_with_positions_overhang(&transposed, t, 4, 0.5, &mut results);
        // Get minimum cost match
        let min_cost = results
            .iter()
            .map(|m| m.cost)
            .fold(f32::INFINITY, |a, b| a.min(b));
        assert!(
            (min_cost - 1.5).abs() < 1e-6,
            "expected 1.5, got {}",
            min_cost
        );
    }

    #[test]
    fn test_longer_overhang_right() {
        let queries = vec![b"AAACCC".to_vec()];
        let transposed = TQueries::new(&queries);
        let t = b"GGGGGAAA";
        let mut results = Vec::new();
        mini_search_with_positions_overhang(&transposed, t, 4, 0.5, &mut results);
        // Get minimum cost match
        let min_cost = results
            .iter()
            .map(|m| m.cost)
            .fold(f32::INFINITY, |a, b| a.min(b));
        assert!(
            (min_cost - 1.5).abs() < 1e-6,
            "expected 1.5, got {}",
            min_cost
        );
    }
}
