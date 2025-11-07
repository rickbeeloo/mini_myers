use crate::backend::SimdBackend;
use crate::tqueries::TQueries;
use std::marker::PhantomData;
use wide::{CmpEq, CmpGt};

/// Match information including position and cost for a query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchInfo {
    /// Index of the query (0-based)
    pub query_idx: usize,
    /// Minimum edit distance found (0 to k), including overhang penalties when used
    pub cost: f32,
    /// Position in target where best match ends (0-based)
    pub pos: i32,
}

/// Marker type for scan mode (returns minimum cost per query).
pub struct Scan;

/// Marker type for positions mode (returns all match positions).
pub struct Positions;

/// Trait for search modes.
pub trait SearchMode {
    type Output;
}

impl SearchMode for Scan {
    type Output = Vec<f32>;
}

impl SearchMode for Positions {
    type Output = Vec<MatchInfo>;
}

/// Main searcher struct that holds backend type, mode, and reusable state.
///
/// # Type Parameters
///
/// * `B` - Backend type: `U32` (i32x8, up to 32-bit queries) or `U64` (i64x4, up to 64-bit queries)
/// * `M` - Mode: `Scan` (minimum cost per query) or `Positions` (all match positions)
///
/// # Examples
///
/// ```
/// use mini_myers::{Searcher, Scan, Positions};
/// use mini_myers::backend::U32;
///
/// // Create a searcher for scan mode with U32 backend
/// let mut searcher = Searcher::<U32, Scan>::new();
/// let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
/// let encoded = searcher.encode(&queries);
/// let target = b"CCCTCGCCCCCCATGCCCCC";
/// let results = searcher.search(&encoded, target, 4, None);
/// assert_eq!(results, vec![0.0, 1.0]);
/// ```
pub struct Searcher<B: SimdBackend, M: SearchMode> {
    state: MyersSearchState<B>,
    _mode: PhantomData<M>,
}

impl<B: SimdBackend, M: SearchMode> Searcher<B, M> {
    /// Creates a new searcher.
    pub fn new() -> Self {
        Self {
            state: MyersSearchState::new(),
            _mode: PhantomData,
        }
    }

    /// Encodes queries for searching.
    ///
    /// All queries must have the same length.
    pub fn encode(&self, queries: &[Vec<u8>]) -> TQueries<B> {
        TQueries::new(queries)
    }

    /// Searches for encoded queries in the target sequence.
    ///
    /// # Arguments
    ///
    /// * `encoded` - Encoded queries from `encode()`
    /// * `target` - Target DNA sequence to search
    /// * `k` - Maximum edit distance threshold
    /// * `alpha` - Optional overhang penalty (0.0-1.0, default 1.0)
    ///
    /// # Returns
    ///
    /// * `Scan` mode: `Vec<f32>` with minimum cost per query (-1.0 if no match)
    /// * `Positions` mode: `Vec<MatchInfo>` with all matches found
    pub fn search(
        &mut self,
        encoded: &TQueries<B>,
        target: &[u8],
        k: u8,
        alpha: Option<f32>,
    ) -> M::Output
    where
        M::Output: SearchExecutor<B>,
    {
        M::Output::execute(&mut self.state, encoded, target, k, alpha)
    }
}

impl<B: SimdBackend, M: SearchMode> Default for Searcher<B, M> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for collecting search results during the search loop (internal).
trait ResultCollector<B: SimdBackend> {
    fn init(&mut self, nq: usize, m: usize, initial_adjusted: i64);
    fn collect_match(
        &mut self,
        adjusted: B::Simd,
        block_idx: usize,
        nq: usize,
        pos: i32,
        k_scaled_vec: B::Simd,
        inv_scale: f32,
    );
    fn update_best(&mut self, adjusted: B::Simd, block_idx: usize);
    fn finalize(&mut self, nq: usize, k_scaled_vec: B::Simd, inv_scale: f32) -> Self::Output;
    type Output;
}

struct ScanCollector<B: SimdBackend> {
    best_adjusted: Vec<B::Simd>,
}

impl<B: SimdBackend> ResultCollector<B> for ScanCollector<B> {
    fn init(&mut self, nq: usize, _m: usize, initial_adjusted: i64) {
        let vectors_in_block = nq.div_ceil(B::LANES);
        let initial_adj_v = B::splat_from_i64(initial_adjusted);
        self.best_adjusted.resize(vectors_in_block, initial_adj_v);
        self.best_adjusted.fill(initial_adj_v);
    }

    #[inline(always)]
    fn collect_match(
        &mut self,
        _adjusted: B::Simd,
        _block_idx: usize,
        _nq: usize,
        _pos: i32,
        _k_scaled_vec: B::Simd,
        _inv_scale: f32,
    ) {
    }

    #[inline(always)]
    fn update_best(&mut self, adjusted: B::Simd, block_idx: usize) {
        unsafe {
            let best_slot = self.best_adjusted.get_unchecked_mut(block_idx);
            *best_slot = B::min(*best_slot, adjusted);
        }
    }

    fn finalize(&mut self, nq: usize, k_scaled_vec: B::Simd, inv_scale: f32) -> Self::Output {
        let all_ones = B::splat_all_ones();
        let neg_mask = B::splat_scalar(B::MAX_POSITIVE);
        let mut result = vec![-1.0f32; nq];

        for (block_idx, &min_adj) in self.best_adjusted.iter().enumerate() {
            let mask = all_ones ^ min_adj.simd_gt(k_scaled_vec);
            let selected = B::blend(mask, min_adj, neg_mask);
            let base = block_idx * B::LANES;
            let end = (base + B::LANES).min(nq);
            let selected_arr = B::to_array(selected);
            let selected_slice = selected_arr.as_ref();

            for (lane_idx, &val) in selected_slice[..(end - base)].iter().enumerate() {
                if val != B::MAX_POSITIVE {
                    result[base + lane_idx] = B::scalar_to_f32(val) * inv_scale;
                }
            }
        }
        result
    }

    type Output = Vec<f32>;
}

struct PositionsCollector<B: SimdBackend> {
    results: Vec<MatchInfo>,
    _phantom: PhantomData<B>,
}

impl<B: SimdBackend> ResultCollector<B> for PositionsCollector<B> {
    fn init(&mut self, _nq: usize, _m: usize, _initial_adjusted: i64) {
        self.results.clear();
    }

    #[inline(always)]
    fn collect_match(
        &mut self,
        adjusted: B::Simd,
        block_idx: usize,
        nq: usize,
        pos: i32,
        k_scaled_vec: B::Simd,
        inv_scale: f32,
    ) {
        let all_ones = B::splat_all_ones();
        let match_mask = all_ones ^ adjusted.simd_gt(k_scaled_vec);
        let match_bits_arr = B::to_array(match_mask);
        let match_bits = match_bits_arr.as_ref();

        let zero_scalar = B::scalar_from_i64(0);
        let zero_scalar_i64 = B::scalar_to_i64(zero_scalar);

        if match_bits
            .iter()
            .any(|&b| B::scalar_to_i64(b) != zero_scalar_i64)
        {
            let adjusted_arr = B::to_array(adjusted);
            let adjusted_slice = adjusted_arr.as_ref();
            let k_scaled_arr = B::to_array(k_scaled_vec);
            let k_scaled_i64 = B::scalar_to_i64(k_scaled_arr.as_ref()[0]);
            let base = block_idx * B::LANES;
            let end = (base + B::LANES).min(nq);

            for (i, &score_scaled) in adjusted_slice[..(end - base)].iter().enumerate() {
                if B::scalar_to_i64(score_scaled) <= k_scaled_i64 {
                    self.results.push(MatchInfo {
                        query_idx: base + i,
                        cost: B::scalar_to_f32(score_scaled) * inv_scale,
                        pos,
                    });
                }
            }
        }
    }

    #[inline(always)]
    fn update_best(&mut self, _adjusted: B::Simd, _block_idx: usize) {}

    fn finalize(&mut self, _nq: usize, _k_scaled_vec: B::Simd, _inv_scale: f32) -> Self::Output {
        std::mem::take(&mut self.results)
    }

    type Output = Vec<MatchInfo>;
}

/// Trait for executing searches (internal).
pub trait SearchExecutor<B: SimdBackend>: Sized {
    fn execute(
        state: &mut MyersSearchState<B>,
        encoded: &TQueries<B>,
        target: &[u8],
        k: u8,
        alpha: Option<f32>,
    ) -> Self;
}

impl<B: SimdBackend> SearchExecutor<B> for Vec<f32> {
    fn execute(
        state: &mut MyersSearchState<B>,
        encoded: &TQueries<B>,
        target: &[u8],
        k: u8,
        alpha: Option<f32>,
    ) -> Self {
        let mut collector = ScanCollector {
            best_adjusted: Vec::new(),
        };
        search(state, &mut collector, encoded, target, k, alpha)
    }
}

impl<B: SimdBackend> SearchExecutor<B> for Vec<MatchInfo> {
    fn execute(
        state: &mut MyersSearchState<B>,
        encoded: &TQueries<B>,
        target: &[u8],
        k: u8,
        alpha: Option<f32>,
    ) -> Self {
        let mut collector = PositionsCollector {
            results: Vec::new(),
            _phantom: PhantomData,
        };
        search(state, &mut collector, encoded, target, k, alpha)
    }
}

/// Holds the reusable state vectors for the Myers search.
pub struct MyersSearchState<B: SimdBackend> {
    pub(crate) pv: Vec<B::Simd>,
    pub(crate) mv: Vec<B::Simd>,
    pub(crate) score: Vec<B::Simd>,
    pub(crate) best_adjusted: Vec<B::Simd>,
    pub result: Vec<f32>,
}

impl<B: SimdBackend> MyersSearchState<B> {
    /// Creates a new, empty state.
    pub fn new() -> Self {
        Self {
            pv: Vec::new(),
            mv: Vec::new(),
            score: Vec::new(),
            best_adjusted: Vec::new(),
            result: Vec::new(),
        }
    }

    /// Resizes and re-initializes state for scan mode (minimum cost per query).
    pub fn reset_for_scan(&mut self, nq: usize, m: usize, initial_adjusted: i64) {
        let vectors_in_block = nq.div_ceil(B::LANES);
        let all_ones = B::splat_all_ones();
        let zero_v = B::splat_zero();
        let m_v = B::splat_from_usize(m);
        let initial_adj_v = B::splat_from_i64(initial_adjusted);

        self.pv.resize(vectors_in_block, all_ones);
        self.pv.fill(all_ones);
        self.mv.resize(vectors_in_block, zero_v);
        self.mv.fill(zero_v);
        self.score.resize(vectors_in_block, m_v);
        self.score.fill(m_v);
        self.best_adjusted.resize(vectors_in_block, initial_adj_v);
        self.best_adjusted.fill(initial_adj_v);
        self.result.resize(nq, -1.0f32);
        self.result.fill(-1.0f32);
    }

    /// Resizes and re-initializes state for positions mode (all matches).
    pub fn reset_for_positions(&mut self, nq: usize, m: usize) {
        let vectors_in_block = nq.div_ceil(B::LANES);
        let all_ones = B::splat_all_ones();
        let zero_v = B::splat_zero();
        let m_v = B::splat_from_usize(m);

        self.pv.resize(vectors_in_block, all_ones);
        self.pv.fill(all_ones);
        self.mv.resize(vectors_in_block, zero_v);
        self.mv.fill(zero_v);
        self.score.resize(vectors_in_block, m_v);
        self.score.fill(m_v);
    }
}

impl<B: SimdBackend> Default for MyersSearchState<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Search that can uses any collector (scan or position)
#[inline(always)]
fn search<B: SimdBackend, C: ResultCollector<B>>(
    state: &mut MyersSearchState<B>,
    collector: &mut C,
    encoded: &TQueries<B>,
    target: &[u8],
    k: u8,
    alpha: Option<f32>,
) -> C::Output {
    let nq = encoded.n_queries;
    if nq == 0 {
        return collector.finalize(0, B::splat_zero(), 1.0);
    }
    let m = encoded.query_length;
    assert!(
        m > 0 && m <= B::LIMB_BITS,
        "query length must be 1..={}",
        B::LIMB_BITS
    );

    let alpha = alpha.unwrap_or(1.0).clamp(0.0, 1.0);
    const SCALE_SHIFT: u32 = 8;
    const SCALE: i32 = 1 << SCALE_SHIFT;
    let alpha_scaled = ((alpha * (SCALE as f32)).round() as i32).clamp(0, SCALE);
    let initial_adjusted = (m as i64) * (alpha_scaled as i64);

    state.reset_for_positions(nq, m);
    collector.init(nq, m, initial_adjusted);
    search_core(
        state,
        collector,
        encoded,
        target,
        k,
        alpha_scaled,
        SCALE,
        SCALE_SHIFT,
    );

    let k_scaled = (k as i32) << SCALE_SHIFT;
    let k_scaled_vec = B::splat_from_i64(k_scaled as i64);
    let inv_scale = 1.0f32 / (SCALE as f32);

    collector.finalize(nq, k_scaled_vec, inv_scale)
}

#[derive(Copy, Clone)]
struct OverhangColumnCtx<B: SimdBackend> {
    all_ones: B::Simd,
    zero_v: B::Simd,
    one_v: B::Simd,
    mask_vec: B::Simd,
    scale_shift: u32,
}

#[inline(always)]
fn advance_column<B: SimdBackend>(
    pv: &mut B::Simd,
    mv: &mut B::Simd,
    score: &mut B::Simd,
    eq: B::Simd,
    adjust_vec: B::Simd,
    ctx: OverhangColumnCtx<B>,
) -> B::Simd {
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
    let ph_bit_mask = (ph & mask_vec).simd_eq(zero_v);
    let ph_bit = (all_ones ^ ph_bit_mask) & one_v;
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

/// Core SIMD Myers implementation
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn search_core<B: SimdBackend, C: ResultCollector<B>>(
    state: &mut MyersSearchState<B>,
    collector: &mut C,
    transposed: &TQueries<B>,
    target: &[u8],
    k: u8,
    alpha_scaled: i32,
    scale: i32,
    scale_shift: u32,
) {
    let nq = transposed.n_queries;
    let m = transposed.query_length;
    let vectors_in_block = nq.div_ceil(B::LANES);

    let extra_penalty_scaled = scale - alpha_scaled;
    let k_scaled = (k as i32) << scale_shift;
    let inv_scale = 1.0f32 / (scale as f32);

    let all_ones = B::splat_all_ones();
    let zero_v = B::splat_zero();
    let one_v = B::splat_one();
    let zero_eq_vec = B::splat_zero();

    let high_bit: u64 = 1u64 << (m - 1);
    let mask_vec = B::splat_from_i64(high_bit as i64);
    let k_scaled_vec = B::splat_from_i64(k_scaled as i64);
    let ctx: OverhangColumnCtx<B> = OverhangColumnCtx {
        all_ones,
        zero_v,
        one_v,
        mask_vec,
        scale_shift,
    };

    // Main loop over target sequence
    for (idx, &tb) in target.iter().enumerate() {
        let encoded = crate::iupac::get_encoded(tb);
        let overhang = m.saturating_sub(idx + 1) as i32;
        let adjust_vec = B::splat_from_i64((overhang as i64) * (extra_penalty_scaled as i64));
        let peq_slice = unsafe { transposed.peqs.get_unchecked(encoded as usize) };
        let pos = idx as i32;

        unsafe {
            for block_idx in 0..vectors_in_block {
                let eq = *peq_slice.get_unchecked(block_idx);
                let adjusted: B::Simd = advance_column(
                    state.pv.get_unchecked_mut(block_idx),
                    state.mv.get_unchecked_mut(block_idx),
                    state.score.get_unchecked_mut(block_idx),
                    eq,
                    adjust_vec,
                    ctx,
                );

                // This can have some overhead, but cleaner code base I guess,
                // we could const_generic def to "see" the different mode at compile time
                // Update best scores (for scan mode)
                collector.update_best(adjusted, block_idx);
                // Collect matches (for positions mode)
                collector.collect_match(adjusted, block_idx, nq, pos, k_scaled_vec, inv_scale);
            }
        }
    }

    // Overhang trailing positions
    if alpha_scaled < scale {
        for trailing in 1..=m {
            let adjust_vec = B::splat_from_i64((trailing as i64) * (extra_penalty_scaled as i64));
            let pos = (target.len() + trailing - 1) as i32;

            unsafe {
                for block_idx in 0..vectors_in_block {
                    let adjusted: B::Simd = advance_column(
                        state.pv.get_unchecked_mut(block_idx),
                        state.mv.get_unchecked_mut(block_idx),
                        state.score.get_unchecked_mut(block_idx),
                        zero_eq_vec,
                        adjust_vec,
                        ctx,
                    );

                    // Update best scores (for scan mode)
                    collector.update_best(adjusted, block_idx);

                    // Collect matches (for positions mode)
                    collector.collect_match(adjusted, block_idx, nq, pos, k_scaled_vec, inv_scale);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Positions, Scan, Searcher};
    use crate::backend::{U32, U64};

    #[test]
    fn test_iupac_query_matches_standard_target() {
        let mut searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"n".to_vec()];
        let encoded = searcher.encode(&queries);
        assert_eq!(searcher.search(&encoded, b"A", 0, None), vec![0.0]);
        assert_eq!(searcher.search(&encoded, b"a", 0, None), vec![0.0]);
    }

    #[test]
    fn test_iupac_target_matches_standard_query() {
        let mut searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"A".to_vec()];
        let encoded = searcher.encode(&queries);
        assert_eq!(searcher.search(&encoded, b"N", 0, None), vec![0.0]);
        assert_eq!(searcher.search(&encoded, b"n", 0, None), vec![0.0]);
    }

    #[test]
    fn test_iupac_mismatch_requires_edit() {
        let mut searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"R".to_vec()];
        let encoded = searcher.encode(&queries);
        assert_eq!(searcher.search(&encoded, b"C", 0, None), vec![-1.0]);
        assert_eq!(searcher.search(&encoded, b"C", 1, None), vec![1.0]);
        assert_eq!(searcher.search(&encoded, b"c", 1, None), vec![1.0]);
    }

    #[test]
    #[should_panic(expected = "invalid IUPAC character")]
    fn test_invalid_query_panics() {
        let searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"AZ".to_vec()];
        let _ = searcher.encode(&queries);
    }

    #[test]
    fn test_mini_search() {
        let mut searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries);
        let result = searcher.search(&encoded, b"CCCCCCCCCATGCCCCC", 4, None);
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    fn test_double_search() {
        let mut searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
        let encoded = searcher.encode(&queries);
        let result = searcher.search(&encoded, b"CCCTTGCCCCCCATGCCCCC", 4, None);
        assert_eq!(result, vec![0.0, 0.0]);
    }

    #[test]
    fn test_edit() {
        let mut searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"ATCAGA".to_vec()];
        let encoded = searcher.encode(&queries);

        // 1 edit
        let result = searcher.search(&encoded, b"ATCTGA", 4, None);
        assert_eq!(result, vec![1.0]);

        // 2 edits
        let result = searcher.search(&encoded, b"GTCTGA", 4, None);
        assert_eq!(result, vec![2.0]);

        // 3 edits (1 del)
        let result = searcher.search(&encoded, b"GTTGA", 4, None);
        assert_eq!(result, vec![3.0]);

        // Match should not be recovered when k == 1
        let result = searcher.search(&encoded, b"GTTGA", 1, None);
        assert_eq!(result, vec![-1.0]);
    }

    #[test]
    fn test_lowest_edits_returned() {
        let mut searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"GGGCATCGATGAC".to_vec()];
        let encoded = searcher.encode(&queries);
        let target = b"CCCCCCCGGGCATCGATGACCCCCCCCCCCCCCCGGGCTTCGATGAC";
        let result = searcher.search(&encoded, target, 4, None);
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    #[should_panic(expected = "All queries must have the same length")]
    fn test_error_unequal_lengths() {
        let searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"GGGCATCGATGAC".to_vec(), b"AAA".to_vec()];
        let _ = searcher.encode(&queries);
    }

    #[test]
    fn read_example() {
        let mut searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
        let encoded = searcher.encode(&queries);
        let target = b"CCCTCGCCCCCCATGCCCCC";
        let result = searcher.search(&encoded, target, 4, None);
        println!("Result: {:?}", result);
        assert_eq!(result, vec![0.0, 1.0]);
    }

    #[test]
    fn test_mini_search_with_positions() {
        let mut searcher = Searcher::<U32, Positions>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries);
        let target = b"CCCCCCCCCATGCCCCC";
        let result = searcher.search(&encoded, target, 4, None);

        // Should find at least one match
        assert!(!result.is_empty());
        // Find the exact match (cost 0)
        let exact_matches: Vec<_> = result.iter().filter(|m| m.cost == 0.0).collect();
        assert!(!exact_matches.is_empty());
        assert_eq!(exact_matches[0].query_idx, 0);
        assert_eq!(exact_matches[0].pos, 11); // Position where match ends
    }

    #[test]
    fn test_positions_multiple_occurrences() {
        let mut searcher = Searcher::<U32, Positions>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries);
        let target = b"ATGCCCATGCCCATG";
        let result = searcher.search(&encoded, target, 0, None);

        // Should find all 3 exact matches
        let exact_matches: Vec<_> = result
            .iter()
            .filter(|m| (m.cost - 0.0).abs() < f32::EPSILON && m.query_idx == 0)
            .collect();
        assert_eq!(exact_matches.len(), 3);
        assert_eq!(exact_matches[0].pos, 2);
        assert_eq!(exact_matches[1].pos, 8);
        assert_eq!(exact_matches[2].pos, 14);
    }

    #[test]
    fn test_positions_match_costs() {
        let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
        let target = b"CCCTTGCCCCCCATGCCCCC";

        let mut scan_searcher = Searcher::<U32, Scan>::new();
        let mut pos_searcher = Searcher::<U32, Positions>::new();
        let encoded_scan = scan_searcher.encode(&queries);
        let encoded_pos = pos_searcher.encode(&queries);

        let result_scan = scan_searcher.search(&encoded_scan, target, 4, None);
        let result_pos = pos_searcher.search(&encoded_pos, target, 4, None);

        // Check that the minimum cost in positions matches scan result
        for (query_idx, _) in queries.iter().enumerate() {
            let min_cost_in_positions = result_pos
                .iter()
                .filter(|m| m.query_idx == query_idx)
                .map(|m| m.cost)
                .fold(f32::INFINITY, f32::min);
            let min_cost_in_positions = if min_cost_in_positions.is_finite() {
                min_cost_in_positions
            } else {
                -1.0
            };
            assert!((result_scan[query_idx] - min_cost_in_positions).abs() < 1e-6);
        }
    }

    #[test]
    fn test_positions_no_match() {
        let mut searcher = Searcher::<U32, Positions>::new();
        let queries = vec![b"AAAAA".to_vec()];
        let encoded = searcher.encode(&queries);
        let result = searcher.search(&encoded, b"CCCCCCCCCCCCC", 1, None);

        // Should have no matches for query 0 with cost <= 1
        let matches_q0: Vec<_> = result.iter().filter(|m| m.query_idx == 0).collect();
        assert_eq!(matches_q0.len(), 0);
    }

    #[test]
    fn test_positions_multiple_queries() {
        let mut searcher = Searcher::<U32, Positions>::new();
        let queries = vec![
            b"AAAA".to_vec(),
            b"TTTT".to_vec(),
            b"GGGG".to_vec(),
            b"CCCC".to_vec(),
        ];
        let encoded = searcher.encode(&queries);
        let result = searcher.search(&encoded, b"AAGGGGTTTTCCCC", 2, None);

        // All queries should have at least one match with k=2
        for query_idx in 0..4 {
            let query_matches: Vec<_> =
                result.iter().filter(|m| m.query_idx == query_idx).collect();
            assert!(!query_matches.is_empty());
            // All matches should be within threshold
            for m in &query_matches {
                assert!(m.cost >= 0.0 && m.cost <= 2.0);
                assert!(m.pos >= 0);
            }
        }
    }

    #[test]
    fn test_overhang_half_penalty() {
        let mut searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"AC".to_vec()];
        let encoded = searcher.encode(&queries);
        let result = searcher.search(&encoded, b"A", 1, Some(0.5));
        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - 0.5).abs() < 1e-6,
            "expected 0.5, got {}",
            result[0]
        );
    }

    #[test]
    fn test_overhang_matches_standard_when_alpha_one() {
        let mut searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries);
        let target = b"CCCCCCCCCATGCCCCC";
        let standard = searcher.search(&encoded, target, 4, None);
        let overhang = searcher.search(&encoded, target, 4, Some(1.0));
        assert_eq!(standard.len(), overhang.len());
        for (i, &val) in standard.iter().enumerate() {
            assert!((overhang[i] - val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_positions_overhang() {
        let mut searcher = Searcher::<U32, Positions>::new();
        let queries = vec![b"AC".to_vec()];
        let encoded = searcher.encode(&queries);
        let results = searcher.search(&encoded, b"A", 1, Some(0.5));
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
        let mut searcher = Searcher::<U32, Positions>::new();
        let queries = vec![b"AAACCC".to_vec()];
        let encoded = searcher.encode(&queries);
        let results = searcher.search(&encoded, b"CCCGGGGGGGG", 4, Some(0.5));
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
        let mut searcher = Searcher::<U32, Positions>::new();
        let queries = vec![b"AAACCC".to_vec()];
        let encoded = searcher.encode(&queries);
        let results = searcher.search(&encoded, b"GGGGGAAA", 4, Some(0.5));
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
    fn test_overhang_with_positions() {
        let mut searcher = Searcher::<U32, Positions>::new();
        let queries = vec![b"AAACCC".to_vec()];
        let encoded = searcher.encode(&queries);
        let results = searcher.search(&encoded, b"GGGGGAAA", 4, Some(0.5));
        // Get minimum cost match
        let _min_cost = results
            .iter()
            .map(|m| m.cost)
            .fold(f32::INFINITY, |a, b| a.min(b));
        println!("results: {:?}", results);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_searcher_u32_scan() {
        let mut searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
        let encoded = searcher.encode(&queries);
        let target = b"CCCTCGCCCCCCATGCCCCC";
        let results = searcher.search(&encoded, target, 4, None);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 0.0);
        assert_eq!(results[1], 1.0);
    }

    #[test]
    fn test_searcher_u64_scan() {
        let mut searcher = Searcher::<U64, Scan>::new();
        let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
        let encoded = searcher.encode(&queries);
        let target = b"CCCTCGCCCCCCATGCCCCC";
        let results = searcher.search(&encoded, target, 4, None);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 0.0);
        assert_eq!(results[1], 1.0);
    }

    #[test]
    fn test_searcher_u32_positions() {
        let mut searcher = Searcher::<U32, Positions>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries);
        let target = b"ATGCCCATGCCCATG";
        let results = searcher.search(&encoded, target, 0, None);

        // Should find all 3 exact matches
        let exact_matches: Vec<_> = results
            .iter()
            .filter(|m| (m.cost - 0.0).abs() < f32::EPSILON && m.query_idx == 0)
            .collect();
        assert_eq!(exact_matches.len(), 3);
        assert_eq!(exact_matches[0].pos, 2);
        assert_eq!(exact_matches[1].pos, 8);
        assert_eq!(exact_matches[2].pos, 14);
    }

    #[test]
    fn test_searcher_u64_positions() {
        use std::iter::repeat_n;
        let mut searcher = Searcher::<U64, Positions>::new();
        let long_query = repeat_n(b'A', 64).collect::<Vec<u8>>();
        let queries = vec![long_query.clone()];
        let encoded = searcher.encode(&queries);
        let mut target = vec![b'C'; 200];
        let insert_pos = 100;
        target.splice(insert_pos..insert_pos, long_query.iter().cloned());
        let results = searcher.search(&encoded, &target, 4, None);
        println!("results: {:?}", results);

        assert!(!results.is_empty());
        let exact_matches: Vec<_> = results.iter().filter(|m| m.cost == 0.0).collect();
        assert!(!exact_matches.is_empty());
        assert_eq!(exact_matches[0].query_idx, 0);
        assert_eq!(exact_matches[0].pos, 163);
    }

    #[test]
    fn test_searcher_with_overhang() {
        let mut searcher = Searcher::<U32, Scan>::new();
        let queries = vec![b"AC".to_vec()];
        let encoded = searcher.encode(&queries);
        let target = b"A";

        // Test with overhang penalty
        let results = searcher.search(&encoded, target, 1, Some(0.5));

        assert_eq!(results.len(), 1);
        assert!((results[0] - 0.5).abs() < 1e-6);
    }
}
