use crate::backend::SimdBackend;
use crate::tqueries::TQueries;
use sassy::{fill, profiles::Iupac, CostMatrix, Match};
use wide::CmpEq;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Strand {
    Fwd,
    Rc,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchInfo {
    pub query_idx: usize,
    pub cost: f32,
    pub pos: i32,
    pub strand: Strand,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WrappedSassyMatch {
    pub sassy_match: Match,
    pub query_idx: usize,
}

impl WrappedSassyMatch {
    pub fn new(sassy_match: Match, query_idx: usize) -> Self {
        Self {
            sassy_match,
            query_idx,
        }
    }
}

/// Main searcher struct
///
/// # Type Parameters
///
/// * `B` - Backend type: `U32` or `U64`
///
/// # Examples
///
/// ```
/// use mini_myers::Searcher;
/// use mini_myers::backend::U32;
///
/// // Create a searcher
/// let mut searcher = Searcher::<U32>::new();
/// let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
/// let encoded = searcher.encode(&queries, false);
/// let target = b"CCCTCGCCCCCCATGCCCCC";
///
/// // Scan mode - returns minimum costs
/// let results = searcher.scan(&encoded, target, 4, None);
/// assert_eq!(results, vec![0.0, 1.0]);
///
/// // Positions mode - returns all match locations
/// let matches = searcher.search(&encoded, target, 4, None);
/// ```
pub struct Searcher<B: SimdBackend> {
    state: MyersSearchState<B>,
}

impl<B: SimdBackend> Searcher<B> {
    pub fn new() -> Self {
        Self {
            state: MyersSearchState::new(),
        }
    }

    /// Encodes queries for searching
    pub fn encode(&self, queries: &[Vec<u8>], include_rc: bool) -> TQueries<B> {
        TQueries::new(queries, include_rc)
    }

    /// Scan mode - returns minimum costs per query
    pub fn scan(
        &mut self,
        encoded: &TQueries<B>,
        target: &[u8],
        k: u8,
        alpha: Option<f32>,
    ) -> Vec<f32> {
        search_impl_scan(&mut self.state, encoded, target, k, alpha)
    }

    /// Positions mode - returns all match positions
    pub fn search(
        &mut self,
        encoded: &TQueries<B>,
        target: &[u8],
        k: u8,
        alpha: Option<f32>,
    ) -> Vec<WrappedSassyMatch> {
        search_impl_positions(&mut self.state, encoded, target, k, alpha)
    }
}

impl<B: SimdBackend> Default for Searcher<B> {
    fn default() -> Self {
        Self::new()
    }
}

// Internal stuff

#[derive(Clone, Default)]
struct QueryState {
    prev_cost: Option<f32>,
    candidate: Option<MatchInfo>,
    last_pos: Option<i32>,
}

impl QueryState {
    #[inline(always)]
    fn reset(&mut self) {
        self.prev_cost = None;
        self.candidate = None;
        self.last_pos = None;
    }
}

pub struct MyersSearchState<B: SimdBackend> {
    // Myers bit-parallel state vectors
    pv: Vec<B::Simd>,
    mv: Vec<B::Simd>,
    score: Vec<B::Simd>,
    best_adjusted: Vec<B::Simd>,

    // Positions mode state
    traced_matches: Vec<WrappedSassyMatch>,
    results: Vec<MatchInfo>,
    states: Vec<QueryState>,
    sassy_cost_matrix: CostMatrix,
}

impl<B: SimdBackend> MyersSearchState<B> {
    pub fn new() -> Self {
        Self {
            pv: Vec::new(),
            mv: Vec::new(),
            score: Vec::new(),
            best_adjusted: Vec::new(),
            traced_matches: Vec::new(),
            results: Vec::new(),
            states: Vec::new(),
            sassy_cost_matrix: CostMatrix::default(),
        }
    }

    fn reset(&mut self, nq: usize, m: usize, initial_adjusted: i64) {
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
    }
}

impl<B: SimdBackend> Default for MyersSearchState<B> {
    fn default() -> Self {
        Self::new()
    }
}

// Core algo

#[derive(Copy, Clone)]
struct SearchContext<B: SimdBackend> {
    all_ones: B::Simd,
    zero_v: B::Simd,
    one_v: B::Simd,
    mask_vec: B::Simd,
    scale_shift: u32,
}

/// Advances the Myers bit-parallel algorithm by one column
#[inline(always)]
fn advance_column<B: SimdBackend>(
    pv: &mut B::Simd,
    mv: &mut B::Simd,
    score: &mut B::Simd,
    eq: B::Simd,
    adjust_vec: B::Simd,
    ctx: SearchContext<B>,
) -> B::Simd {
    let pv_val = *pv;
    let mv_val = *mv;

    // Myers bit-parallel computation
    let xv = eq | mv_val;
    let xh = (((eq & pv_val) + pv_val) ^ pv_val) | eq;
    let ph = mv_val | (ctx.all_ones ^ (xh | pv_val));
    let mh = pv_val & xh;

    // Extract high bits
    let ph_bit_mask = (ph & ctx.mask_vec).simd_eq(ctx.zero_v);
    let ph_bit = (ctx.all_ones ^ ph_bit_mask) & ctx.one_v;
    let mh_bit_mask = (mh & ctx.mask_vec).simd_eq(ctx.zero_v);
    let mh_bit = (ctx.all_ones ^ mh_bit_mask) & ctx.one_v;

    // Update state
    let ph_shift = ph << 1;
    *pv = (mh << 1) | (ctx.all_ones ^ (xv | ph_shift));
    *mv = ph_shift & xv;

    // Update and return adjusted score
    *score = *score + (ph_bit - mh_bit);
    (*score << ctx.scale_shift) - adjust_vec
}

/// Handles match detection and local minimum tracking for positions mode
#[inline(always)]
fn handle_match<B: SimdBackend>(state: &mut MyersSearchState<B>, match_info: MatchInfo) {
    let query_idx = match_info.query_idx;
    if query_idx >= state.states.len() {
        return;
    }

    let query_state = &mut state.states[query_idx];

    // Check for gap in positions (new match region)
    if let Some(prev_pos) = query_state.last_pos {
        if match_info.pos > prev_pos + 1 {
            if let Some(candidate) = query_state.candidate.take() {
                state.results.push(candidate);
            }
            query_state.prev_cost = None;
        }
    }

    // Track local minimum within contiguous match region
    if let Some(prev) = query_state.prev_cost {
        if match_info.cost + f32::EPSILON < prev {
            // Found new minimum
            query_state.candidate = Some(match_info);
        } else if (match_info.cost - prev).abs() <= f32::EPSILON {
            // Same cost - keep rightmost position
            match &query_state.candidate {
                Some(current) => {
                    if (match_info.cost - current.cost).abs() <= f32::EPSILON
                        || match_info.cost + f32::EPSILON < current.cost
                    {
                        query_state.candidate = Some(match_info);
                    }
                }
                None => {
                    query_state.candidate = Some(match_info);
                }
            }
        } else if match_info.cost > prev + f32::EPSILON {
            // Cost increasing - end of local minimum
            if let Some(candidate) = query_state.candidate.take() {
                state.results.push(candidate);
            }
            query_state.candidate = None;
        }
    } else {
        query_state.candidate = Some(match_info);
    }

    query_state.prev_cost = Some(match_info.cost);
    query_state.last_pos = Some(match_info.pos);
}

//  SEARCH

/// Common search parameters
struct SearchParams {
    alpha: f32,
    alpha_scaled: i32,
    k_scaled: i32,
    extra_penalty_scaled: i32,
    inv_scale: f32,
}

impl SearchParams {
    fn new(alpha: Option<f32>, k: u8) -> Self {
        const SCALE_SHIFT: u32 = 8;
        const SCALE: i32 = 1 << SCALE_SHIFT;

        let alpha = alpha.unwrap_or(1.0).clamp(0.0, 1.0);
        let alpha_scaled = ((alpha * (SCALE as f32)).round() as i32).clamp(0, SCALE);
        let k_scaled = (k as i32) << SCALE_SHIFT;
        let extra_penalty_scaled = SCALE - alpha_scaled;
        let inv_scale = 1.0f32 / (SCALE as f32);

        Self {
            alpha,
            alpha_scaled,
            k_scaled,
            extra_penalty_scaled,
            inv_scale,
        }
    }
}

#[inline(always)]
fn search_impl_scan<B: SimdBackend>(
    state: &mut MyersSearchState<B>,
    encoded: &TQueries<B>,
    target: &[u8],
    k: u8,
    alpha: Option<f32>,
) -> Vec<f32> {
    let nq = encoded.n_queries;
    let n_original_queries = encoded.n_original_queries;

    if nq == 0 {
        return vec![-1.0; n_original_queries];
    }

    let m = encoded.query_length;
    assert!(
        m > 0 && m <= B::LIMB_BITS,
        "query length must be 1..={}",
        B::LIMB_BITS
    );

    let params = SearchParams::new(alpha, k);
    let initial_adjusted = (m as i64) * (params.alpha_scaled as i64);
    state.reset(nq, m, initial_adjusted);

    let vectors_in_block = nq.div_ceil(B::LANES);
    let ctx = create_context::<B>(m);
    let k_scaled_vec = B::splat_from_i64(params.k_scaled as i64);

    if params.alpha_scaled < (1 << 8) {
        process_prefix_overhang::<B>(state, m, &params, &ctx, vectors_in_block);
    }

    // Main loop over target
    process_target_scan::<B>(state, encoded, target, &ctx, vectors_in_block);

    // Overhang trailing positions
    if params.alpha_scaled < (1 << 8) {
        process_overhang_scan::<B>(state, m, &params, &ctx, vectors_in_block);
    }

    // Extract and return minimum costs
    extract_minimum_costs::<B>(
        state,
        nq,
        n_original_queries,
        k_scaled_vec,
        params.inv_scale,
    )
}

#[inline(always)]
fn search_impl_positions<B: SimdBackend>(
    state: &mut MyersSearchState<B>,
    encoded: &TQueries<B>,
    target: &[u8],
    k: u8,
    alpha: Option<f32>,
) -> Vec<WrappedSassyMatch> {
    let nq = encoded.n_queries;
    let n_original_queries = encoded.n_original_queries;

    if nq == 0 {
        return Vec::new();
    }

    let m = encoded.query_length;
    assert!(
        m > 0 && m <= B::LIMB_BITS,
        "query length must be 1..={}",
        B::LIMB_BITS
    );

    let params = SearchParams::new(alpha, k);
    let initial_adjusted = (m as i64) * (params.alpha_scaled as i64);
    state.reset(nq, m, initial_adjusted);

    // Initialize positions mode state
    state.traced_matches.clear();
    state.results.clear();
    if state.states.len() < nq {
        state.states.resize_with(nq, QueryState::default);
    }
    for s in state.states.iter_mut().take(nq) {
        s.reset();
    }

    let vectors_in_block = nq.div_ceil(B::LANES);
    let ctx = create_context::<B>(m);
    let k_scaled_vec = B::splat_from_i64(params.k_scaled as i64);

    if params.alpha_scaled < (1 << 8) {
        process_prefix_overhang::<B>(state, m, &params, &ctx, vectors_in_block);
    }

    // Main loop over target
    process_target_positions::<B>(
        state,
        encoded,
        target,
        &ctx,
        k_scaled_vec,
        vectors_in_block,
        nq,
        n_original_queries,
        params.inv_scale,
        None,
    );

    // Overhang trailing positions
    if params.alpha_scaled < (1 << 8) {
        process_overhang_positions::<B>(
            state,
            target.len(),
            m,
            &params,
            &ctx,
            k_scaled_vec,
            vectors_in_block,
            nq,
            n_original_queries,
        );
    }

    finalize_and_traceback::<B>(state, encoded, target, params.alpha, nq, n_original_queries)
}

// Search helpers

fn create_context<B: SimdBackend>(m: usize) -> SearchContext<B> {
    const SCALE_SHIFT: u32 = 8;
    let high_bit: u64 = 1u64 << (m - 1);

    SearchContext {
        all_ones: B::splat_all_ones(),
        zero_v: B::splat_zero(),
        one_v: B::splat_one(),
        mask_vec: B::splat_from_i64(high_bit as i64),
        scale_shift: SCALE_SHIFT,
    }
}

#[inline(always)]
fn process_target_scan<B: SimdBackend>(
    state: &mut MyersSearchState<B>,
    encoded: &TQueries<B>,
    target: &[u8],
    ctx: &SearchContext<B>,
    vectors_in_block: usize,
) {
    let adjust_vec = B::splat_zero();

    for &tb in target.iter() {
        let encoded_char = crate::iupac::get_encoded(tb);
        let peq_slice = unsafe { encoded.peqs.get_unchecked(encoded_char as usize) };

        unsafe {
            for block_idx in 0..vectors_in_block {
                let eq = *peq_slice.get_unchecked(block_idx);
                let adjusted = advance_column(
                    state.pv.get_unchecked_mut(block_idx),
                    state.mv.get_unchecked_mut(block_idx),
                    state.score.get_unchecked_mut(block_idx),
                    eq,
                    adjust_vec,
                    *ctx,
                );

                let best_slot = state.best_adjusted.get_unchecked_mut(block_idx);
                *best_slot = B::min(*best_slot, adjusted);
            }
        }
    }
}

#[inline(always)]
fn process_prefix_overhang<B: SimdBackend>(
    state: &mut MyersSearchState<B>,
    m: usize,
    params: &SearchParams,
    ctx: &SearchContext<B>,
    vectors_in_block: usize,
) {
    // not working yet
}

#[inline(always)]
fn process_overhang_scan<B: SimdBackend>(
    state: &mut MyersSearchState<B>,
    m: usize,
    params: &SearchParams,
    ctx: &SearchContext<B>,
    vectors_in_block: usize,
) {
    let zero_eq_vec = B::splat_zero();

    for trailing in 1..=m {
        let adjust_vec =
            B::splat_from_i64((trailing as i64) * (params.extra_penalty_scaled as i64));

        unsafe {
            for block_idx in 0..vectors_in_block {
                let adjusted = advance_column(
                    state.pv.get_unchecked_mut(block_idx),
                    state.mv.get_unchecked_mut(block_idx),
                    state.score.get_unchecked_mut(block_idx),
                    zero_eq_vec,
                    adjust_vec,
                    *ctx,
                );

                let best_slot = state.best_adjusted.get_unchecked_mut(block_idx);
                *best_slot = B::min(*best_slot, adjusted);
            }
        }
    }
}

#[inline(always)]
fn extract_minimum_costs<B: SimdBackend>(
    state: &MyersSearchState<B>,
    nq: usize,
    n_original_queries: usize,
    k_scaled_vec: B::Simd,
    inv_scale: f32,
) -> Vec<f32> {
    let all_ones = B::splat_all_ones();
    let neg_mask = B::splat_scalar(B::MAX_POSITIVE);
    let mut result = vec![-1.0f32; n_original_queries];

    for (block_idx, &min_adj) in state.best_adjusted.iter().enumerate() {
        let mask = all_ones ^ B::simd_gt(min_adj, k_scaled_vec);
        let selected = B::blend(mask, min_adj, neg_mask);
        let base = block_idx * B::LANES;
        let end = (base + B::LANES).min(nq);
        let selected_arr = B::to_array(selected);
        let selected_slice = selected_arr.as_ref();

        for (lane_idx, &val) in selected_slice[..(end - base)].iter().enumerate() {
            if val != B::MAX_POSITIVE {
                let query_idx = base + lane_idx;
                let cost = B::scalar_to_f32(val) * inv_scale;
                let orig_idx = query_idx % n_original_queries;

                if result[orig_idx] < 0.0 || cost < result[orig_idx] {
                    result[orig_idx] = cost;
                }
            }
        }
    }
    result
}

#[inline(always)]
fn process_target_positions<B: SimdBackend>(
    state: &mut MyersSearchState<B>,
    encoded: &TQueries<B>,
    target: &[u8],
    ctx: &SearchContext<B>,
    k_scaled_vec: B::Simd,
    vectors_in_block: usize,
    nq: usize,
    n_original_queries: usize,
    inv_scale: f32,
    target_block_idx: Option<usize>, // used in tracing
) {
    let adjust_vec = B::splat_zero();

    for (idx, &tb) in target.iter().enumerate() {
        let encoded_char = crate::iupac::get_encoded(tb);
        let peq_slice = unsafe { encoded.peqs.get_unchecked(encoded_char as usize) };
        let pos = idx as i32;

        unsafe {
            // If we have a specific target block we have to trace it
            if let Some(block_idx) = target_block_idx {
                let eq = *peq_slice.get_unchecked(block_idx);
                let adjusted = advance_column(
                    state.pv.get_unchecked_mut(block_idx),
                    state.mv.get_unchecked_mut(block_idx),
                    state.score.get_unchecked_mut(block_idx),
                    eq,
                    adjust_vec,
                    *ctx,
                );

                collect_matches::<B>(
                    state,
                    adjusted,
                    k_scaled_vec,
                    block_idx,
                    nq,
                    pos,
                    n_original_queries,
                    inv_scale,
                );
            } else {
                // In this case we are just aligning so to all blocks
                for block_idx in 0..vectors_in_block {
                    let eq = *peq_slice.get_unchecked(block_idx);
                    let adjusted = advance_column(
                        state.pv.get_unchecked_mut(block_idx),
                        state.mv.get_unchecked_mut(block_idx),
                        state.score.get_unchecked_mut(block_idx),
                        eq,
                        adjust_vec,
                        *ctx,
                    );

                    collect_matches::<B>(
                        state,
                        adjusted,
                        k_scaled_vec,
                        block_idx,
                        nq,
                        pos,
                        n_original_queries,
                        inv_scale,
                    );
                }
            }
        }
    }
}

#[inline(always)]
pub fn trace_idea_scalar<B: SimdBackend>(
    state: &mut MyersSearchState<B>,
    encoded: &TQueries<B>,
    target: &[u8],
    suffix_overhang: usize, // difference between entire target and returned position by search
    k: u8,
    alpha: Option<f32>,
    target_query_idx: usize, // we only need to process the block for this query index, we still waste LANES -1 here!
) -> Vec<WrappedSassyMatch> {
    let nq = encoded.n_queries;
    let n_original_queries = encoded.n_original_queries;
    let m = encoded.query_length;
    assert!(
        m > 0 && m <= B::LIMB_BITS,
        "query length must be 1..={}",
        B::LIMB_BITS
    );

    let params = SearchParams::new(alpha, k);
    let initial_adjusted = (m as i64) * (params.alpha_scaled as i64);
    state.reset(nq, m, initial_adjusted);

    state.traced_matches.clear();
    state.results.clear();
    if state.states.len() < nq {
        state.states.resize_with(nq, QueryState::default);
    }
    for s in state.states.iter_mut().take(nq) {
        s.reset();
    }

    let block_idx = target_query_idx.div_ceil(B::LANES);

    let ctx = create_context::<B>(m);
    let mut deltas: Vec<(B::Simd, B::Simd, B::Simd, B::Simd)> =
        Vec::with_capacity(target.len() + suffix_overhang);
    let zero_eq_vec = B::splat_zero();
    let mut adjust_vec = B::splat_zero();
    let k_scaled_vec = B::splat_from_i64(params.k_scaled as i64);

    for (idx, &tb) in target.iter().enumerate() {
        let encoded_char = crate::iupac::get_encoded(tb);
        let peq_slice = unsafe { encoded.peqs.get_unchecked(encoded_char as usize) };
        let pos = idx as i32;

        unsafe {
            let eq = *peq_slice.get_unchecked(block_idx);
            let adjusted = advance_column(
                state.pv.get_unchecked_mut(block_idx),
                state.mv.get_unchecked_mut(block_idx),
                state.score.get_unchecked_mut(block_idx),
                eq,
                adjust_vec,
                ctx,
            );
            deltas.push((
                *state.pv.get_unchecked(block_idx),
                *state.mv.get_unchecked(block_idx),
                *state.score.get_unchecked(block_idx),
                adjusted,
            ));

            collect_matches::<B>(
                state,
                adjusted,
                k_scaled_vec,
                block_idx,
                nq,
                pos,
                n_original_queries,
                params.inv_scale,
            );
        }
    }

    if params.alpha_scaled < (1 << 8) {
        for trailing in 1..=suffix_overhang {
            adjust_vec =
                B::splat_from_i64((trailing as i64) * (params.extra_penalty_scaled as i64));
            let pos: i32 = (target.len() + trailing - 1) as i32;

            unsafe {
                let adjusted = advance_column(
                    state.pv.get_unchecked_mut(block_idx),
                    state.mv.get_unchecked_mut(block_idx),
                    state.score.get_unchecked_mut(block_idx),
                    zero_eq_vec,
                    adjust_vec,
                    ctx,
                );

                deltas.push((
                    *state.pv.get_unchecked(block_idx),
                    *state.mv.get_unchecked(block_idx),
                    *state.score.get_unchecked(block_idx),
                    adjusted,
                ));

                collect_matches::<B>(
                    state,
                    adjusted,
                    k_scaled_vec,
                    block_idx,
                    nq,
                    pos,
                    n_original_queries,
                    params.inv_scale,
                );
            }
        }
    }

    {
        let target_lane = target_query_idx % B::LANES;
        for (step_idx, (pv, mv, score, adjusted)) in deltas.iter().enumerate() {
            let pv_arr = B::to_array(*pv);
            let mv_arr = B::to_array(*mv);
            let score_arr = B::to_array(*score);
            let adj_arr = B::to_array(*adjusted);

            let pv_lane = pv_arr.as_ref()[target_lane];
            let mv_lane = mv_arr.as_ref()[target_lane];
            let score_lane = score_arr.as_ref()[target_lane];
            let adj_lane = adj_arr.as_ref()[target_lane];

            let pv_lane_i64 = B::scalar_to_i64(pv_lane);
            let mv_lane_i64 = B::scalar_to_i64(mv_lane);
            let score_lane_i64 = B::scalar_to_i64(score_lane);
            let adj_lane_i64 = B::scalar_to_i64(adj_lane);
            let cost_f32 = (adj_lane_i64 as f32) * params.inv_scale;

            println!(
                "step {:3}: pv_lane={} mv_lane={} raw_score={} adjusted_scaled={} cost={}",
                step_idx, pv_lane_i64, mv_lane_i64, score_lane_i64, adj_lane_i64, cost_f32
            );
        }
    }

    vec![]
}

#[inline(always)]
fn process_overhang_positions<B: SimdBackend>(
    state: &mut MyersSearchState<B>,
    target_len: usize,
    m: usize,
    params: &SearchParams,
    ctx: &SearchContext<B>,
    k_scaled_vec: B::Simd,
    vectors_in_block: usize,
    nq: usize,
    n_original_queries: usize,
) {
    let zero_eq_vec = B::splat_zero();

    for trailing in 1..=m {
        let adjust_vec =
            B::splat_from_i64((trailing as i64) * (params.extra_penalty_scaled as i64));
        let pos = (target_len + trailing - 1) as i32;

        unsafe {
            for block_idx in 0..vectors_in_block {
                let adjusted = advance_column(
                    state.pv.get_unchecked_mut(block_idx),
                    state.mv.get_unchecked_mut(block_idx),
                    state.score.get_unchecked_mut(block_idx),
                    zero_eq_vec,
                    adjust_vec,
                    *ctx,
                );

                collect_matches::<B>(
                    state,
                    adjusted,
                    k_scaled_vec,
                    block_idx,
                    nq,
                    pos,
                    n_original_queries,
                    params.inv_scale,
                );
            }
        }
    }
}

#[inline(always)]
fn collect_matches<B: SimdBackend>(
    state: &mut MyersSearchState<B>,
    adjusted: B::Simd,
    k_scaled_vec: B::Simd,
    block_idx: usize,
    nq: usize,
    pos: i32,
    n_original_queries: usize,
    inv_scale: f32,
) {
    let adjusted_arr = B::to_array(adjusted);
    let adjusted_slice = adjusted_arr.as_ref();
    let k_scaled_arr = B::to_array(k_scaled_vec);
    let k_scaled_i64 = B::scalar_to_i64(k_scaled_arr.as_ref()[0]);
    let base = block_idx * B::LANES;
    let end = (base + B::LANES).min(nq);

    for (i, &score_scaled) in adjusted_slice[..(end - base)].iter().enumerate() {
        let score_scaled_i64 = B::scalar_to_i64(score_scaled);

        if score_scaled_i64 <= k_scaled_i64 {
            let query_idx = base + i;
            let cost = (score_scaled_i64 as f32) * inv_scale;
            let strand = if query_idx >= n_original_queries {
                Strand::Rc
            } else {
                Strand::Fwd
            };
            let match_info = MatchInfo {
                query_idx,
                cost,
                pos,
                strand,
            };
            handle_match(state, match_info);
        }
    }
}

fn finalize_and_traceback<B: SimdBackend>(
    state: &mut MyersSearchState<B>,
    encoded: &TQueries<B>,
    target: &[u8],
    alpha: f32,
    nq: usize,
    n_original_queries: usize,
) -> Vec<WrappedSassyMatch> {
    // Flush remaining candidates
    for s in state.states.iter_mut().take(nq) {
        if let Some(candidate) = s.candidate.take() {
            state.results.push(candidate);
        }
        s.reset();
    }

    // Perform traceback for each candidate
    state.sassy_cost_matrix.alpha = Some(alpha);

    for candidate in &state.results {
        let query_seq = encoded.get_query_seq(candidate.query_idx);
        let overhanged_text_end = candidate.pos as usize + 1;
        let inbound_text_end = overhanged_text_end.min(target.len());
        let text_start = inbound_text_end.saturating_sub(query_seq.len() * 2).max(0);
        let text_slice = &target[text_start..inbound_text_end];
        let original_query_idx = candidate.query_idx % n_original_queries;

        fill::<Iupac>(
            query_seq,
            text_slice,
            overhanged_text_end + 1,
            &mut state.sassy_cost_matrix,
            Some(alpha),
        );

        let mut m = sassy::get_trace::<Iupac>(
            query_seq,
            text_start,
            overhanged_text_end,
            text_slice,
            &state.sassy_cost_matrix,
            Some(alpha),
        );

        m.strand = if candidate.query_idx >= n_original_queries {
            sassy::Strand::Rc
        } else {
            sassy::Strand::Fwd
        };

        state
            .traced_matches
            .push(WrappedSassyMatch::new(m, original_query_idx));
    }

    std::mem::take(&mut state.traced_matches)
}

#[cfg(test)]
mod tests {
    use super::{Searcher, WrappedSassyMatch};
    use crate::backend::{U32, U64};

    #[test]
    fn test_iupac_query_matches_standard_target() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"n".to_vec()];
        let encoded = searcher.encode(&queries, false);
        assert_eq!(searcher.scan(&encoded, b"A", 0, None), vec![0.0]);
        assert_eq!(searcher.scan(&encoded, b"a", 0, None), vec![0.0]);
    }

    #[test]
    fn test_iupac_target_matches_standard_query() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"A".to_vec()];
        let encoded = searcher.encode(&queries, false);
        assert_eq!(searcher.scan(&encoded, b"N", 0, None), vec![0.0]);
        assert_eq!(searcher.scan(&encoded, b"n", 0, None), vec![0.0]);
    }

    #[test]
    fn test_iupac_mismatch_requires_edit() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"R".to_vec()];
        let encoded = searcher.encode(&queries, false);
        assert_eq!(searcher.scan(&encoded, b"C", 0, None), vec![-1.0]);
        assert_eq!(searcher.scan(&encoded, b"C", 1, None), vec![1.0]);
        assert_eq!(searcher.scan(&encoded, b"c", 1, None), vec![1.0]);
    }

    #[test]
    #[should_panic(expected = "invalid IUPAC character")]
    fn test_invalid_query_panics() {
        let searcher = Searcher::<U32>::new();
        let queries = vec![b"AZ".to_vec()];
        let _ = searcher.encode(&queries, false);
    }

    #[test]
    fn test_mini_search() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let result = searcher.scan(&encoded, b"CCCCCCCCCATGCCCCC", 4, None);
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    fn test_double_search() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let result = searcher.scan(&encoded, b"CCCTTGCCCCCCATGCCCCC", 4, None);
        assert_eq!(result, vec![0.0, 0.0]);
    }

    #[test]
    fn test_edit() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATCAGA".to_vec()];
        let encoded = searcher.encode(&queries, false);

        // 1 edit
        let result = searcher.scan(&encoded, b"ATCTGA", 4, None);
        assert_eq!(result, vec![1.0]);

        // 2 edits
        let result = searcher.scan(&encoded, b"GTCTGA", 4, None);
        assert_eq!(result, vec![2.0]);

        // 3 edits (1 del)
        let result = searcher.scan(&encoded, b"GTTGA", 4, None);
        assert_eq!(result, vec![3.0]);

        // Match should not be recovered when k == 1
        let result = searcher.scan(&encoded, b"GTTGA", 1, None);
        assert_eq!(result, vec![-1.0]);
    }

    #[test]
    fn test_lowest_edits_returned() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"GGGCATCGATGAC".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let target = b"CCCCCCCGGGCATCGATGACCCCCCCCCCCCCCCGGGCTTCGATGAC";
        let result = searcher.scan(&encoded, target, 4, None);
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    #[should_panic(expected = "All queries must have the same length")]
    fn test_error_unequal_lengths() {
        let searcher = Searcher::<U32>::new();
        let queries = vec![b"GGGCATCGATGAC".to_vec(), b"AAA".to_vec()];
        let _ = searcher.encode(&queries, false);
    }

    #[test]
    fn read_example() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let target = b"CCCTCGCCCCCCATGCCCCC";
        let result = searcher.scan(&encoded, target, 4, None);
        println!("Result: {:?}", result);
        assert_eq!(result, vec![0.0, 1.0]);
    }

    #[test]
    fn test_mini_search_with_positions() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let target = b"CCCCCCCCCATGCCCCC";
        let result = searcher.search(&encoded, target, 4, None);

        println!("Result: {:?}", result);

        // Should find at least one match
        assert!(!result.is_empty());
        // Find the exact match (cost 0)
        let exact_matches = result
            .iter()
            .filter(|m| m.sassy_match.cost == 0)
            .collect::<Vec<_>>();

        assert!(!exact_matches.is_empty());
        assert_eq!(exact_matches[0].query_idx, 0);
        assert_eq!(exact_matches[0].sassy_match.text_end, 12); // Position where match ends (exclusive)
    }

    #[test]
    fn test_positions_multiple_occurrences() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let target = b"ATGCCCATGCCCATG";
        let result = searcher.search(&encoded, target, 0, None);

        // Should find all 3 exact matches
        let exact_matches = result
            .iter()
            .filter(|m| m.sassy_match.cost == 0)
            .collect::<Vec<_>>();

        assert_eq!(exact_matches.len(), 3);
        assert_eq!(exact_matches[0].sassy_match.text_end, 3); // Position where match ends (exclusive)
        assert_eq!(exact_matches[1].sassy_match.text_end, 9); // Position where match ends (exclusive)
        assert_eq!(exact_matches[2].sassy_match.text_end, 15); // Position where match ends (exclusive)
    }

    #[test]
    fn test_positions_local_minima_only() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![vec![b'A'; 6]];
        let encoded = searcher.encode(&queries, false);

        let mut target = Vec::new();
        target.extend_from_slice(b"TTT");
        for _ in 0..6 {
            target.push(b'A');
        }
        target.extend_from_slice(b"TTT");

        let results = searcher.search(&encoded, &target, 3, None);
        assert_eq!(results.len(), 1);
        assert!(results[0].sassy_match.cost == 0);
        assert_eq!(results[0].sassy_match.text_end, 9); // Position where match ends (exclusive)
    }

    #[test]
    fn test_positions_plateau_keeps_rightmost() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![vec![b'A'; 6]];
        let encoded = searcher.encode(&queries, false);

        let mut target = Vec::new();
        target.extend_from_slice(b"TTT");
        for _ in 0..7 {
            target.push(b'A');
        }
        target.extend_from_slice(b"TTT");

        println!("queries: {:?}", String::from_utf8_lossy(&queries[0]));
        println!("target: {:?}", String::from_utf8_lossy(&target));

        let results = searcher.search(&encoded, &target, 3, None);
        assert_eq!(results.len(), 1);
        assert!(results[0].sassy_match.cost == 0);
        assert_eq!(results[0].sassy_match.text_end, 10); // Position where match ends (exclusive)
    }

    #[test]
    fn test_positions_plateau_at_end_flushes() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![vec![b'A'; 6]];
        let encoded = searcher.encode(&queries, false);

        let mut target = Vec::new();
        target.extend_from_slice(b"TTT");
        for _ in 0..7 {
            target.push(b'A');
        }

        println!("queries: {:?}", String::from_utf8_lossy(&queries[0]));
        println!("target: {:?}", String::from_utf8_lossy(&target));

        let results = searcher.search(&encoded, &target, 3, None);
        assert_eq!(results.len(), 1);
        assert!(results[0].sassy_match.cost == 0);
        assert_eq!(results[0].sassy_match.text_end, 10); // Position where match ends (exclusive)
    }

    #[test]
    fn test_local_min_double_match() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![vec![b'A'; 6]];
        let encoded = searcher.encode(&queries, false);

        let mut target = std::iter::repeat_n(b'T', 100).collect::<Vec<_>>();
        // Insert query at position 10 and 50
        let query = queries[0].clone();
        target[10..10 + query.len()].copy_from_slice(&query);
        target[50..50 + query.len()].copy_from_slice(&query);

        let results = searcher.search(&encoded, &target, 3, None);
        assert_eq!(results[0].sassy_match.text_end, 16); // Position where match ends (exclusive)
        assert_eq!(results[1].sassy_match.text_end, 56); // Position where match ends (exclusive)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_positions_match_costs() {
        let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
        let target = b"CCCTTGCCCCCCATGCCCCC";

        let mut scan_searcher = Searcher::<U32>::new();
        let mut pos_searcher = Searcher::<U32>::new();
        let encoded_scan = scan_searcher.encode(&queries, false);
        let encoded_pos = pos_searcher.encode(&queries, false);

        let result_scan = scan_searcher.scan(&encoded_scan, target, 4, None);
        let result_pos = pos_searcher.search(&encoded_pos, target, 4, None);

        // Check that the minimum cost in positions matches scan result
        for (query_idx, _) in queries.iter().enumerate() {
            let min_cost_in_positions = result_pos
                .iter()
                .filter(|m| m.query_idx == query_idx)
                .map(|m| m.sassy_match.cost)
                .fold(i32::MAX, |a, b| a.min(b));
            assert!(((result_scan[query_idx].ceil() as i32) - min_cost_in_positions) == 0);
        }
    }

    #[test]
    fn test_positions_no_match() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"AAAAA".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let result = searcher.search(&encoded, b"CCCCCCCCCCCCC", 1, None);

        // Should have no matches for query 0 with cost <= 1
        let matches_q0: Vec<_> = result.iter().filter(|m| m.query_idx == 0).collect();
        assert_eq!(matches_q0.len(), 0);
    }

    #[test]
    fn test_positions_multiple_queries() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![
            b"AAAA".to_vec(),
            b"TTTT".to_vec(),
            b"GGGG".to_vec(),
            b"CCCC".to_vec(),
        ];
        let encoded = searcher.encode(&queries, false);
        let result = searcher.search(&encoded, b"AAGGGGTTTTCCCC", 2, None);

        // All queries should have at least one match with k=2
        for query_idx in 0..4 {
            let query_matches: Vec<_> =
                result.iter().filter(|m| m.query_idx == query_idx).collect();
            assert!(!query_matches.is_empty());
            // All matches should be within threshold
            for m in &query_matches {
                assert!(m.sassy_match.cost >= 0 && m.sassy_match.cost <= 2);
                assert!(m.sassy_match.text_end > 0);
            }
        }
    }

    #[test]
    fn test_overhang_half_penalty() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"AC".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let result = searcher.scan(&encoded, b"A", 1, Some(0.5));
        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - 0.5).abs() < 1e-6,
            "expected 0.5, got {}",
            result[0]
        );
    }

    #[test]
    fn test_overhang_left() {
        let q = b"CCGGGG";
        let t = b"GGGGAAAAAAAAAAAAAAA";
        let mut searcher = Searcher::<U32>::new();
        let encoded = searcher.encode(&vec![q.to_vec()], false);
        let results = searcher.search(&encoded, t, 2, Some(0.5));
        println!("results: {:?}", results);
        assert_ne!(results.len(), 0);
    }

    #[test]
    fn test_overhang_matches_standard_when_alpha_one() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let target = b"CCCCCCCCCATGCCCCC";
        let standard = searcher.scan(&encoded, target, 4, None);
        let overhang = searcher.scan(&encoded, target, 4, Some(1.0));
        assert_eq!(standard.len(), overhang.len());
        for (i, &val) in standard.iter().enumerate() {
            assert!((overhang[i] - val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_positions_overhang() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ACCC".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let results = searcher.search(&encoded, b"A", 2, Some(0.5));
        assert!(!results.is_empty());
        let min_cost_entry = results
            .iter()
            .min_by(|a, b| a.sassy_match.cost.partial_cmp(&b.sassy_match.cost).unwrap())
            .unwrap();
        assert_eq!(min_cost_entry.query_idx, 0);
        assert_eq!(min_cost_entry.sassy_match.cost, 1);
    }

    #[test]
    fn test_longer_overhang_left() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"AAACCC".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let results = searcher.search(&encoded, b"CCCGGGGGGGG", 4, Some(0.5));
        // Get minimum cost match
        let min_cost = results
            .iter()
            .map(|m| m.sassy_match.cost)
            .fold(i32::MAX, |a, b| a.min(b));
        assert_eq!(min_cost, 1);
    }

    #[test]
    fn test_longer_overhang_right() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"AAACCC".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let results = searcher.search(&encoded, b"GGGGGAAA", 4, Some(0.5));
        // Get minimum cost match
        let min_cost = results
            .iter()
            .map(|m| m.sassy_match.cost)
            .fold(i32::MAX, |a, b| a.min(b));
        assert_eq!(min_cost, 1);
    }

    #[test]
    fn test_overhang_with_positions() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"AAACCC".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let results = searcher.search(&encoded, b"GGGGGAAA", 4, Some(0.5));
        // Get minimum cost match
        let result = results
            .iter()
            .min_by(|a, b| a.sassy_match.cost.partial_cmp(&b.sassy_match.cost).unwrap())
            .unwrap();
        assert_eq!(result.query_idx, 0);
        assert_eq!(result.sassy_match.cost, 1);
    }

    #[test]
    fn test_searcher_u32_scan() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let target = b"CCCTCGCCCCCCATGCCCCC";
        let results = searcher.scan(&encoded, target, 4, None);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 0.0);
        assert_eq!(results[1], 1.0);
    }

    #[test]
    fn test_searcher_u64_scan() {
        let mut searcher = Searcher::<U64>::new();
        let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let target = b"CCCTCGCCCCCCATGCCCCC";
        let results = searcher.scan(&encoded, target, 4, None);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 0.0);
        assert_eq!(results[1], 1.0);
    }

    #[test]
    fn test_searcher_u32_positions() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let target = b"ATGCCCATGCCCATG";
        let results = searcher.search(&encoded, target, 0, None);

        // Should find all 3 exact matches
        let exact_matches: Vec<_> = results.iter().filter(|m| m.sassy_match.cost == 0).collect();
        assert_eq!(exact_matches.len(), 3);
        assert_eq!(exact_matches[0].sassy_match.text_end, 3);
        assert_eq!(exact_matches[1].sassy_match.text_end, 9);
        assert_eq!(exact_matches[2].sassy_match.text_end, 15);
    }

    #[test]
    fn test_searcher_u64_positions() {
        use std::iter::repeat_n;
        let mut searcher = Searcher::<U64>::new();
        let long_query = repeat_n(b'A', 64).collect::<Vec<u8>>();
        let queries = vec![long_query.clone()];
        let encoded = searcher.encode(&queries, false);
        let mut target = vec![b'C'; 200];
        let insert_pos = 100;
        target.splice(insert_pos..insert_pos, long_query.iter().cloned());
        let results = searcher.search(&encoded, &target, 4, None);
        println!("results: {:?}", results);

        assert!(!results.is_empty());
        let exact_matches: Vec<_> = results.iter().filter(|m| m.sassy_match.cost == 0).collect();
        assert!(!exact_matches.is_empty());
        assert_eq!(exact_matches[0].query_idx, 0);
        assert_eq!(exact_matches[0].sassy_match.text_end, 164);
    }

    #[test]
    fn test_searcher_with_overhang() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"AC".to_vec()];
        let encoded = searcher.encode(&queries, false);
        let target = b"A";

        // Test with overhang penalty
        let results = searcher.scan(&encoded, target, 1, Some(0.5));

        assert_eq!(results.len(), 1);
        assert!((results[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_full_32_bit_coverage() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![
            b"ATGATCATCTACGACTACTACCAATGCTAGCT".to_vec(),
            //12345678901234567890123456789012
        ];
        let encoded = searcher.encode(&queries, false);
        let target = b"CCCCCCCCCCCcATGATCATCTACGACTACTACCAATGCTAGCT";
        let results = searcher.scan(&encoded, target, 0, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 0.0);
    }

    #[test]
    fn test_rc_scan_forward_match() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries, true);
        let target = b"CCCCCCCCCATGCCCCC";
        let results = searcher.scan(&encoded, target, 0, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 0.0); // Should find forward match
    }

    #[test]
    fn test_rc_scan_reverse_match() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries, true);
        // RC of ATG is CAT
        let target = b"CCCCCCCCCATCCCCC";
        let results = searcher.scan(&encoded, target, 0, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 0.0); // Should find RC match
    }

    #[test]
    fn test_rc_scan_both_matches() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries, true);
        // Target has both ATG and CAT (RC of ATG)
        let target = b"ATGCCCCCCAT";
        let results = searcher.scan(&encoded, target, 0, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 0.0); // Should find minimum (which is 0.0 for both)
    }

    #[test]
    fn test_rc_scan_prefers_better_match() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries, true);
        // Forward has 1 mismatch (TTG), RC (CAT) has exact match
        let target = b"TTGCCCCCCAT";
        let results = searcher.scan(&encoded, target, 2, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 0.0); // Should prefer RC's exact match over forward's 1-edit
    }

    #[test]
    fn test_rc_positions_forward_match() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries, true);
        let target = b"CCCCCCCCCATGCCCCC";
        let results = searcher.search(&encoded, target, 0, None);

        let exact_matches: Vec<_> = results.iter().filter(|m| m.sassy_match.cost == 0).collect();
        assert!(!exact_matches.is_empty());
        assert_eq!(exact_matches[0].query_idx, 0);
        assert_eq!(exact_matches[0].sassy_match.strand, sassy::Strand::Fwd);
        assert_eq!(exact_matches[0].sassy_match.text_end, 12);
    }

    #[test]
    fn test_rc_positions_reverse_match() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries, true);
        // RC of ATG is CAT
        let target = b"CCCCCCCCCATCCCCC";
        let results = searcher.search(&encoded, target, 0, None);

        let exact_matches: Vec<_> = results.iter().filter(|m| m.sassy_match.cost == 0).collect();
        assert!(!exact_matches.is_empty());
        assert_eq!(exact_matches[0].query_idx, 0);
        assert_eq!(exact_matches[0].sassy_match.strand, sassy::Strand::Rc);
        assert_eq!(exact_matches[0].sassy_match.text_end, 11); // CAT ends at position 10
    }

    #[test]
    fn test_rc_positions_both_matches() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries, true);
        // Target has both ATG and CAT (RC of ATG)
        let target = b"ATGCCCCCCAT";
        let results = searcher.search(&encoded, target, 0, None);
        println!("results: {:?}", results);

        let exact_matches: Vec<_> = results.iter().filter(|m| m.sassy_match.cost == 0).collect();
        assert_eq!(exact_matches.len(), 2); // Should find both

        // // Check that we have one Fwd and one Rc match
        // let fwd_count = exact_matches
        //     .iter()
        //     .filter(|m| m.sassy_match.strand == sassy::Strand::Fwd)
        //     .count();
        // let rc_count = exact_matches
        //     .iter()
        //     .filter(|m| m.sassy_match.strand == sassy::Strand::Rc)
        //     .count();
        // assert_eq!(fwd_count, 1);
        // assert_eq!(rc_count, 1);
    }

    #[test]
    fn test_rc_multiple_queries() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
        let encoded = searcher.encode(&queries, true);
        // Target has CAT (RC of ATG) and CAA (RC of TTG)
        let target = b"CATCCCCCCAA";
        let results = searcher.scan(&encoded, target, 0, None);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 0.0); // ATG found via RC (CAT)
        assert_eq!(results[1], 0.0); // TTG found via RC (CAA)
    }

    #[test]
    fn test_rc_disabled() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"ATG".to_vec()];
        let encoded = searcher.encode(&queries, false);
        // Only has RC (CAT), should not match when RC disabled
        let target = b"CCCCCCCCCATCCCCC";
        let results = searcher.scan(&encoded, target, 0, None);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], -1.0); // No match found
    }

    #[test]
    fn test_trace_idea_scalar() {
        let mut searcher = Searcher::<U32>::new();
        let queries = vec![b"CCGGG".to_vec()];
        let target = b"GGGAAAA"; // Ends at match (we need that to trace)
        let encoded = searcher.encode(&queries, false);

        println!("\n=== Testing CCGGG vs GGGAAAA ===");
        println!("Query: CCGGG (len=5)");
        println!("Target: GGGAAAA (len=7)");
        println!("Expected: CC overhang (2*0.5=1.0) + GGG exact match = 1.0 total");

        // Test with k=1 (should fail)
        let matches_k1 = searcher.scan(&encoded, target, 1, Some(0.5));
        println!("k=1 result: {:?}", matches_k1);

        // Test with k=2 (should succeed with cost 1.0)
        let matches_k2 = searcher.scan(&encoded, target, 2, Some(0.5));
        println!("k=2 result: {:?}", matches_k2);

        // Also test positions mode to see what's happening
        let pos_matches = searcher.search(&encoded, target, 1, Some(0.5));
        for m in pos_matches.iter() {
            println!(
                "Position match: pos={}, cost={}",
                m.sassy_match.text_end, m.sassy_match.cost
            );
        }

        assert_eq!(
            matches_k2[0], 1.0,
            "Expected cost 1.0 (2 chars overhang at 0.5 each)"
        );
    }
}
