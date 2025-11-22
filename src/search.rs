use crate::backend::SimdBackend;
use crate::tqueries::TQueries;

pub struct Searcher<B: SimdBackend> {
    vp: Vec<B::Simd>,
    vn: Vec<B::Simd>,
    score: Vec<B::Simd>,
    failure_masks: Vec<B::Simd>, // 1 = no match <=k (i.e. >k), 0 = match <=k
    results: Vec<bool>,
    alpha_pattern: u64,
}

impl<B: SimdBackend> Default for Searcher<B> {
    fn default() -> Self {
        Self::new(None)
    }
}

impl<B: SimdBackend> Searcher<B> {
    pub fn new(alpha: Option<f32>) -> Self {
        let alpha_val = alpha.unwrap_or(1.0); // default uses 111.. start penalties (i.e. no overhang reduction)
        Self {
            vp: Vec::new(),
            vn: Vec::new(),
            score: Vec::new(),
            failure_masks: Vec::new(),
            results: Vec::new(),
            alpha_pattern: Self::generate_alpha_mask(alpha_val, 64),
        }
    }

    #[inline(always)]
    fn ensure_capacity(&mut self, num_blocks: usize, total_queries: usize) {
        if self.vp.len() < num_blocks {
            let all_ones = B::splat_all_ones();
            let zero = B::splat_zero();
            self.vp.resize(num_blocks, all_ones);
            self.vn.resize(num_blocks, zero);
            self.score.resize(num_blocks, zero);
            self.failure_masks.resize(num_blocks, all_ones);
        }
        if self.results.capacity() < total_queries {
            self.results.reserve(total_queries - self.results.len());
        }
    }

    #[inline(always)]
    fn reset_state(&mut self, t_queries: &TQueries<B>, alpha_pattern: u64) {
        let mask_len = t_queries.query_length;
        let length_mask = if mask_len >= 64 {
            !0
        } else {
            (1u64 << mask_len) - 1
        };
        let masked_alpha: u64 = alpha_pattern & length_mask;

        let initial_score = B::splat_from_usize(masked_alpha.count_ones() as usize);
        let alpha_simd = B::splat_scalar(B::mask_word_to_scalar(alpha_pattern));
        let zero = B::splat_zero();
        let all_ones = B::splat_all_ones();
        let num_blocks = t_queries.n_simd_blocks;
        unsafe {
            for i in 0..num_blocks {
                *self.failure_masks.get_unchecked_mut(i) = all_ones; // all start as fails, if match <=k, it will be false
                *self.vp.get_unchecked_mut(i) = alpha_simd;
                *self.vn.get_unchecked_mut(i) = zero;
                *self.score.get_unchecked_mut(i) = initial_score;
            }
        }
    }

    #[inline(always)]
    fn myers_step(
        vp: B::Simd,
        vn: B::Simd,
        score: B::Simd,
        eq: B::Simd,
        last_bit_shift: u32,
        last_bit_mask: B::Simd,
    ) -> (B::Simd, B::Simd, B::Simd) {
        let all_ones = B::splat_all_ones();

        let eq_and_pv = eq & vp;
        let xh = ((eq_and_pv + vp) ^ vp) | eq;
        let mh = vp & xh;
        let ph = vn | (all_ones ^ (xh | vp));

        let ph_shifted = ph << 1;
        let mh_shifted = mh << 1;

        let xv = eq | vn;
        let vp_out = mh_shifted | (all_ones ^ (xv | ph_shifted));
        let vn_out = ph_shifted & xv;

        let ph_bit = (ph & last_bit_mask) >> last_bit_shift;
        let mh_bit = (mh & last_bit_mask) >> last_bit_shift;

        let score_out = (score + ph_bit) - mh_bit;

        (vp_out, vn_out, score_out)
    }

    #[inline(never)] // to run just search_asm have to use inline(never)
    pub fn scan(&mut self, t_queries: &TQueries<B>, text: &[u8], k: u32) -> &[bool] {
        let stride = t_queries.n_simd_blocks;
        let num_blocks = t_queries.n_simd_blocks;
        let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.query_length));
        let alpha_pattern = self.alpha_pattern & length_mask;

        self.ensure_capacity(num_blocks, t_queries.n_queries);
        self.reset_state(t_queries, alpha_pattern);

        let k_simd = B::splat_from_usize(k as usize);
        let last_bit_shift = (t_queries.query_length - 1) as u32;
        let last_bit_mask = B::splat_one() << last_bit_shift;
        // todo: maybe just not unsafe this at all?
        let peqs_ptr: *const <B as SimdBackend>::Simd = t_queries.peqs.as_ptr();
        let vp_ptr = self.vp.as_mut_ptr();
        let vn_ptr = self.vn.as_mut_ptr();
        let score_ptr = self.score.as_mut_ptr();
        let fail_ptr = self.failure_masks.as_mut_ptr();
        for &c in text {
            let encoded = crate::iupac::get_encoded(c) as usize;
            let base_offset = encoded * stride;
            for block_i in 0..num_blocks {
                unsafe {
                    let eq = *peqs_ptr.add(base_offset + block_i);
                    let vp_in = *vp_ptr.add(block_i);
                    let vn_in = *vn_ptr.add(block_i);
                    let score_in = *score_ptr.add(block_i);
                    let (vp_out, vn_out, score_out) =
                        Self::myers_step(vp_in, vn_in, score_in, eq, last_bit_shift, last_bit_mask);
                    *vp_ptr.add(block_i) = vp_out;
                    *vn_ptr.add(block_i) = vn_out;
                    *score_ptr.add(block_i) = score_out;
                    let gt_mask = B::simd_gt(score_out, k_simd);
                    *fail_ptr.add(block_i) &= gt_mask;
                }
            }
        }

        if self.alpha_pattern != !0 {
            self.process_overhangs(t_queries, k_simd, alpha_pattern, num_blocks);
        }

        self.extract_bools(t_queries)
    }

    #[inline(always)]
    fn process_overhangs(
        &mut self,
        t_queries: &TQueries<B>,
        k_simd: B::Simd,
        alpha: u64,
        num_blocks: usize,
    ) {
        // We can just directly use the alpha mask as `eq``, as this already reflects alternating "matches" and "mismatches"
        let eq = B::splat_scalar(B::mask_word_to_scalar(alpha));
        let last_bit_shift = (t_queries.query_length - 1) as u32;
        let last_bit_mask = B::splat_one() << last_bit_shift;
        let steps_needed = t_queries.query_length;

        let vp_ptr = self.vp.as_mut_ptr();
        let vn_ptr = self.vn.as_mut_ptr();
        let score_ptr = self.score.as_mut_ptr();
        let fail_ptr = self.failure_masks.as_mut_ptr();

        for _ in 0..steps_needed {
            for block_i in 0..num_blocks {
                unsafe {
                    let vp_in = *vp_ptr.add(block_i);
                    let vn_in = *vn_ptr.add(block_i);
                    let score_in = *score_ptr.add(block_i);

                    let (vp_out, vn_out, score_out) =
                        Self::myers_step(vp_in, vn_in, score_in, eq, last_bit_shift, last_bit_mask);

                    *vp_ptr.add(block_i) = vp_out;
                    *vn_ptr.add(block_i) = vn_out;
                    *score_ptr.add(block_i) = score_out;

                    *fail_ptr.add(block_i) &= B::simd_gt(score_out, k_simd);
                }
            }
        }
    }

    #[inline(always)]
    fn extract_bools(&mut self, t_queries: &TQueries<B>) -> &[bool] {
        let n_queries = t_queries.n_queries;

        self.results.clear();
        self.results.resize(n_queries, false);
        let zero_scalar = B::scalar_from_i64(0);
        let mut remaining = n_queries;
        let mut results_ptr = self.results.as_mut_ptr();

        for &fail_simd in self.failure_masks.iter() {
            if remaining == 0 {
                break;
            }

            let fail_array = B::to_array(fail_simd);
            let fail_slice = fail_array.as_ref();
            let lanes_to_process = B::LANES.min(remaining);

            for lane_i in 0..lanes_to_process {
                unsafe {
                    // if fail_mask is 0 -> matched <=k, which is what we want to report
                    let val = *fail_slice.get_unchecked(lane_i);
                    *results_ptr = val == zero_scalar;
                    results_ptr = results_ptr.add(1);
                }
            }
            remaining -= lanes_to_process;
        }
        &self.results
    }

    #[inline(always)]
    fn generate_alpha_mask(alpha: f32, length: usize) -> u64 {
        let mut mask = 0u64;
        let limit = length.min(64);
        for i in 0..limit {
            let val = ((i + 1) as f32 * alpha).floor() as u64 - (i as f32 * alpha).floor() as u64;
            if val >= 1 {
                mask |= 1 << i;
            }
        }
        mask
    }
}
