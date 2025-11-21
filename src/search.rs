use crate::backend::SimdBackend;
use crate::tqueries::TQueries;
use wide::CmpEq;

pub struct Searcher<B: SimdBackend> {
    vp: Vec<B::Simd>,
    vn: Vec<B::Simd>,
    score: Vec<B::Simd>,
    match_masks: Vec<B::Simd>,
    results: Vec<bool>,
    alpha_pattern: u64,
}

impl<B: SimdBackend> Default for Searcher<B> {
    fn default() -> Self {
        Self {
            vp: Vec::new(),
            vn: Vec::new(),
            score: Vec::new(),
            match_masks: Vec::new(),
            results: Vec::new(),
            alpha_pattern: Self::generate_alpha_mask(1.0, 64),
        }
    }
}

impl<B: SimdBackend> Searcher<B> {
    pub fn new(alpha: Option<f32>) -> Self {
        let alpha_val = alpha.unwrap_or(1.0);
        Self {
            vp: Vec::new(),
            vn: Vec::new(),
            score: Vec::new(),
            match_masks: Vec::new(),
            results: Vec::new(),
            alpha_pattern: Self::generate_alpha_mask(alpha_val, 64),
        }
    }

    #[inline(always)]
    fn ensure_capacity(&mut self, num_blocks: usize, total_queries: usize) {
        if self.vp.len() < num_blocks {
            let zero = B::splat_zero();
            let all_ones = B::splat_all_ones();
            self.vp.resize(num_blocks, all_ones);
            self.vn.resize(num_blocks, zero);
            self.score.resize(num_blocks, zero);
            self.match_masks.resize(num_blocks, zero);
        }
        if self.results.capacity() < total_queries {
            self.results.reserve(total_queries - self.results.len());
        }
    }

    #[inline(always)]
    fn reset_state(&mut self, t_queries: &TQueries<B>, alpha_pattern: u64) {
        let mask_len = t_queries.query_length;

        let length_mask = if mask_len >= B::LIMB_BITS {
            !0
        } else {
            (1u64 << mask_len) - 1
        };
        let masked_alpha: u64 = alpha_pattern & length_mask;
        let initial_score_val = masked_alpha.count_ones();

        let initial_score = B::splat_from_usize(initial_score_val as usize);
        let alpha_scalar = B::mask_word_to_scalar(alpha_pattern);
        let alpha_simd = B::splat_scalar(alpha_scalar);
        let zero = B::splat_zero();

        let num_blocks = t_queries.peqs[0].len();

        for i in 0..num_blocks {
            self.match_masks[i] = zero;
            self.vp[i] = alpha_simd;
            self.vn[i] = zero;
            self.score[i] = initial_score;
        }
    }

    #[inline(always)]
    fn myers_col(
        pv: &mut B::Simd,
        mv: &mut B::Simd,
        score: &mut B::Simd,
        eq: B::Simd,
        last_bit_shift: u32,
        last_bit_mask: B::Simd,
    ) {
        let all_ones = B::splat_all_ones();
        let eq_and_pv = eq & *pv;
        let xh = ((eq_and_pv + *pv) ^ *pv) | eq;
        let mh = *pv & xh;
        let ph = *mv | (all_ones ^ (xh | *pv));

        let ph_shifted = ph << 1;
        let mh_shifted = mh << 1;

        let xv = eq | *mv;
        *pv = mh_shifted | (all_ones ^ (xv | ph_shifted));
        *mv = ph_shifted & xv;

        // Score update
        let ph_bit = (ph & last_bit_mask) >> last_bit_shift;
        let mh_bit = (mh & last_bit_mask) >> last_bit_shift;
        *score = (*score + ph_bit) - mh_bit;
    }

    #[inline(always)]
    fn generate_alpha_mask(alpha: f32, length: usize) -> u64 {
        let mut mask = 0u64;
        let limit = length.min(64);
        // Same as sassy
        for i in 0..limit {
            let val = ((i + 1) as f32 * alpha).floor() as u64 - (i as f32 * alpha).floor() as u64;
            if val >= 1 {
                mask |= 1 << i;
            }
        }
        mask
    }

    #[inline(never)]
    pub fn scan(&mut self, t_queries: &TQueries<B>, text: &[u8], k: u32) -> &[bool] {
        let num_blocks = t_queries.peqs[0].len();

        // Mask the pre-computed alpha pattern to the actual query length
        let length_mask = if t_queries.query_length >= 64 {
            !0
        } else {
            (1u64 << t_queries.query_length) - 1
        };
        let alpha_pattern = self.alpha_pattern & length_mask;

        self.ensure_capacity(num_blocks, t_queries.n_queries);
        self.reset_state(t_queries, alpha_pattern);

        let k_simd = B::splat_from_usize(k as usize);
        let last_bit_shift = (t_queries.query_length - 1) as u32;
        let last_bit_mask = B::splat_one() << last_bit_shift;
        let all_ones = B::splat_all_ones();

        if num_blocks == 1 {
            for &c in text {
                let encoded = crate::iupac::get_encoded(c);
                let peq_block_list = unsafe { t_queries.peqs.get_unchecked(encoded as usize) };
                let eq = peq_block_list[0];

                Self::myers_col(
                    &mut self.vp[0],
                    &mut self.vn[0],
                    &mut self.score[0],
                    eq,
                    last_bit_shift,
                    last_bit_mask,
                );

                let gt_mask = B::simd_gt(self.score[0], k_simd);
                self.match_masks[0] |= gt_mask ^ all_ones;
            }
        } else {
            for &c in text {
                let encoded = crate::iupac::get_encoded(c);
                let peq_block_list = unsafe { t_queries.peqs.get_unchecked(encoded as usize) };

                let vp_slice = &mut self.vp[..num_blocks];
                let vn_slice = &mut self.vn[..num_blocks];
                let score_slice = &mut self.score[..num_blocks];
                let mask_slice = &mut self.match_masks[..num_blocks];

                for block_i in 0..num_blocks {
                    let eq = peq_block_list[block_i];

                    Self::myers_col(
                        &mut vp_slice[block_i],
                        &mut vn_slice[block_i],
                        &mut score_slice[block_i],
                        eq,
                        last_bit_shift,
                        last_bit_mask,
                    );

                    let gt_mask = B::simd_gt(score_slice[block_i], k_simd);
                    mask_slice[block_i] |= gt_mask ^ all_ones;
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
        let all_ones = B::splat_all_ones();
        let alpha_scalar = B::mask_word_to_scalar(alpha);
        let eq = B::splat_scalar(alpha_scalar);
        let last_bit_shift = (t_queries.query_length - 1) as u32;
        let last_bit_mask = B::splat_one() << last_bit_shift;
        let steps_needed = t_queries.query_length;

        if num_blocks == 1 {
            for _ in 0..steps_needed {
                Self::myers_col(
                    &mut self.vp[0],
                    &mut self.vn[0],
                    &mut self.score[0],
                    eq,
                    last_bit_shift,
                    last_bit_mask,
                );

                let gt_mask = B::simd_gt(self.score[0], k_simd);
                self.match_masks[0] |= gt_mask ^ all_ones;
            }
        } else {
            for _ in 0..steps_needed {
                let vp_slice = &mut self.vp[..num_blocks];
                let vn_slice = &mut self.vn[..num_blocks];
                let score_slice = &mut self.score[..num_blocks];
                let mask_slice = &mut self.match_masks[..num_blocks];

                for block_i in 0..num_blocks {
                    Self::myers_col(
                        &mut vp_slice[block_i],
                        &mut vn_slice[block_i],
                        &mut score_slice[block_i],
                        eq,
                        last_bit_shift,
                        last_bit_mask,
                    );

                    let gt_mask = B::simd_gt(score_slice[block_i], k_simd);
                    mask_slice[block_i] |= gt_mask ^ all_ones;
                }
            }
        }
    }

    #[inline(always)]
    fn extract_bools(&mut self, t_queries: &TQueries<B>) -> &[bool] {
        self.results.clear();

        self.results.reserve_exact(t_queries.n_queries);

        let zero = B::splat_zero();
        let zero_scalar = B::scalar_from_i64(0);

        let mut remaining = t_queries.n_queries;

        for &mask_simd in self.match_masks.iter() {
            if remaining == 0 {
                break;
            }

            let ne_mask = mask_simd.simd_eq(zero);
            let mask_array = B::to_array(ne_mask);
            let mask_slice = mask_array.as_ref();

            let lanes_to_process = B::LANES.min(remaining);

            for lane_i in 0..lanes_to_process {
                self.results.push(mask_slice[lane_i] == zero_scalar);
            }

            remaining -= lanes_to_process;
        }

        &self.results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::U32;

    #[test]
    fn test_simple_match() {
        let queries = vec![b"GGCC".to_vec(), b"TTAA".to_vec()];
        let transposed = TQueries::<U32>::new(&queries, false);
        let mut searcher = Searcher::<U32>::new(None);
        let text = b"AAAAAAAGGCCAAAAAAAAAAA";
        let matches = searcher.scan(&transposed, text, 1);
        assert!(matches[0]);
        assert!(!matches[1]);
    }

    #[test]
    fn test_simple_rc_match() {
        let queries = vec![b"GGCC".to_vec(), b"TTAA".to_vec()];
        let transposed = TQueries::<U32>::new(&queries, false);
        let mut searcher = Searcher::<U32>::new(None);
        let text = b"TTTTTTTTTTTGGCCTTTTTTT";
        let matches = searcher.scan(&transposed, text, 1);
        assert!(matches[0]);
        assert!(!matches[1]);
    }

    #[test]
    fn test_overhang_prefix() {
        let queries = vec![b"TTTTAA".to_vec()];
        let transposed = TQueries::<U32>::new(&queries, false);

        let mut searcher = Searcher::<U32>::new(None);
        let text = b"AAAAAGGGG";
        let matches = searcher.scan(&transposed, text, 2);
        assert!(!matches[0]);

        let mut searcher = Searcher::<U32>::new(Some(0.5));
        let matches = searcher.scan(&transposed, text, 2);
        assert!(matches[0]);
    }

    #[test]
    fn test_overhang_suffix() {
        let queries = vec![b"GGGGCC".to_vec()];
        let transposed = TQueries::<U32>::new(&queries, false);

        let mut searcher = Searcher::<U32>::new(None);
        let text = b"AAAAAGGGG";
        let matches = searcher.scan(&transposed, text, 1);
        assert!(!matches[0]);

        let mut searcher = Searcher::<U32>::new(Some(0.5));
        let matches = searcher.scan(&transposed, text, 1);
        assert!(matches[0]);
    }

    #[test]
    fn test_overhang_both_sides() {
        let queries = vec![b"GGGGCC".to_vec(), b"TTTTAA".to_vec()];
        let transposed = TQueries::<U32>::new(&queries, false);
        let mut searcher = Searcher::<U32>::new(Some(0.5));
        let text = b"AAAAAGGGG";
        let matches = searcher.scan(&transposed, text, 1);
        println!("matches with alpha 0.5: {:?}", matches);
    }

    fn random_dna_seq(l: usize) -> Vec<u8> {
        use rand::thread_rng;
        use rand::Rng;
        const DNA: &[u8; 4] = b"ACGT";
        let mut rng = thread_rng();
        let mut dna = Vec::with_capacity(l);
        for _ in 0..l {
            let idx = rng.gen_range(0..4);
            dna.push(DNA[idx]);
        }
        dna
    }

    #[test]
    fn test_more_than_single_block_queries() {
        let n_queries = 32 * 2 + 1;
        let query_len = 32;
        let mut queries = Vec::with_capacity(n_queries);
        for _ in 0..n_queries {
            queries.push(random_dna_seq(query_len));
        }
        let transposed = TQueries::<U32>::new(&queries, false);
        let mut searcher = Searcher::<U32>::new(None);
        let text = random_dna_seq(1_000);
        let matches = searcher.scan(&transposed, &text, 1);
        assert_eq!(matches.len(), n_queries);
    }
}
