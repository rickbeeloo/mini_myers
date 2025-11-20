use crate::backend::SimdBackend;
use crate::tqueries::TQueries;

// To keep vp, vn, and score close in cache
#[derive(Clone, Copy)]
struct BlockState<B: SimdBackend> {
    pub vp: B::Simd,
    pub vn: B::Simd,
    pub score: B::Simd,
}

pub struct Searcher<B: SimdBackend> {
    blocks: Vec<BlockState<B>>,
    match_masks: Vec<B::Simd>,
    results: Vec<bool>,
}

impl<B: SimdBackend> Default for Searcher<B> {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            match_masks: Vec::new(),
            results: Vec::new(),
        }
    }
}

impl<B: SimdBackend> Searcher<B> {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_capacity(&mut self, num_blocks: usize, total_queries: usize) {
        if self.blocks.len() < num_blocks {
            self.blocks.resize(
                num_blocks,
                BlockState {
                    vp: B::splat_zero(),
                    vn: B::splat_zero(),
                    score: B::splat_zero(),
                },
            );
            self.match_masks.resize(num_blocks, B::splat_zero());
        }

        // Ensure results buffer has enough space to avoid re-allocs during extraction
        if self.results.capacity() < total_queries {
            self.results.reserve(total_queries - self.results.len());
        }
    }

    fn reset_state(&mut self, t_queries: &TQueries<B>, alpha_pattern: u64) {
        let mask_len = t_queries.query_length;

        // Logic unchanged, just scalar calculation
        let length_mask = if mask_len >= B::LIMB_BITS {
            !0
        } else {
            (1u64 << mask_len) - 1
        };
        let masked_alpha: u64 = alpha_pattern & length_mask;
        let initial_score_val = masked_alpha.count_ones();

        // Pre-compute SIMD vectors to broadcast
        let initial_score = B::splat_from_usize(initial_score_val as usize);
        let alpha_scalar = B::mask_word_to_scalar(alpha_pattern);
        let alpha_simd = B::splat_scalar(alpha_scalar);
        let zero = B::splat_zero();

        let num_blocks = t_queries.vectors[0].len();

        for (block, mask) in self
            .blocks
            .iter_mut()
            .zip(self.match_masks.iter_mut())
            .take(num_blocks)
        {
            *mask = zero;
            block.vp = alpha_simd;
            block.vn = zero;
            block.score = initial_score;
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
        let xv = eq | *mv;
        let eq_and_pv = eq & *pv;
        let sum = eq_and_pv + *pv;
        let xh = (sum ^ *pv) | eq;
        let ph = *mv | (all_ones ^ (xh | *pv));
        let mh = *pv & xh;
        let ph_shifted = ph << 1;
        let mh_shifted = mh << 1;
        *pv = mh_shifted | (all_ones ^ (xv | ph_shifted));
        *mv = ph_shifted & xv;

        // Score update
        let ph_bit = (ph & last_bit_mask) >> last_bit_shift;
        let mh_bit = (mh & last_bit_mask) >> last_bit_shift;
        *score = (*score + ph_bit) - mh_bit;
    }

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

    pub fn scan(
        &mut self,
        t_queries: &TQueries<B>,
        text: &[u8],
        k: u32,
        alpha: Option<f32>,
    ) -> &[bool] {
        let num_blocks = t_queries.vectors[0].len();
        let alpha_val = alpha.unwrap_or(1.0);
        let alpha_pattern = Self::generate_alpha_mask(alpha_val, t_queries.query_length);
        let all_ones = B::splat_all_ones();

        self.ensure_capacity(num_blocks, t_queries.n_queries);
        self.reset_state(t_queries, alpha_pattern);

        let k_simd = B::splat_from_usize(k as usize);
        let last_bit_shift = (t_queries.query_length - 1) as u32;
        let last_bit_mask = B::splat_one() << last_bit_shift;

        // Main Loop
        for &c in text {
            let encoded = crate::iupac::get_encoded(c);
            // Accessing the pre-computed pattern equality bitmasks
            let peq_block_list = &t_queries.peqs[encoded as usize];

            for block_i in 0..num_blocks {
                let eq = peq_block_list[block_i];

                let block = &mut self.blocks[block_i];

                Self::myers_col(
                    &mut block.vp,
                    &mut block.vn,
                    &mut block.score,
                    eq,
                    last_bit_shift,
                    last_bit_mask,
                );

                // Update mask
                let gt_mask = B::simd_gt(block.score, k_simd);
                let le_mask = gt_mask ^ all_ones;
                self.match_masks[block_i] = self.match_masks[block_i] | le_mask;
            }
        }

        if alpha.is_some() {
            self.process_overhangs(t_queries, k_simd, alpha_pattern, num_blocks);
        }

        // Extract to internal buffer and return slice
        self.extract_bools(t_queries)
    }

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

        for _ in 0..steps_needed {
            for block_i in 0..num_blocks {
                let block = &mut self.blocks[block_i];

                Self::myers_col(
                    &mut block.vp,
                    &mut block.vn,
                    &mut block.score,
                    eq,
                    last_bit_shift,
                    last_bit_mask,
                );

                let gt_mask = B::simd_gt(block.score, k_simd);
                let le_mask = gt_mask ^ all_ones;
                self.match_masks[block_i] = self.match_masks[block_i] | le_mask;
            }
        }
    }

    fn extract_bools(&mut self, t_queries: &TQueries<B>) -> &[bool] {
        let total_queries = t_queries.n_queries;
        let zero = B::scalar_from_i64(0);

        // Clear retains capacity, so no de-alloc
        self.results.clear();

        for (block_idx, &mask_simd) in self.match_masks.iter().enumerate() {
            // Skip padding
            if block_idx * B::LANES >= total_queries {
                break;
            }

            let lanes_array = B::to_array(mask_simd);
            let lanes_slice = lanes_array.as_ref();

            for lane_i in 0..B::LANES {
                if self.results.len() == total_queries {
                    break;
                }
                // Simple push is very fast when capacity is reserved
                self.results.push(lanes_slice[lane_i] != zero);
            }
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
        let mut searcher = Searcher::<U32>::new();
        let text = b"AAAAAAAGGCCAAAAAAAAAAA";
        let matches = searcher.scan(&transposed, text, 1, None);
        assert!(matches[0]);
        assert!(!matches[1]);
    }

    #[test]
    fn test_clean_search() {
        let queries = vec![b"GGGGCC".to_vec(), b"TTTTAA".to_vec()];
        let transposed = TQueries::<U32>::new(&queries, false);
        // 2. Setup
        let mut searcher = Searcher::<U32>::new();

        // 3. Stream Text
        let text = b"AAAAAGGGG";
        let matches = searcher.scan(&transposed, text, 1, None);
        println!("matches: {:?}", matches);
        let matches = searcher.scan(&transposed, text, 1, Some(0.5));
        println!("matches with alpha 0.5: {:?}", matches);
    }
}
