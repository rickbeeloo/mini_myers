use crate::backend::SimdBackend;
use crate::iupac::get_encoded;
use crate::tqueries::TQueries;

#[derive(Clone, Copy, Default)]
struct BlockState<S: Copy> {
    vp: S,
    vn: S,
    score: S,
    failure_mask: S,
}

pub struct Searcher<B: SimdBackend> {
    blocks: Vec<BlockState<B::Simd>>,
    results: Vec<bool>,
    positions: Vec<Vec<usize>>,
    alpha_pattern: u64,
    alpha: f32,
}

impl<B: SimdBackend> Default for Searcher<B> {
    fn default() -> Self {
        Self::new(None)
    }
}

impl<B: SimdBackend> Searcher<B> {
    pub fn new(alpha: Option<f32>) -> Self {
        let alpha_val = alpha.unwrap_or(1.0);
        Self {
            blocks: Vec::new(),
            results: Vec::new(),
            positions: Vec::new(),
            alpha_pattern: Self::generate_alpha_mask(alpha_val, 64),
            alpha: alpha_val,
        }
    }

    #[inline(always)]
    fn ensure_capacity(&mut self, num_blocks: usize, total_queries: usize) {
        if self.blocks.len() < num_blocks {
            let all_ones = B::splat_all_ones();
            let zero = B::splat_zero();
            self.blocks.resize(
                num_blocks,
                BlockState {
                    vp: all_ones,
                    vn: zero,
                    score: zero,
                    failure_mask: all_ones,
                },
            );
        }
        if self.results.capacity() < total_queries {
            self.results.reserve(total_queries - self.results.len());
        }
        if self.positions.len() < total_queries {
            self.positions.resize(total_queries, Vec::new());
        }
    }

    #[inline(always)]
    fn reset_state(&mut self, t_queries: &TQueries<B>, alpha_pattern: u64) {
        let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.query_length));
        let masked_alpha: u64 = alpha_pattern & length_mask;
        let initial_score = B::splat_scalar(B::scalar_from_i64(masked_alpha.count_ones() as i64));
        // println!(
        //     "Initial edits: {:?}",
        //     B::to_array(initial_score).as_ref()[0]
        // );
        let alpha_simd = B::splat_scalar(B::mask_word_to_scalar(alpha_pattern));
        let zero = B::splat_zero();
        let all_ones = B::splat_all_ones();
        let num_blocks = t_queries.n_simd_blocks;

        for i in 0..num_blocks {
            unsafe {
                let block = self.blocks.get_unchecked_mut(i);
                block.failure_mask = all_ones;
                block.vp = alpha_simd;
                block.vn = zero;
                block.score = initial_score;
            }
        }
    }

    //#[inline(always)]
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

        // println!("ph_bit: {:?}", B::to_array(ph_bit).as_ref()[0]);
        // println!("mh_bit: {:?}", B::to_array(mh_bit).as_ref()[0]);

        let score_out = (score + ph_bit) - mh_bit;

        (vp_out, vn_out, score_out)
    }

    #[inline(always)]
    pub fn scan(&mut self, t_queries: &TQueries<B>, text: &[u8], k: u32) -> &[bool] {
        let num_blocks = t_queries.n_simd_blocks;

        let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.query_length));
        let alpha_pattern = self.alpha_pattern & length_mask;

        self.ensure_capacity(num_blocks, t_queries.n_queries);
        self.reset_state(t_queries, alpha_pattern);

        let k_simd = B::splat_scalar(B::scalar_from_i64(k as i64));
        let last_bit_shift = (t_queries.query_length - 1) as u32;
        let last_bit_mask = B::splat_one() << last_bit_shift;

        let peqs_ptr: *const <B as SimdBackend>::Simd = t_queries.peqs.as_ptr();
        let blocks_ptr = self.blocks.as_mut_ptr();

        for &c in text {
            let encoded = get_encoded(c) as usize;
            let peq_block_start_index = encoded * num_blocks;

            for block_i in 0..num_blocks {
                unsafe {
                    let eq = *peqs_ptr.add(peq_block_start_index + block_i);
                    let block = &mut *blocks_ptr.add(block_i);
                    let (vp_out, vn_out, score_out) = Self::myers_step(
                        block.vp,
                        block.vn,
                        block.score,
                        eq,
                        last_bit_shift,
                        last_bit_mask,
                    );
                    block.vp = vp_out;
                    block.vn = vn_out;
                    block.score = score_out;
                    let gt_mask = B::simd_gt(score_out, k_simd);
                    block.failure_mask &= gt_mask;
                }
            }
        }

        if self.alpha_pattern != !0 {
            self.process_overhangs(t_queries, k_simd, num_blocks);
        }

        self.extract_bools(t_queries)
    }

    pub fn multi_text_scan(&mut self, t_queries: &TQueries<B>, texts: &[&[u8]], k: u32) -> &[bool] {
        assert_eq!(texts.len(), t_queries.n_queries);

        let num_blocks = t_queries.n_simd_blocks;
        self.ensure_capacity(num_blocks, t_queries.n_queries);

        let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.query_length));
        self.reset_state(t_queries, self.alpha_pattern & length_mask);

        let k_simd = B::splat_scalar(B::scalar_from_i64(k as i64));
        let last_bit_shift = (t_queries.query_length - 1) as u32;
        let last_bit_mask = B::splat_one() << last_bit_shift;
        let all_ones = B::splat_all_ones();
        let zero_scalar = B::scalar_from_i64(0);
        let one_mask: <B as SimdBackend>::Scalar = B::mask_word_to_scalar(!0);

        let blocks_ptr = self.blocks.as_mut_ptr();
        let max_len = texts.iter().map(|t| t.len()).max().unwrap_or(0) + 1;

        for i in 0..max_len {
            for block_i in 0..num_blocks {
                unsafe {
                    let block = &mut *blocks_ptr.add(block_i);

                    let mut eq_arr = B::LaneArray::default();
                    let mut keep_mask_arr = B::LaneArray::default();

                    let eq_slice = eq_arr.as_mut();
                    let keep_slice = keep_mask_arr.as_mut();
                    let base_idx = block_i * B::LANES;

                    for lane in 0..B::LANES {
                        let q_idx = base_idx + lane;
                        if q_idx < texts.len() {
                            if i < texts[q_idx].len() {
                                let enc = get_encoded(texts[q_idx][i]) as usize;
                                eq_slice[lane] =
                                    B::mask_word_to_scalar(t_queries.peq_masks[enc][q_idx]);
                                keep_slice[lane] = one_mask;
                            } else {
                                // We just zero eq as "no matches" if we go beyond the text lenght
                                // kinda like 'X'
                                eq_slice[lane] = zero_scalar;
                                keep_slice[lane] = zero_scalar;
                            }
                        }
                    }

                    let eq = B::from_array(eq_arr);
                    let keep_mask = B::from_array(keep_mask_arr);
                    let freeze_mask = all_ones ^ keep_mask; // Inverse of keep

                    let (vp_new, vn_new, score_new) = Self::myers_step(
                        block.vp,
                        block.vn,
                        block.score,
                        eq,
                        last_bit_shift,
                        last_bit_mask,
                    );
                    // Blending, if we are 'in text' we take the new values, if
                    // we are 'in overhang' we take the old value (freezing the end state)
                    block.vp = (vp_new & keep_mask) | (block.vp & freeze_mask);
                    block.vn = (vn_new & keep_mask) | (block.vn & freeze_mask);
                    block.score = (score_new & keep_mask) | (block.score & freeze_mask);
                    let current_fail = block.failure_mask & B::simd_gt(score_new, k_simd);
                    block.failure_mask =
                        (current_fail & keep_mask) | (block.failure_mask & freeze_mask);
                }
            }
        }

        if self.alpha_pattern != !0 {
            self.process_overhangs(t_queries, k_simd, num_blocks);
        }

        self.extract_bools(t_queries)
    }

    pub fn search(&mut self, t_queries: &TQueries<B>, text: &[u8], k: u32) -> &[Vec<usize>] {
        let num_blocks = t_queries.n_simd_blocks;

        let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.query_length));
        let alpha_pattern = self.alpha_pattern & length_mask;

        self.ensure_capacity(num_blocks, t_queries.n_queries);
        self.reset_state(t_queries, alpha_pattern);

        // Clear previous positions
        for pos_vec in self.positions.iter_mut().take(t_queries.n_queries) {
            pos_vec.clear();
        }

        let k_simd = B::splat_scalar(B::scalar_from_i64(k as i64));
        let last_bit_shift = (t_queries.query_length - 1) as u32;
        let last_bit_mask = B::splat_one() << last_bit_shift;

        let peqs_ptr: *const <B as SimdBackend>::Simd = t_queries.peqs.as_ptr();
        let blocks_ptr = self.blocks.as_mut_ptr();
        let zero_scalar = B::scalar_from_i64(0);

        for (idx, &c) in text.iter().enumerate() {
            // println!("Text char: {} at idx {}", c as char, idx);
            let encoded = get_encoded(c) as usize;
            let peq_block_start_index = encoded * num_blocks;

            for block_i in 0..num_blocks {
                unsafe {
                    let eq = *peqs_ptr.add(peq_block_start_index + block_i);
                    let block = &mut *blocks_ptr.add(block_i);
                    let (vp_out, vn_out, score_out) = Self::myers_step(
                        block.vp,
                        block.vn,
                        block.score,
                        eq,
                        last_bit_shift,
                        last_bit_mask,
                    );
                    block.vp = vp_out;
                    block.vn = vn_out;
                    block.score = score_out;
                    //    println!("edits: {:?}", B::to_array(score_out).as_ref()[0]);
                    let gt_mask = B::simd_gt(score_out, k_simd);
                    //  block.failure_mask &= gt_mask;

                    if gt_mask != B::splat_all_ones() {
                        let mask_arr = B::to_array(gt_mask);
                        let mask_slice = mask_arr.as_ref();

                        for (lane_idx, &val) in mask_slice.iter().enumerate() {
                            if val == zero_scalar {
                                let query_idx = block_i * B::LANES + lane_idx;
                                if query_idx < t_queries.n_queries {
                                    self.positions[query_idx].push(idx);
                                }
                            }
                        }
                    }
                }
            }
        }

        if self.alpha_pattern != !0 {
            let eq = B::splat_zero(); // No matches at all
            let steps_needed = t_queries.query_length;
            let blocks_ptr = self.blocks.as_mut_ptr();
            let mut current_text_pos = text.len();
            for i in 0..steps_needed {
                for block_i in 0..num_blocks {
                    unsafe {
                        let block = &mut *blocks_ptr.add(block_i);
                        let (vp_out, vn_out, score_out) = Self::myers_step(
                            block.vp,
                            block.vn,
                            block.score,
                            eq,
                            last_bit_shift,
                            last_bit_mask,
                        );
                        block.vp = vp_out;
                        block.vn = vn_out;
                        block.score = score_out;
                        for (lane_idx, score) in B::to_array(score_out).as_ref().iter().enumerate()
                        {
                            let score = B::scalar_to_u32(*score);
                            let new_score = score - (self.alpha * (i + 1) as f32).floor() as u32;
                            if new_score <= k {
                                let query_idx = block_i * B::LANES + lane_idx;
                                if query_idx < t_queries.n_queries {
                                    self.positions[query_idx].push(current_text_pos);
                                }
                            }
                        }
                    }
                }
                current_text_pos += 1;
            }
        }

        &self.positions
    }

    #[inline(always)]
    fn process_overhangs(&mut self, t_queries: &TQueries<B>, k_simd: B::Simd, num_blocks: usize) {
        let eq = B::splat_zero(); // All zeros, no matches at all
        let last_bit_shift = (t_queries.query_length - 1) as u32;
        let last_bit_mask = B::splat_one() << last_bit_shift;
        let steps_needed = t_queries.query_length;

        let blocks_ptr = self.blocks.as_mut_ptr();

        for i in 0..steps_needed {
            for block_i in 0..num_blocks {
                unsafe {
                    let block = &mut *blocks_ptr.add(block_i);

                    let (vp_out, vn_out, score_out) = Self::myers_step(
                        block.vp,
                        block.vn,
                        block.score,
                        eq,
                        last_bit_shift,
                        last_bit_mask,
                    );

                    block.vp = vp_out;
                    block.vn = vn_out;

                    // Adjust the score for this overhang step to account for alpha
                    let score_out_arr = B::to_array(score_out);
                    let mut adj_score_arr = score_out_arr;
                    for lane_idx in 0..B::LANES {
                        let s = B::scalar_to_u32(score_out_arr.as_ref()[lane_idx]);
                        let new_score =
                            s.saturating_sub((self.alpha * (i + 1) as f32).floor() as u32);
                        adj_score_arr.as_mut()[lane_idx] = B::scalar_from_i64(new_score as i64);
                    }
                    let adj_score = B::from_array(adj_score_arr);
                    // println!("Adjusted score: {:?}", B::to_array(adj_score).as_ref());
                    block.score = score_out;

                    // Use the adjusted score for failure_mask
                    let gt_mask = B::simd_gt(adj_score, k_simd);
                    block.failure_mask &= gt_mask;
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

        for block in &self.blocks {
            if remaining == 0 {
                break;
            }

            let fail_array = B::to_array(block.failure_mask);
            let fail_slice = fail_array.as_ref();
            let lanes_to_process = B::LANES.min(remaining);

            for lane_i in 0..lanes_to_process {
                unsafe {
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
    // Like in sassy
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

mod tests {

    use super::Searcher;
    use crate::backend::{U32, U64};
    use crate::tqueries::TQueries;

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

    #[test]
    fn test_multi_text_scan() {
        let queries = vec![b"GGCC".to_vec(), b"AAAA".to_vec()];
        // Second matches first text slice, but these are not "paired" and should thus be false
        let texts: Vec<&[u8]> = vec![b"AAAAAAAAAGG", b"GGGGGGGGG"];

        let transposed = TQueries::<U32>::new(&queries, false);
        let mut searcher = Searcher::<U32>::new(Some(0.5));

        let matches = searcher.multi_text_scan(&transposed, &texts, 1);
        println!("matches: {:?}", matches);
        assert!(matches[0], "First query should match first text");
        assert!(
            !matches[1],
            "Second query should NOT match second text (does first but not paired)"
        );
    }

    #[test]
    fn test_multi_text_scan_overhang() {
        let queries = vec![b"GGCC".to_vec(), b"AAAA".to_vec(), b"CCGG".to_vec()];
        // Second matches first text slice, but these are not "paired" and should thus be false

        #[rustfmt::skip]
        let texts: Vec<&[u8]> = vec![b"AAAAAAAAAAAAAAAAAAAAAAAAAAGG", 
                                     b"GGGGGGGG", 
                                     b"AAAAAAAAAACC"];

        let transposed = TQueries::<U32>::new(&queries, false);
        let mut searcher = Searcher::<U32>::new(Some(0.5));
        let matches = searcher.multi_text_scan(&transposed, &texts, 1);
        assert_eq!(matches, vec![true, false, true]);
    }

    #[test]
    fn test_search_positions() {
        let queries = vec![b"GGCC".to_vec()];
        // "GGCC" is at index 0 and 10
        let text = b"GGCCAAAAAAGGCC";

        let transposed = TQueries::<U32>::new(&queries, false);
        let mut searcher = Searcher::<U32>::new(None);

        let positions = searcher.search(&transposed, text, 0);
        assert_eq!(
            positions[0],
            vec![3, 13],
            "Should find exact matches at end indices 3 and 13"
        );

        // With 1 error
        let text_err = b"GGCCTAAAAAGCCC"; // GCCC matches GGCC with 1 sub (G->C) or del/ins
                                          // GCCC vs GGCC:
                                          // G G C C
                                          // G C C C
                                          // M M S M -> 1 error

        let positions_err = searcher.search(&transposed, text_err, 1);
        // GGCCT ends at 4 (0-indexed? length is 5. indices 0,1,2,3,4. 'T' is at 4).
        // GGCC match in GGCCT -> GGCC matches at 3.
        // GCCC match in ...GCCC -> ends at 13.
        assert!(positions_err[0].contains(&3));
        assert!(positions_err[0].contains(&13));
    }

    #[test]
    fn test_search_overhang_positions() {
        // // Test that matches in overhang are reported with correct positions (> text.len())
        // let queries = vec![b"GGCC".to_vec()];
        // let transposed: TQueries<crate::I32x8Backend> = TQueries::<U32>::new(&queries, false);
        let mut searcher: Searcher<crate::I32x8Backend> = Searcher::<U32>::new(Some(0.5));
        // Allow prefix/overhang matches
        // let text = b"AAAAAAGG";
        // let positions = searcher.search(&transposed, text, 1);
        // println!("suffix positions: {:?}", positions);
        // // Check for 9 as user requested, but also tolerate 8 and 10 which are valid for k=1
        // assert!(positions[0].contains(&9), "Should find match at index 9");

        // Lets also do prefix
        let queries = vec![b"GGGGGGCC".to_vec()];
        let transposed: TQueries<crate::I32x8Backend> = TQueries::<U32>::new(&queries, false);
        let text = b"CCAAAAAAAAAAAAA";

        println!(
            "Query: {:?}",
            String::from_utf8_lossy(queries[0].as_slice())
        );
        println!("Text: {:?}", String::from_utf8_lossy(text));
        let positions = searcher.search(&transposed, text, 3);
        println!("prefix positions: {:?}", positions); // [[0,1]]

        // Testing with sassy all minima
        use sassy::profiles::Iupac;
        use sassy::Searcher as SassySearcher;
        let mut sassy_searcher = SassySearcher::<Iupac>::new_fwd_with_overhang(0.5);
        let matches = sassy_searcher.search_all(queries[0].as_slice(), text, 3);
        for m in matches {
            println!("sassy match text end: {}", m.text_end);
        }
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

    #[test]
    fn test_u64_query() {
        use crate::backend::U64;
        let query_len = 64;
        let queries = vec![random_dna_seq(query_len)];
        let transposed = TQueries::<U64>::new(&queries, false);
        let mut searcher = Searcher::<U64>::new(None);
        let mut text = random_dna_seq(1_000);
        let matches = searcher.scan(&transposed, &text, 1);
        assert_eq!(matches.len(), 1);
        assert!(!matches[0]);
        // Insert the match at location 100
        text.splice(100..100 + query_len, queries[0].clone());
        let matches = searcher.scan(&transposed, &text, 1);
        assert!(matches[0]);
    }

    #[test]
    fn test_u64_overhang() {
        use crate::backend::U64;
        let query_len = 64;
        let queries = vec![random_dna_seq(query_len)];
        let transposed = TQueries::<U64>::new(&queries, false);
        let mut searcher = Searcher::<U64>::new(None);
        let mut text = random_dna_seq(1_000);
        // get last 32 chars of query, and prefix the text with it
        let last_32_chars = queries[0]
            .iter()
            .rev()
            .take(32)
            .rev()
            .copied()
            .collect::<Vec<_>>();
        text.splice(0..0, last_32_chars);
        let matches = searcher.scan(&transposed, &text, (query_len / 4) as u32);
        println!("matches: {:?}", matches);
        assert!(!matches[0]);
        let mut searcher = Searcher::<U64>::new(Some(0.5));
        let matches = searcher.scan(&transposed, &text, (query_len / 4) as u32);
        println!("matches: {:?}", matches);
        assert!(matches[0]);
    }

    fn apply_edits(seq: &[u8], k: u32) -> Vec<u8> {
        use rand::thread_rng;
        use rand::Rng;

        let mut rng = thread_rng();
        let mut res = seq.to_vec();
        const BASES: &[u8] = b"ACGT";

        for _ in 0..k {
            // To not go less than 2 bases
            let op = if res.len() < 2 {
                1
            } else {
                rng.gen_range(0..3)
            };

            match op {
                0 => {
                    // Substitution
                    let idx = rng.gen_range(0..res.len());
                    let old_base = res[idx];
                    let mut new_base = BASES[rng.gen_range(0..4)];
                    // Ensure we actually change the base
                    while new_base == old_base {
                        new_base = BASES[rng.gen_range(0..4)];
                    }
                    res[idx] = new_base;
                }
                1 => {
                    // Insertion
                    let idx = rng.gen_range(0..=res.len());
                    let new_base = BASES[rng.gen_range(0..4)];
                    res.insert(idx, new_base);
                }
                2 => {
                    // Deletion
                    let idx = rng.gen_range(0..res.len());
                    res.remove(idx);
                }
                _ => unreachable!(),
            }
        }
        res
    }

    #[test]
    fn test_fuzz_search_correctness() {
        use crate::backend::U32;
        use rand::thread_rng;
        use rand::Rng;

        let mut rng = thread_rng();
        let num_iterations = 1_000;
        let mut len_skipped = 0;

        for i in 0..num_iterations {
            let text_len = rng.gen_range(500..1500);
            let query_len = rng.gen_range(20..32);
            let k = rng.gen_range(1..=3);
            let mut text = random_dna_seq(text_len);

            let query_original = random_dna_seq(query_len);

            // Mutate it with at most k edits
            let mutated_query = apply_edits(&query_original, k);

            // With U32 backend we can only search for up to 32nts
            if mutated_query.len() > 32 {
                len_skipped += 1;
            }

            let insert_idx =
                rng.gen_range(0..text.len().saturating_sub(mutated_query.len()).max(1));
            text.splice(
                insert_idx..insert_idx + mutated_query.len(),
                mutated_query.clone(),
            );

            let queries = vec![query_original.clone()];
            let transposed = TQueries::<U32>::new(&queries, false);
            let mut searcher = Searcher::<U32>::new(None);

            let matches = searcher.scan(&transposed, &text, k);

            if !matches[0] {
                panic!(
                    "Fuzz test failed at iteration {}.\n\
                     Query (Original): {:?}\n\
                     Query (Mutated):  {:?}\n\
                     Text Snippet:     {:?}\n\
                     K: {}\n\
                     Inserted At: {}",
                    i,
                    std::str::from_utf8(&query_original).unwrap(),
                    std::str::from_utf8(&mutated_query).unwrap(),
                    std::str::from_utf8(
                        &text[insert_idx..(insert_idx + mutated_query.len() + 5).min(text.len())]
                    )
                    .unwrap(),
                    k,
                    insert_idx
                );
            }
        }
        println!("Passed {} fuzz iterations.", num_iterations);
        println!(
            "Skipped {} queries because they were too long.",
            len_skipped
        );
        // Lets error if we skipped more than half
        if len_skipped > num_iterations / 2 {
            panic!("Skipped more than half of the queries, that seems odd");
        }
    }

    fn fuzz_against_sassy(alpha: Option<f32>) {
        use rand::thread_rng;
        use rand::Rng;
        use sassy::profiles::Iupac;
        use sassy::Searcher as SassySearcher;

        // NOTE: rc test would be nice but sassy is q vs rc(text), whereas
        // in mini it's rc(q) vs text - thus not identical end positions

        // Sassy searcher
        let mut sassy_searcher =
            SassySearcher::<Iupac>::new_fwd_with_overhang(alpha.unwrap_or(1.0));
        let mut mini_searcher = Searcher::<U32>::new(alpha);

        let num_iterations = 10_000;
        let k = 6;
        let mut rng = thread_rng();

        for i in 0..num_iterations {
            // Ranomd query length between 16 and 32
            let q_len = rng.gen_range(16..32);
            let text_len = rng.gen_range(50..100);
            // Ranomd text length 50, 1000
            let text = random_dna_seq(text_len);
            let query = random_dna_seq(q_len);

            let query_transposed = TQueries::<U32>::new(&[query.clone()], false);
            let sassy_all_matches = sassy_searcher.search_all(&query, &text, k as usize);
            let mini_matches = mini_searcher.search(&query_transposed, &text, k as u32);
            let mini_all_matches = mini_matches.iter().map(|x| x.len()).sum();

            // Only keep the first Sassy match for each unique text_end, but retain the full Match object
            let mut seen_ends = std::collections::HashSet::new();
            let sassy_all_matches = sassy_all_matches
                .into_iter()
                .filter(|m| seen_ends.insert(m.text_end))
                .collect::<Vec<_>>();

            eprintln!("\n-------");
            eprintln!("Query: {:?}", String::from_utf8_lossy(&query));
            eprintln!("Text: {:?}", String::from_utf8_lossy(&text));
            eprintln!("Sassy matches");
            for m in &sassy_all_matches {
                eprintln!(
                    "Sassy match text end: {} on strand {:?}",
                    m.text_end, m.strand
                );
                let match_slice = &text[m.text_start..m.text_end];

                println!("S match slice: {:?}", String::from_utf8_lossy(match_slice));
            }
            eprintln!("Mini matches");
            for m in mini_matches.iter().flatten() {
                eprintln!("Mini match text end: {}", m);
                let match_slice =
                    &text[(m.saturating_sub(q_len + k as usize))..(m + 1).min(text.len())];
                println!(
                    "Mini match slice: {:?}",
                    String::from_utf8_lossy(match_slice)
                );
            }
            assert_eq!(sassy_all_matches.len(), mini_all_matches);
        }
    }

    #[test]
    fn fuzz_against_sassy_no_overhang() {
        fuzz_against_sassy(None);
    }

    #[test]
    #[ignore = "Not sure if this is a bug - see overhang_bug_or_not test"]
    fn fuzz_against_sassy_with_overhang() {
        fuzz_against_sassy(Some(0.5));
    }

    #[test]
    fn overhang_bug_or_not() {
        use sassy::profiles::Iupac;
        use sassy::Searcher as SassySearcher;

        #[rustfmt::skip]
        let q =               b"TCCGGACCCATGGATT";
        let t = b"CGGCTCAAGATGAGTCC"; //^^^^^^^^^ overhang = 6.5 -> floored -> 6.0?

        // Sassy
        let mut sassy_searcher = SassySearcher::<Iupac>::new_fwd_with_overhang(0.5);
        let sassy_matches = sassy_searcher.search_all(q, &t, 6);
        let unique_ends = sassy_matches.iter().map(|m| m.text_end).collect::<Vec<_>>();
        for m in unique_ends {
            println!("{:?}", m);
        }

        // Mini
        let mut mini_searcher = Searcher::<U32>::new(Some(0.5));
        let query_transposed = TQueries::<U32>::new(&[q.to_vec()], false);
        let mini_matches = mini_searcher.search(&query_transposed, t, 6);
        for m in mini_matches {
            println!("{:?}", m);
        }

        // So mini search does not find the overhang using k = 6, but does find it using k = 7
        // though score stil is
        // let new_score = score - (self.alpha * (i + 1) as f32).floor() as u32;
    }
}

/*

    CCTCGGATATACTCCAG

    sassy
    CTGGAGTACATACTAGA

    mini
    AATTCGCTGGAGTACATACTAG
    ATTCGCTGGAGTACATACTAGA

*/
