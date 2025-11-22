use crate::backend::SimdBackend;
use crate::constant::IUPAC_MASKS;
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

    #[inline(always)] // to run just search_asm have to use inline(never)
    pub fn scan(&mut self, t_queries: &TQueries<B>, text: &[u8], k: u32) -> &[bool] {
        let num_blocks = t_queries.n_simd_blocks;
        let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.query_length));
        let alpha_pattern = self.alpha_pattern & length_mask;

        self.ensure_capacity(num_blocks, t_queries.n_queries);
        self.reset_state(t_queries, alpha_pattern);

        let k_simd = B::splat_from_usize(k as usize);
        let last_bit_shift = (t_queries.query_length - 1) as u32;
        let last_bit_mask = B::splat_one() << last_bit_shift;
        for &c in text {
            let encoded = crate::iupac::get_encoded(c) as usize;
            for block_i in 0..num_blocks {
                unsafe {
                    let eq = *t_queries
                        .peqs
                        .get_unchecked(block_i * IUPAC_MASKS + encoded);
                    let vp_in = *self.vp.get_unchecked(block_i);
                    let vn_in = *self.vn.get_unchecked(block_i);
                    let score_in = *self.score.get_unchecked(block_i);

                    let (vp_out, vn_out, score_out) =
                        Self::myers_step(vp_in, vn_in, score_in, eq, last_bit_shift, last_bit_mask);

                    *self.vp.get_unchecked_mut(block_i) = vp_out;
                    *self.vn.get_unchecked_mut(block_i) = vn_out;
                    *self.score.get_unchecked_mut(block_i) = score_out;
                    *self.failure_masks.get_unchecked_mut(block_i) &= B::simd_gt(score_out, k_simd);
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

        for _ in 0..steps_needed {
            for block_i in 0..num_blocks {
                unsafe {
                    let vp_in = *self.vp.get_unchecked(block_i);
                    let vn_in = *self.vn.get_unchecked(block_i);
                    let score_in = *self.score.get_unchecked(block_i);

                    let (vp_out, vn_out, score_out) =
                        Self::myers_step(vp_in, vn_in, score_in, eq, last_bit_shift, last_bit_mask);

                    *self.vp.get_unchecked_mut(block_i) = vp_out;
                    *self.vn.get_unchecked_mut(block_i) = vn_out;
                    *self.score.get_unchecked_mut(block_i) = score_out;

                    *self.failure_masks.get_unchecked_mut(block_i) &= B::simd_gt(score_out, k_simd);
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
            // 0: Substitution, 1: Insertion, 2: Deletion
            // If sequence is getting too short, force insertion
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
                    let idx = rng.gen_range(0..=res.len()); // Can insert at end
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
}
