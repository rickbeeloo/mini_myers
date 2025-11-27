use crate::iupac::get_encoded;
use crate::tqueries::TQueries;
use crate::{backend::SimdBackend, tqueries};
use pa_types::{Cigar, CigarOp, Cost};

#[derive(Clone, Copy, Default)]
struct BlockState<S: Copy> {
    vp: S,
    vn: S,
    cost: S,
    failure_mask: S,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub text_start: usize,
    pub text_end: usize,
    pub pattern_start: usize,
    pub pattern_end: usize,
    pub cost: Cost,
    pub strand: Strand,
    pub cigar: Cigar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strand {
    Fwd,
    Rc,
}

#[derive(Debug, Clone)]
pub struct Alignment {
    pub edits: u32,
    pub operations: Cigar,
    pub start: usize,
    pub end: usize,
    pub query_idx: usize,
    pub strand: Strand,
}

// We use this in multi-trace
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub query_idx: usize,
    pub end_position: usize,
}

pub struct Searcher<B: SimdBackend> {
    blocks: Vec<BlockState<B::Simd>>,
    positions: Vec<Vec<usize>>,
    alpha_pattern: u64,
    alpha: f32,
    history: Vec<QueryHistory<B::Simd>>,
    search_hits: Vec<SearchHit>,
}

impl<B: SimdBackend> Default for Searcher<B> {
    fn default() -> Self {
        Self::new(None)
    }
}

pub struct QueryHistory<S: Copy> {
    pub steps: Vec<SimdHistoryStep<S>>,
}

struct SimdHistoryStep<S: Copy> {
    vp: S,
    vn: S,
    eq: S,
}

impl<B: SimdBackend> Searcher<B> {
    pub fn new(alpha: Option<f32>) -> Self {
        let alpha_val = alpha.unwrap_or(1.0);
        Self {
            blocks: Vec::new(),
            positions: Vec::new(),
            alpha_pattern: Self::generate_alpha_mask(alpha_val, 64),
            alpha: alpha_val,
            history: Vec::new(),
            search_hits: Vec::new(),
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
                    cost: zero,
                    failure_mask: all_ones,
                },
            );
        }
        if self.positions.len() < total_queries {
            self.positions.resize(total_queries, Vec::new());
        }
        if self.history.len() < total_queries {
            self.history
                .resize_with(total_queries, || QueryHistory { steps: Vec::new() });
        }
    }

    #[inline(always)]
    fn reset_state(&mut self, t_queries: &TQueries<B>, alpha_pattern: u64) {
        let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.query_length)); // bit much maybe to not have branches
        let masked_alpha: u64 = alpha_pattern & length_mask;
        let initial_cost = B::splat_scalar(B::scalar_from_i64(masked_alpha.count_ones() as i64));
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
                block.cost = initial_cost;
            }
        }
    }

    #[inline(always)]
    fn myers_step(
        vp: B::Simd,
        vn: B::Simd,
        cost: B::Simd,
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

        let cost_out = (cost + ph_bit) - mh_bit;

        (vp_out, vn_out, cost_out)
    }

    // Searching without tracing
    pub fn search_with_hits(
        &mut self,
        t_queries: &TQueries<B>,
        text: &[u8],
        k: u32,
    ) -> &[SearchHit] {
        let num_blocks = t_queries.n_simd_blocks;
        let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.query_length));
        let alpha_pattern = self.alpha_pattern & length_mask;

        self.ensure_capacity(num_blocks, t_queries.n_queries);
        self.reset_state(t_queries, alpha_pattern);
        self.search_hits.clear();

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
            let encoded = get_encoded(c) as usize;
            let peq_block_start_index = encoded * num_blocks;

            for block_i in 0..num_blocks {
                unsafe {
                    let eq = *peqs_ptr.add(peq_block_start_index + block_i);
                    let block = &mut *blocks_ptr.add(block_i);
                    let (vp_out, vn_out, cost_out) = Self::myers_step(
                        block.vp,
                        block.vn,
                        block.cost,
                        eq,
                        last_bit_shift,
                        last_bit_mask,
                    );
                    block.vp = vp_out;
                    block.vn = vn_out;
                    block.cost = cost_out;
                    let gt_mask = B::simd_gt(cost_out, k_simd);
                    if gt_mask != B::splat_all_ones() {
                        let mask_arr = B::to_array(gt_mask);
                        let mask_slice = mask_arr.as_ref();

                        for (lane_idx, &val) in mask_slice.iter().enumerate() {
                            if val == zero_scalar {
                                let query_idx = block_i * B::LANES + lane_idx;
                                if query_idx < t_queries.n_queries {
                                    self.search_hits.push(SearchHit {
                                        query_idx,
                                        end_position: idx,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Handle suffix overhang
        if self.alpha_pattern != !0 {
            let eq = B::splat_all_ones();
            let steps_needed = t_queries.query_length;
            let blocks_ptr = self.blocks.as_mut_ptr();
            let mut current_text_pos = text.len();

            for i in 0..steps_needed {
                for block_i in 0..num_blocks {
                    unsafe {
                        let block = &mut *blocks_ptr.add(block_i);
                        let (vp_out, vn_out, cost_out) = Self::myers_step(
                            block.vp,
                            block.vn,
                            block.cost,
                            eq,
                            last_bit_shift,
                            last_bit_mask,
                        );
                        block.vp = vp_out;
                        block.vn = vn_out;
                        block.cost = cost_out;

                        for (lane_idx, cost) in B::to_array(cost_out).as_ref().iter().enumerate() {
                            // Current cost is just vs 1111 eq mask, all matches, we correct for overhang alpha cost here
                            // reduce the edits
                            let cost = B::scalar_to_u64(*cost);
                            let new_cost = cost as f32 + (self.alpha * (i + 1) as f32);
                            let new_cost = new_cost.floor() as u32;
                            if new_cost <= k {
                                let query_idx = block_i * B::LANES + lane_idx;
                                if query_idx < t_queries.n_queries {
                                    self.search_hits.push(SearchHit {
                                        query_idx,
                                        end_position: current_text_pos,
                                    });
                                }
                            }
                        }
                    }
                }
                current_text_pos += 1;
            }
        }

        &self.search_hits
    }

    // We trace multiple queries against multiple "text ends" of the same
    // text (could also be different though)
    fn trace_batch(
        &mut self,
        t_queries: &TQueries<B>,
        text: &[u8],
        hits: &[SearchHit],
        k: u32,
    ) -> Vec<Alignment> {
        assert!(hits.len() <= B::LANES, "Batch size must be <= LANES");

        let left_buffer = t_queries.query_length + k as usize;

        // todo: re-use alloc. Although we subtract the same |q| +k
        // if a query ends near the start of the text this slices
        // can we shorter than the others - important later to
        // stop pushing to history
        let mut query_indices = vec![0; B::LANES];
        let mut approx_slices = vec![(0, 0); B::LANES];
        let batch_size = hits.len();
        for (i, hit) in hits.iter().enumerate() {
            query_indices[i] = hit.query_idx;
            approx_slices[i] = (
                hit.end_position.saturating_sub(left_buffer),
                hit.end_position,
            );
        }

        let num_blocks = 1; // Just one block now which we check in assert
        self.ensure_capacity(num_blocks, batch_size);

        for i in 0..batch_size {
            self.history[i].steps.clear();
        }

        let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.query_length));
        self.reset_state(t_queries, self.alpha_pattern & length_mask);

        let last_bit_shift = (t_queries.query_length - 1) as u32;
        let last_bit_mask = B::splat_one() << last_bit_shift;
        let all_ones = B::splat_all_ones();
        let zero_scalar = B::scalar_from_i64(0);
        let one_mask: <B as SimdBackend>::Scalar = B::mask_word_to_scalar(!0);

        let blocks_ptr = self.blocks.as_mut_ptr();

        // Ensure we covert at least the longer slice entirely
        let max_len = t_queries.query_length + k as usize + 1;

        // Forward pass but now we do track history vp,vn, (eq)
        for i in 0..max_len {
            unsafe {
                let block = &mut *blocks_ptr;
                let mut eq_arr = B::LaneArray::default();
                let mut keep_mask_arr = B::LaneArray::default();

                let eq_slice = eq_arr.as_mut();
                let keep_slice = keep_mask_arr.as_mut();

                // To not have to re-encode the queries for a trace we get the `eq` mask
                // and 'set' the eq by indexing into the encoded characters for the queries we are tracing
                // each lane here then is an eq mask for different text regions
                for lane in 0..batch_size {
                    let q_idx = query_indices[lane];
                    let start = approx_slices[lane].0;
                    // Each lane holds a slice from the text but all have different start positions
                    // we use to use those (abs_pos) to  know when we are in the text still
                    let abs_pos = i + start;
                    if abs_pos < text.len() {
                        let cur_char = text[abs_pos];
                        let enc = get_encoded(cur_char) as usize;
                        eq_slice[lane] = B::mask_word_to_scalar(t_queries.peq_masks[enc][q_idx]);
                        // If keep is all ones we are in bounds of the text, if all zeros (else)
                        // we are outside the current text
                        keep_slice[lane] = one_mask;
                    } else {
                        eq_slice[lane] = zero_scalar;
                        keep_slice[lane] = zero_scalar;
                    }
                }

                let eq = B::from_array(eq_arr);
                let keep_mask = B::from_array(keep_mask_arr);
                let freeze_mask = all_ones ^ keep_mask;

                let (vp_new, vn_new, cost_new) = Self::myers_step(
                    block.vp,
                    block.vn,
                    block.cost,
                    eq,
                    last_bit_shift,
                    last_bit_mask,
                );

                let vp_masked = (vp_new & keep_mask) | (block.vp & freeze_mask);
                let vn_masked = (vn_new & keep_mask) | (block.vn & freeze_mask);
                let cost_masked = (cost_new & keep_mask) | (block.cost & freeze_mask);
                let freeze_arr = B::to_array(freeze_mask);

                let eq_arr = B::to_array(eq);
                for lane in 0..batch_size {
                    // We only add to the history if we were progressing IN the text
                    let is_frozen = B::scalar_to_u64(freeze_arr.as_ref()[lane]) != 0;
                    if !is_frozen {
                        let eq_scalar = eq_arr.as_ref()[lane];
                        self.history[lane].steps.push(SimdHistoryStep {
                            vp: vp_masked,
                            vn: vn_masked,
                            eq: B::splat_scalar(eq_scalar),
                        });
                    }
                }

                block.vp = vp_masked;
                block.vn = vn_masked;
                block.cost = cost_masked;
            }
        }

        // Handle suffix overhang, eq bits to 1111 (i.e. match)
        // NOTE: in search we do correct the cost here based on overhang cost, but we don't mutate vn,vp.
        // so in the trace these would show as regular matches we handled that later in `traceback_single`
        if self.alpha_pattern != !0 {
            let eq = B::splat_all_ones();
            let blocks_ptr = self.blocks.as_mut_ptr();
            for i in 0..t_queries.query_length {
                unsafe {
                    let block = &mut *blocks_ptr;
                    let (vp_out, vn_out, cost_out) = Self::myers_step(
                        block.vp,
                        block.vn,
                        block.cost,
                        eq,
                        last_bit_shift,
                        last_bit_mask,
                    );
                    let eq_arr = B::to_array(eq);
                    for lane in 0..batch_size {
                        let eq_scalar = eq_arr.as_ref()[lane];
                        self.history[lane].steps.push(SimdHistoryStep {
                            vp: vp_out,
                            vn: vn_out,
                            eq: B::splat_scalar(eq_scalar),
                        });
                    }
                    block.vp = vp_out;
                    block.vn = vn_out;
                }
            }
        }

        // todo: also re-use this alloc via self
        let mut alignments = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let aln = self.traceback_single(
                i,
                query_indices[i],
                t_queries,
                approx_slices[i],
                text.len() as isize,
            );
            alignments.push(aln);
        }

        alignments
    }

    pub fn trace_all_hits(
        &mut self,
        t_queries: &TQueries<B>,
        text: &[u8],
        k: u32,
    ) -> Vec<Alignment> {
        self.search_with_hits(t_queries, text, k);
        let total_hits = self.search_hits.len();

        let mut all_alignments = Vec::with_capacity(total_hits);

        // Process in batches
        for chunk_start in (0..total_hits).step_by(B::LANES) {
            let chunk_end = (chunk_start + B::LANES).min(total_hits);
            // todo: prevent this clone later
            let batch: Vec<SearchHit> = self.search_hits[chunk_start..chunk_end].to_vec();
            let batch_alignments = self.trace_batch(t_queries, text, &batch, k);
            all_alignments.extend(batch_alignments);
        }

        all_alignments
    }

    fn get_cost_at(
        &self,
        query_idx: usize,
        step_idx: isize, // Index into the history vector - target/text
        query_pos_idx: isize,
    ) -> isize {
        // If step is <0 we are basically in the alpha column, for example 0.5 = 10101010
        // so at any points along the first column the cost is the sum of ones
        if step_idx < 0 {
            let mask = if query_pos_idx >= 63 {
                !0u64
            } else {
                (1u64 << (query_pos_idx + 1)) - 1
            };
            return (self.alpha_pattern & mask).count_ones() as isize;
        }

        // This is "normal" history
        let step_data = &self.history[query_idx].steps[step_idx as usize];
        let lane = query_idx % B::LANES;
        let vp_bits = Self::extract_simd_lane(step_data.vp, lane);
        let vn_bits = Self::extract_simd_lane(step_data.vn, lane);

        let mask = if query_pos_idx >= 63 {
            !0u64
        } else {
            (1u64 << (query_pos_idx + 1)) - 1
        };

        let pos = (vp_bits & mask).count_ones() as isize;
        let neg = (vn_bits & mask).count_ones() as isize;

        pos - neg
    }

    #[inline(always)]
    fn extract_simd_lane(simd_val: B::Simd, lane: usize) -> u64 {
        let arr = B::to_array(simd_val);
        B::scalar_to_u64(arr.as_ref()[lane])
    }

    #[inline(always)]
    fn traceback_single(
        &self,
        query_idx: usize,          // in CURRENT batch
        original_query_idx: usize, // originally based on encoding
        t_queries: &TQueries<B>,
        slice: (usize, usize),
        text_len: isize,
    ) -> Alignment {
        let history = &self.history[query_idx];
        let steps = &history.steps;
        let query_len = t_queries.query_length as isize;

        let max_step = (slice.1 - slice.0) as isize;
        let mut curr_step = max_step;
        let mut query_pos = query_len - 1;
        let mut cigar: Cigar = Cigar::default();

        // The trace while inside the computed history
        let mut cost_correction = 0;
        while curr_step >= 0 && query_pos >= 0 {
            let curr_cost = self.get_cost_at(query_idx, curr_step, query_pos);

            // Neighbors
            let diag_cost = self.get_cost_at(query_idx, curr_step - 1, query_pos - 1);
            let up_cost = self.get_cost_at(query_idx, curr_step, query_pos - 1);
            let left_cost = self.get_cost_at(query_idx, curr_step - 1, query_pos);

            let step = &steps[curr_step as usize];
            let lane = query_idx % B::LANES;
            let eq_bits = Self::extract_simd_lane(step.eq, lane);
            let is_match = (eq_bits & (1u64 << query_pos)) != 0;
            let match_cost = if is_match { 0 } else { 1 };

            // suffix overhang
            if slice.0 + curr_step as usize > (text_len as usize - 1) {
                query_pos -= 1;
                curr_step -= 1;
                // Overhang are just "matches" (eq=111..) in the code above
                // so fix it here by keeping track of a cost_correction
                cost_correction += 1;
            // regular, same order as sassy for our fuzz
            } else if curr_cost == diag_cost + match_cost && is_match {
                cigar.push(CigarOp::Match);
                curr_step -= 1;
                query_pos -= 1;
            } else if curr_cost == left_cost + 1 {
                cigar.push(CigarOp::Ins);
                curr_step -= 1;
            } else if curr_cost == diag_cost + match_cost && !is_match {
                cigar.push(CigarOp::Sub);
                curr_step -= 1;
                query_pos -= 1;
            } else if curr_cost == up_cost + 1 {
                cigar.push(CigarOp::Del);
                query_pos -= 1;
            } else {
                panic!("Invalid traceback step reached :(");
            }
        }

        // If there is any query left after the trace it's overhanging
        // only if alpha is disabled (==1.0) we store deletions like in sassy
        if query_pos >= 0 && self.alpha == 1.0 {
            let overhang_len = (query_pos.abs() + 1) as usize;
            for _ in 0..overhang_len {
                cigar.push(CigarOp::Del);
            }
        }

        // Correct start so for the user we have in-bound indices
        let slice_start_offset = (curr_step + 1).max(0);
        let text_start = slice.0 + slice_start_offset as usize;

        // todo: nicer to verify this against the trace
        let mut final_raw_cost = self.get_cost_at(query_idx, max_step, query_len - 1);

        // Apply correction for overhang
        if cost_correction > 0 {
            // eprintln!("Cost correction");
            // eprintln!("Alpha: {}", self.alpha);
            final_raw_cost += (cost_correction as f32 * self.alpha).floor() as isize;
        }

        cigar.reverse();

        Alignment {
            edits: final_raw_cost as u32,
            operations: cigar,
            start: text_start,
            end: (slice.1 as usize).min((text_len - 1) as usize),
            query_idx: original_query_idx % t_queries.n_original_queries, // for rc, use fwd idx
            strand: if original_query_idx >= t_queries.n_original_queries {
                Strand::Rc
            } else {
                Strand::Fwd
            },
        }
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
    use crate::backend::SimdBackend;

    #[cfg(test)]
    type TestBackend = crate::backend::U32; // todo: trace on 16, 64

    #[test]
    fn test_search_with_hits_simple() {
        let mut searcher = Searcher::<TestBackend>::new(None);

        let queries = vec![b"ACGT".to_vec(), b"TGCA".to_vec()];
        let t_queries = TQueries::<TestBackend>::new(&queries, false);
        let text = b"AAACGTTTGCAAA";
        //                      0123456789-
        let k = 0; // exact match only

        let hits = searcher.search_with_hits(&t_queries, text, k);

        // Should find ACGT at position 5
        // Should find TGCA at position 10
        assert_eq!(hits.len(), 2, "Should find 2 exact matches");

        let hit0 = hits.iter().find(|h| h.query_idx == 0).unwrap();
        assert_eq!(hit0.end_position, 5, "ACGT should end at position 5");

        let hit1 = hits.iter().find(|h| h.query_idx == 1).unwrap();
        assert_eq!(hit1.end_position, 10, "TGCA should end at position 10");
    }

    #[test]
    fn test_search_with_hits_with_errors() {
        let mut searcher = Searcher::<TestBackend>::new(None);

        let queries = vec![b"ACGT".to_vec()];
        let t_queries = TQueries::<TestBackend>::new(&queries, false);

        // Text has ATGT (1 substitution from ACGT)
        let text = b"AAATGTAAA";
        let k = 1;

        let hits = searcher.search_with_hits(&t_queries, text, k);

        assert!(hits.len() >= 1, "Should find match with 1 edit");
        assert_eq!(hits[0].query_idx, 0);
        assert_eq!(hits[0].end_position, 5, "Match should end at position 5");
    }

    #[test]
    fn test_trace_batch_exact_match() {
        let mut searcher = Searcher::<TestBackend>::new(None);

        let queries = vec![b"ACGT".to_vec()];
        let t_queries = TQueries::<TestBackend>::new(&queries, false);
        let text = b"AAACGTAAA";
        let k = 0;

        let hits = vec![SearchHit {
            query_idx: 0,
            end_position: 5,
        }];

        let alignments = searcher.trace_batch(&t_queries, text, &hits, k);

        let a = &alignments[0];
        assert_eq!(a.start, 2);
        assert_eq!(a.end, 5);
        assert_eq!(a.operations.to_string(), "4=");
    }
    //}

    #[test]
    fn test_trace_batch_with_substitution() {
        let mut searcher = Searcher::<TestBackend>::new(None);

        let queries = vec![b"ACGT".to_vec()];
        let t_queries = TQueries::<TestBackend>::new(&queries, false);

        // Text has ATGT (C->T substitution)
        let text = b"AAATGTAAA";
        let k = 1;

        let hits = vec![SearchHit {
            query_idx: 0,
            end_position: 5,
        }];

        let alignments = searcher.trace_batch(&t_queries, text, &hits, k);
        for a in alignments.iter() {
            println!("a: {:?}", a);
        }
        assert_eq!(alignments.len(), 1);
        assert_eq!(alignments[0].edits, 1, "Should have 1 edit");
        assert_eq!(alignments[0].start, 2);
        assert_eq!(alignments[0].end, 5);
    }

    #[test]
    fn test_trace_batch_with_insertion() {
        let mut searcher = Searcher::<TestBackend>::new(None);

        let queries = vec![b"ACGT".to_vec()];
        let t_queries = TQueries::<TestBackend>::new(&queries, false);

        // Text has ACAGT (extra A inserted)
        let text = b"AAACAGTTAAA";
        let k = 1;

        let hits = vec![SearchHit {
            query_idx: 0,
            end_position: 6,
        }];

        let alignments = searcher.trace_batch(&t_queries, text, &hits, k);

        assert_eq!(alignments.len(), 1);
        assert_eq!(alignments[0].edits, 1, "Should have 1 edit");

        println!("{:?}", alignments);
    }

    #[test]
    fn test_trace_batch_multiple_queries() {
        let mut searcher = Searcher::<TestBackend>::new(None);

        let queries = vec![b"ACGT".to_vec(), b"TGCA".to_vec(), b"GGCC".to_vec()];
        let t_queries = TQueries::<TestBackend>::new(&queries, false);
        let text = b"AAACGTTTGCAAGGCCAA";
        //                      0123456789-1234567
        let k = 0;

        let hits = vec![
            SearchHit {
                query_idx: 0,
                end_position: 5,
            },
            SearchHit {
                query_idx: 1,
                end_position: 10,
            },
            SearchHit {
                query_idx: 2,
                end_position: 15,
            },
        ];

        let alignments = searcher.trace_batch(&t_queries, text, &hits, k);

        assert_eq!(alignments.len(), 3);

        for a in &alignments {
            println!("a: {:?}", a);
        }

        // Verify all have 0 edits (exact matches)
        for aln in &alignments {
            assert_eq!(aln.edits, 0, "All are exact");
        }
    }

    #[test]
    fn test_trace_all_hits_integration() {
        let mut searcher = Searcher::<TestBackend>::new(None);

        let queries = vec![b"ACGT".to_vec(), b"TGCA".to_vec()];
        let t_queries = TQueries::<TestBackend>::new(&queries, false);
        let text = b"AAACGTTTGCAAA";
        //                      0123456789-12
        let k = 0;

        let alignments = searcher.trace_all_hits(&t_queries, text, k);

        assert_eq!(alignments.len(), 2, "Should find and trace 2 matches");

        // Check both alignments are correct
        let aln0 = alignments.iter().find(|a| a.query_idx == 0).unwrap();
        assert_eq!(aln0.edits, 0);
        assert_eq!(aln0.start, 2);
        assert_eq!(aln0.end, 5);

        let aln1 = alignments.iter().find(|a| a.query_idx == 1).unwrap();
        assert_eq!(aln1.edits, 0);
        assert_eq!(aln1.start, 7);
        assert_eq!(aln1.end, 10);
    }

    #[test]
    fn test_alpha_overhang() {
        let mut searcher = Searcher::<TestBackend>::new(Some(0.5));

        let queries = vec![b"ACGT".to_vec()];
        let t_queries = TQueries::<TestBackend>::new(&queries, false);

        // Query can match at the end with suffix overhang
        let text = b"AC";
        let k = 2;

        let hits = searcher.search_with_hits(&t_queries, text, k);

        // With alpha=0.5, missing 2 characters costs: floor(0.5*1) + floor(0.5*2) = 0 + 1 = 1
        assert!(!hits.is_empty(), "Should find match with suffix overhang");
    }

    #[test]
    fn test_prefix_overhang() {
        let mut searcher = Searcher::<TestBackend>::new(Some(0.5));

        let queries = vec![b"AAAGT".to_vec()];
        let t_queries = TQueries::<TestBackend>::new(&queries, false);
        let text = b"GTCCCCCCCCC";
        let k = 2;
        let hits = searcher.trace_all_hits(&t_queries, text, k);
        assert!(!hits.is_empty(), "Should find match with prefix overhang");
        for h in hits.iter() {
            println!(
                "start: {}, end: {}, edits: {}, cigar: {}",
                h.start,
                h.end,
                h.edits,
                h.operations.to_string()
            );
        }
    }

    #[test]
    fn test_no_matches() {
        let mut searcher = Searcher::<TestBackend>::new(None);

        let queries = vec![b"ACGT".to_vec()];
        let t_queries = TQueries::<TestBackend>::new(&queries, false);
        let text = b"TTTTTTTT";
        let k = 1; // Not enough to match

        let hits = searcher.search_with_hits(&t_queries, text, k);

        assert_eq!(hits.len(), 0, "Should find no matches");
    }

    #[test]
    fn test_multiple_hits_same_query() {
        let mut searcher = Searcher::<TestBackend>::new(None);

        let queries = vec![b"AC".to_vec()];
        let t_queries = TQueries::<TestBackend>::new(&queries, false);
        let text = b"ACACAC"; //1,3,5
        let k = 0;

        let hits = searcher.search_with_hits(&t_queries, text, k);

        // Should find AC at positions 1, 3, 5
        assert_eq!(hits.len(), 3, "Should find 3");
        assert!(hits.iter().all(|h| h.query_idx == 0));

        let positions: Vec<_> = hits.iter().map(|h| h.end_position).collect();
        assert!(positions.contains(&1));
        assert!(positions.contains(&3));
        assert!(positions.contains(&5));
    }

    #[test]
    fn test_batch_size_edge_case() {
        let mut searcher = Searcher::<TestBackend>::new(None);

        // Create exactly LANES queries to test batch boundary
        let n_queries = TestBackend::LANES;
        let queries: Vec<Vec<u8>> = (0..n_queries)
            .map(|i| vec![b'A', b'C', b'G', b'T'][i % 4])
            .map(|c| vec![c; 4])
            .collect();

        for q in queries.iter() {
            println!("q: {}", String::from_utf8_lossy(&q));
        }

        println!("Number of queries: {}", n_queries);

        /*
           q: AAAA
           q: CCCC
           q: GGGG
           q: TTTT
           q: AAAA
           q: CCCC
           q: GGGG
           q: TTTT

        */

        let t_queries = TQueries::<TestBackend>::new(&queries, false);
        let text = b"AAAACCCCGGGGTTTT";
        let k = 2;

        let alignments = searcher.trace_all_hits(&t_queries, text, k);

        // Should handle full LANES batch correctly
        assert!(alignments.len() > 0, "Should find some matches");
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

    fn fuzz_against_sassy(alpha: Option<f32>, include_rc: bool) {
        use crate::backend::U32;
        use rand::thread_rng;
        use rand::Rng;
        use sassy::profiles::Iupac;
        use sassy::Searcher as SassySearcher;
        // Sassy searcher
        let mut sassy_searcher = if alpha.is_some() {
            SassySearcher::<Iupac>::new_fwd_with_overhang(alpha.unwrap())
        } else {
            println!("Creating sassy searcher without overhang");
            SassySearcher::<Iupac>::new_fwd()
        };
        let mut mini_searcher = Searcher::<U32>::new(alpha);

        let num_iterations = 1_000_000;
        let mut rng = thread_rng();

        for _i in 0..num_iterations {
            // K in range of 0..10
            let k = rng.gen_range(0..10);
            // Random query length between 5 and 8
            let q_len = rng.gen_range(5..8);
            let text_len = rng.gen_range(10..60);
            let mut text = random_dna_seq(text_len);
            let query = random_dna_seq(q_len);

            // Add mutated query to the end of the text
            let mutated_query = apply_edits(&query, k / 2);
            let text_end = text.len().saturating_sub(mutated_query.len());
            let mutated_prefix = mutated_query[..mutated_query.len() / 2].to_vec();
            text.splice(
                text_end..text_end + mutated_prefix.len(),
                mutated_prefix.clone(),
            );

            let query_transposed = TQueries::<U32>::new(&[query.clone()], include_rc);

            let (sassy_matches, sassy_pairs) = if !include_rc {
                let matches = sassy_searcher.search_all(&query, &text, k as usize);
                let mut pairs = matches
                    .iter()
                    .map(|m| {
                        (
                            m.text_start,
                            m.text_end,
                            m.cost as isize,
                            m.cigar.to_string(),
                        )
                    })
                    .collect::<Vec<_>>();
                pairs
                    .dedup_by_key(|(start, end, cost, cigar)| (*start, *end, *cost, cigar.clone()));
                // Sort them
                pairs.sort_by_key(|(start, end, cost, cigar)| (*start, *end, *cost, cigar.clone()));

                (matches, pairs)
            } else {
                // fwd search
                let mut matches = sassy_searcher.search_all(&query, &text, k as usize);
                // rc search
                let query_rc = crate::iupac::reverse_complement(&query);
                let mut rc_matches = sassy_searcher.search_all(&query_rc, &text, k as usize);

                // For pairs first get unique per fwd, rc
                let mut fwd_pairs = matches
                    .iter()
                    .map(|m| {
                        (
                            m.text_start,
                            m.text_end,
                            m.cost as isize,
                            m.cigar.to_string(),
                        )
                    })
                    .collect::<Vec<_>>();
                fwd_pairs
                    .dedup_by_key(|(start, end, cost, cigar)| (*start, *end, *cost, cigar.clone()));

                let mut rc_pairs = rc_matches
                    .iter()
                    .map(|m| {
                        (
                            m.text_start,
                            m.text_end,
                            m.cost as isize,
                            m.cigar.to_string(),
                        )
                    })
                    .collect::<Vec<_>>();
                rc_pairs
                    .dedup_by_key(|(start, end, cost, cigar)| (*start, *end, *cost, cigar.clone()));

                fwd_pairs.extend(rc_pairs.iter().cloned());

                // Sort it
                fwd_pairs
                    .sort_by_key(|(start, end, cost, cigar)| (*start, *end, *cost, cigar.clone()));

                // combine both forward and reverse complement matches
                matches.append(&mut rc_matches);
                (matches, fwd_pairs)
            };

            let mini_matches = mini_searcher.trace_all_hits(&query_transposed, &text, k as u32);

            let mut mini_pairs = mini_matches
                .iter()
                .map(|m| {
                    (
                        m.start,
                        m.end + 1,
                        m.edits as isize,
                        m.operations.to_string(),
                    )
                })
                .collect::<Vec<_>>();

            mini_pairs
                .sort_by_key(|(start, end, edits, cigar)| (*start, *end, *edits, cigar.clone()));

            // eprintln!("Query: {:?}", String::from_utf8_lossy(&query));
            // eprintln!("Text:  {:?}", String::from_utf8_lossy(&text));

            // // Print nicely formatted columns for better readability,
            // // using same spacing for header as for data lines
            // eprintln!(
            //     "{:>5}:{:<5} {:<14} {:<16}     {:>5}:{:<5} {:<14} {:<16}     {}",
            //     "Sassy",
            //     "end",
            //     "Sassy cost",
            //     "Sassy cigar",
            //     "Mini",
            //     "end",
            //     "Mini edits",
            //     "Mini cigar",
            //     "Equal"
            // );
            for (s, m) in sassy_pairs.iter().zip(mini_pairs.iter()) {
                let (s_start, s_end, s_cost, s_cigar) = s;
                let (m_start, m_end, m_edits, m_cigar) = m;
                eprintln!(
                    "{:>5}:{:<5} {:<14} {:<16}     {:>5}:{:<5} {:<14} {:<16}     {}",
                    s_start,
                    s_end,
                    s_cost,
                    s_cigar,
                    m_start,
                    m_end,
                    m_edits,
                    m_cigar,
                    if s == m { "OK" } else { "FAIL" }
                );
            }
            // eprintln!(
            //     "Same length: {}, diff: {}",
            //     sassy_pairs.len() == mini_pairs.len(),
            //     sassy_pairs.len() as isize - mini_pairs.len() as isize
            // );
            // if sassy_pairs.len() != mini_pairs.len() {
            //     if sassy_pairs.len() < mini_pairs.len() {
            //         for m in mini_pairs.iter() {
            //             eprintln!("\tMini: {:?}", m);
            //         }
            //         for m in sassy_matches.iter() {
            //             eprintln!("\tSassy: {:?}", m);
            //         }
            //     } else {
            //         for m in sassy_matches.iter() {
            //             eprintln!("\tSassy: {:?}", m);
            //         }
            //         for m in mini_pairs.iter() {
            //             eprintln!("\tMini: {:?}", m);
            //         }
            //     }
            // }

            assert_eq!(sassy_pairs, mini_pairs);
        }
    }

    #[test]
    fn fuzz_without_overhang() {
        fuzz_against_sassy(None, false);
    }

    #[test]
    fn fuzz_with_overhang() {
        fuzz_against_sassy(Some(0.5), false);
    }

    #[test]
    fn fuzz_without_overhang_inc_rc() {
        fuzz_against_sassy(None, true);
    }

    #[test]
    fn fuzz_with_overhang_inc_rc() {
        fuzz_against_sassy(Some(0.5), true);
    }

    #[test]
    fn mini_trace_bug() {
        use crate::backend::U32;
        use crate::iupac::reverse_complement;
        use crate::TQueries;
        use sassy::profiles::Iupac;
        use sassy::Searcher as SassySearcher;

        let q = b"GTCCGAC";
        let q_rc = crate::iupac::reverse_complement(q);
        //                   0123456789-1
        let t = b"AAACGAAGTCCTTAGACTGACTTGGCACCAGTATACTCACTTTTTTGTCTCC";
        println!("q: {:?}", String::from_utf8_lossy(q));
        println!("q_rc: {:?}", String::from_utf8_lossy(&q_rc));

        let k = 2;
        let mut searcher = Searcher::<U32>::new(Some(0.5));
        let query_transposed = TQueries::<U32>::new(&[q.to_vec()], true);
        let mini_matches = searcher.trace_all_hits(&query_transposed, t, k as u32);
        for m in mini_matches {
            println!(
                "Mini edits: {} cigar: {} end: {} strand: {:?}",
                m.edits,
                m.operations.to_string(),
                m.end,
                m.strand
            );
        }

        let mut sassy_searcher = SassySearcher::<Iupac>::new_fwd_with_overhang(0.5);
        let sassy_matches = sassy_searcher.search_all(q, &t, k as usize);
        for m in sassy_matches {
            println!(
                "Sassy edits: {} cigar: {} end: {} strand: {:?}",
                m.cost,
                m.cigar.to_string(),
                m.text_end,
                m.strand
            );
        }
        let sassy_matches_rc = sassy_searcher.search_all(&q_rc, &t, k as usize);
        for m in sassy_matches_rc {
            println!(
                "Sassy RC edits: {} cigar: {} end: {} strand: {:?}",
                m.cost,
                m.cigar.to_string(),
                m.text_end,
                m.strand
            );
        }
    }
}
