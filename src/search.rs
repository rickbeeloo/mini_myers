use crate::constant::*;
#[cfg(not(feature = "latest_wide"))]
use wide_v07 as wide;
#[cfg(feature = "latest_wide")]
use wide_v08 as wide;

use crate::tqueries::*;
use wide::{i32x8, CmpEq, CmpGt};

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

/// Searches for all queries in the target sequence using the Myers algorithm.
/// Returning the minimum edits found for the query in the entire target, or
/// `-1.0` if below the provided maximum edit distance `k`.
///
///
/// # Arguments
///
/// * `transposed` - A `TQueries` structure containing the preprocessed queries
/// * `target` - The target DNA sequence to search in (as a byte slice)
/// * `k` - Maximum edit distance threshold.
/// * `alpha` - Optional overhang penalty. When `Some(alpha)`, trailing characters that fall
///   outside the target incur a reduced penalty of `alpha` per character. When `None` or
///   `Some(1.0)`, the standard Myers behaviour is used.
///
/// # Returns
///
/// A vector of `f32` values, one per query, containing:
/// - The minimum edit distance found (0 to k) if a match within threshold is found
/// - `-1.0` if no match within the threshold k was found
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
/// // Search with k=4 (allow up to 4 edits) and no overhang penalty
/// let result = mini_search(&transposed, target, 4, None);
/// // Result: [0.0, 1.0] - ATG found with 0 edits, TTG found with 1 edit
///
/// ```
#[inline(always)]
pub fn mini_search(transposed: &TQueries, target: &[u8], k: u8, alpha: Option<f32>) -> Vec<f32> {
    let nq = transposed.n_queries;
    if nq == 0 {
        return Vec::new();
    }
    let m = transposed.query_length;
    assert!(m > 0 && m <= 32, "query length must be 1..=32");

    let alpha = alpha.unwrap_or(1.0).clamp(0.0, 1.0);
    search_simd_core(transposed, target, k, alpha)
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
/// * `alpha` - Optional overhang penalty (see [`mini_search`]).
/// * `results` - Vector that will be populated with all matches found.
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
/// mini_search_with_positions(&transposed, target, 4, None, &mut results);
/// // Matches are now in results
/// ```
#[inline(always)]
pub fn mini_search_with_positions(
    transposed: &TQueries,
    target: &[u8],
    k: u8,
    alpha: Option<f32>,
    results: &mut Vec<MatchInfo>,
) {
    let nq = transposed.n_queries;
    if nq == 0 {
        return;
    }
    let m = transposed.query_length;
    assert!(m > 0 && m <= 32, "query length must be 1..=32");
    let alpha = alpha.unwrap_or(1.0).clamp(0.0, 1.0);
    search_simd_with_positions_core(transposed, target, k, alpha, results)
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
fn advance_column(
    pv: &mut i32x8,
    mv: &mut i32x8,
    score: &mut i32x8,
    eq: i32x8,
    adjust_vec: i32x8,
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

/// Core SIMD Myers search optionally applying an overhang penalty `alpha`.
/// `alpha` should be in the range `[0.0, 1.0]`, where `1.0` corresponds to the
/// standard Myers edit distance and lower values discount overhangs.
#[inline(never)]
pub(crate) fn search_simd_core(
    transposed: &TQueries,
    target: &[u8],
    k: u8,
    alpha: f32,
) -> Vec<f32> {
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

    let initial_adjusted = (m as i32) * alpha_scaled;
    let mut pv = vec![all_ones; vectors_in_block];
    let mut mv = vec![zero_v; vectors_in_block];
    let mut score = vec![i32x8::splat(m as i32); vectors_in_block];
    let mut best_adjusted = vec![i32x8::splat(initial_adjusted); vectors_in_block];

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
        let encoded = crate::iupac::get_encoded(tb);
        if encoded == INVALID_IUPAC {
            panic!("Target contains invalid IUPAC character: {:?}", tb as char);
        }
        let overhang = m.saturating_sub(idx + 1) as i32;
        let adjust_vec = i32x8::splat(overhang * extra_penalty_scaled);
        let peq_slice = unsafe { transposed.peqs.get_unchecked(encoded as usize) };

        unsafe {
            for block_idx in 0..vectors_in_block {
                let eq = *peq_slice.get_unchecked(block_idx);
                let adjusted = advance_column(
                    pv.get_unchecked_mut(block_idx),
                    mv.get_unchecked_mut(block_idx),
                    score.get_unchecked_mut(block_idx),
                    eq,
                    adjust_vec,
                    ctx,
                );
                let best_slot = best_adjusted.get_unchecked_mut(block_idx);
                *best_slot = (*best_slot).min(adjusted);
            }
        }
    }

    if alpha_scaled < SCALE {
        for trailing in 1..=m {
            let adjust_vec = i32x8::splat((trailing as i32) * extra_penalty_scaled);
            unsafe {
                for block_idx in 0..vectors_in_block {
                    let adjusted = advance_column(
                        pv.get_unchecked_mut(block_idx),
                        mv.get_unchecked_mut(block_idx),
                        score.get_unchecked_mut(block_idx),
                        zero_eq_vec,
                        adjust_vec,
                        ctx,
                    );
                    let best_slot = best_adjusted.get_unchecked_mut(block_idx);
                    *best_slot = (*best_slot).min(adjusted);
                }
            }
        }
    }

    let mut result = vec![-1.0f32; nq];
    let neg_mask = i32x8::splat(i32::MAX);

    for (block_idx, &min_adj) in best_adjusted.iter().enumerate() {
        #[cfg(not(feature = "latest_wide"))]
        let mask = all_ones ^ min_adj.cmp_gt(k_scaled_vec);
        #[cfg(feature = "latest_wide")]
        let mask = all_ones ^ min_adj.simd_gt(k_scaled_vec);

        let selected = mask.blend(min_adj, neg_mask);
        let base = block_idx * SIMD_LANES;
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

/// SIMD search with position tracking.
/// Tracks ALL positions where score <= k by incrementally collecting matches.
/// Core SIMD Myers search with position tracking and optional overhang penalty `alpha`.
#[inline(always)]
fn search_simd_with_positions_core(
    transposed: &TQueries,
    target: &[u8],
    k: u8,
    alpha: f32,
    results: &mut Vec<MatchInfo>,
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

    let mut pv = vec![all_ones; vectors_in_block];
    let mut mv = vec![zero_v; vectors_in_block];
    let mut score = vec![i32x8::splat(m as i32); vectors_in_block];

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
        let encoded = crate::iupac::get_encoded(tb);
        if encoded == INVALID_IUPAC {
            panic!("Target contains invalid IUPAC character: {:?}", tb as char);
        }
        let overhang = m.saturating_sub(idx + 1) as i32;
        let adjust_vec = i32x8::splat(overhang * extra_penalty_scaled);
        let peq_slice = unsafe { transposed.peqs.get_unchecked(encoded as usize) };
        let pos = idx as i32;

        unsafe {
            for block_idx in 0..vectors_in_block {
                let eq = *peq_slice.get_unchecked(block_idx);
                let adjusted = advance_column(
                    pv.get_unchecked_mut(block_idx),
                    mv.get_unchecked_mut(block_idx),
                    score.get_unchecked_mut(block_idx),
                    eq,
                    adjust_vec,
                    ctx,
                );

                #[cfg(not(feature = "latest_wide"))]
                let match_mask = all_ones ^ adjusted.cmp_gt(k_scaled_vec);
                #[cfg(feature = "latest_wide")]
                let match_mask = all_ones ^ adjusted.simd_gt(k_scaled_vec);

                let match_bits = match_mask.to_array();
                if match_bits.iter().any(|&b| b != 0) {
                    let adjusted_arr = adjusted.to_array();
                    let base = block_idx * SIMD_LANES;
                    let end = (base + SIMD_LANES).min(nq);

                    for (i, &score_scaled) in adjusted_arr[..(end - base)].iter().enumerate() {
                        if score_scaled <= k_scaled {
                            results.push(MatchInfo {
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

    if alpha_scaled < SCALE {
        for trailing in 1..=m {
            let adjust_vec = i32x8::splat((trailing as i32) * extra_penalty_scaled);
            let pos = (target.len() + trailing - 1) as i32;

            unsafe {
                for block_idx in 0..vectors_in_block {
                    let adjusted = advance_column(
                        pv.get_unchecked_mut(block_idx),
                        mv.get_unchecked_mut(block_idx),
                        score.get_unchecked_mut(block_idx),
                        zero_eq_vec,
                        adjust_vec,
                        ctx,
                    );

                    #[cfg(not(feature = "latest_wide"))]
                    let match_mask = all_ones ^ adjusted.cmp_gt(k_scaled_vec);
                    #[cfg(feature = "latest_wide")]
                    let match_mask = all_ones ^ adjusted.simd_gt(k_scaled_vec);

                    let match_bits = match_mask.to_array();
                    if match_bits.iter().any(|&b| b != 0) {
                        let adjusted_arr = adjusted.to_array();
                        let base = block_idx * SIMD_LANES;
                        let end = (base + SIMD_LANES).min(nq);

                        for (i, &score_scaled) in adjusted_arr[..(end - base)].iter().enumerate() {
                            if score_scaled <= k_scaled {
                                results.push(MatchInfo {
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
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iupac_query_matches_standard_target() {
        let queries = vec![b"n".to_vec()];
        let transposed = TQueries::new(&queries);
        assert_eq!(mini_search(&transposed, b"A", 0, None), vec![0.0]);
        assert_eq!(mini_search(&transposed, b"a", 0, None), vec![0.0]);
    }

    #[test]
    fn test_iupac_target_matches_standard_query() {
        let queries = vec![b"A".to_vec()];
        let transposed = TQueries::new(&queries);
        assert_eq!(mini_search(&transposed, b"N", 0, None), vec![0.0]);
        assert_eq!(mini_search(&transposed, b"n", 0, None), vec![0.0]);
    }

    #[test]
    fn test_iupac_mismatch_requires_edit() {
        let queries = vec![b"R".to_vec()];
        let transposed = TQueries::new(&queries);
        assert_eq!(mini_search(&transposed, b"C", 0, None), vec![-1.0]);
        assert_eq!(mini_search(&transposed, b"C", 1, None), vec![1.0]);
        assert_eq!(mini_search(&transposed, b"c", 1, None), vec![1.0]);
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
        let result = mini_search(&transposed, t, 4, None);
        assert_eq!(result, vec![0.0]); // Lowest edits, 0 for q at idx 0
    }

    #[test]
    fn test_double_search() {
        let q1 = b"ATG".to_vec();
        let q2 = b"TTG".to_vec();
        let queries = vec![q1, q2];
        let t = b"CCCTTGCCCCCCATGCCCCC";
        let transposed = TQueries::new(&queries);
        let result = mini_search(&transposed, t, 4, None);
        assert_eq!(result, vec![0.0, 0.0]);
    }

    #[test]
    fn test_edit() {
        let q1 = b"ATCAGA";
        let queries = vec![q1.to_vec()];
        let t = b"ATCTGA"; // 1 edit
        let transposed = TQueries::new(&queries);
        let result = mini_search(&transposed, t, 4, None);
        assert_eq!(result, vec![1.0]);
        let t = b"GTCTGA"; // 2 edits
        let result = mini_search(&transposed, t, 4, None);
        assert_eq!(result, vec![2.0]);
        let t = b"GTTGA"; // 3 edits (1 del)
        let result = mini_search(&transposed, t, 4, None);
        assert_eq!(result, vec![3.0]);
        // Match should not be recovered when k == 1
        let result = mini_search(&transposed, t, 1, None);
        assert_eq!(result, vec![-1.0]);
    }

    #[test]
    fn test_lowest_edits_returned() {
        let q1 = b"GGGCATCGATGAC";
        let queries = vec![q1.to_vec()];
        let t = b"CCCCCCCGGGCATCGATGACCCCCCCCCCCCCCCGGGCTTCGATGAC";
        //                                                     ^^^^^^^^^^^^^ has one mutation
        let transposed = TQueries::new(&queries);
        let result = mini_search(&transposed, t, 4, None);
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    #[should_panic(expected = "All queries must have the same length")]
    fn test_error_unequal_lengths() {
        let q1 = b"GGGCATCGATGAC";
        let q2 = b"AAA";
        let t = b"CCC";
        let queries = vec![q1.to_vec(), q2.to_vec()];
        let transposed = TQueries::new(&queries);
        let result = mini_search(&transposed, t, 4, None);
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    fn read_example() {
        let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
        let transposed = TQueries::new(&queries);
        let target = b"CCCTCGCCCCCCATGCCCCC";
        let result = mini_search(&transposed, target, 4, None);
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
        mini_search_with_positions(&transposed, t, 4, None, &mut result);

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
        let queries = vec![b"ATG".to_vec()];
        let t = b"ATGCCCATGCCCATG";
        //        012345678901234
        // ATG at positions 0-2 (ends at 2), 6-8 (ends at 8), 12-14 (ends at 14)
        let transposed = TQueries::new(&queries);
        let mut result = Vec::new();
        mini_search_with_positions(&transposed, t, 0, None, &mut result);

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
        let q1 = b"ATG".to_vec();
        let q2 = b"TTG".to_vec();
        let queries = vec![q1, q2];
        let t = b"CCCTTGCCCCCCATGCCCCC";
        let transposed = TQueries::new(&queries);

        let result_basic = mini_search(&transposed, t, 4, None);
        let mut result_pos = Vec::new();
        mini_search_with_positions(&transposed, t, 4, None, &mut result_pos);

        // mini_search returns minimum cost per query
        // mini_search_with_positions returns all matches
        // Check that the minimum cost in positions matches mini_search result
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
            assert!((result_basic[query_idx] - min_cost_in_positions).abs() < 1e-6);
        }
    }

    #[test]
    fn test_positions_no_match() {
        let q = b"AAAAA".to_vec();
        let queries = vec![q];
        let t = b"CCCCCCCCCCCCC";
        let transposed = TQueries::new(&queries);
        let mut result = Vec::new();
        mini_search_with_positions(&transposed, t, 1, None, &mut result);

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
        mini_search_with_positions(&transposed, t, 2, None, &mut result);

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
        let queries = vec![b"AC".to_vec()];
        let transposed = TQueries::new(&queries);
        let t = b"A";
        let result = mini_search(&transposed, t, 1, Some(0.5));
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
        let standard = mini_search(&transposed, t, 4, None);
        let overhang = mini_search(&transposed, t, 4, Some(1.0));
        assert_eq!(standard.len(), overhang.len());
        for (i, &val) in standard.iter().enumerate() {
            assert!((overhang[i] - val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_positions_overhang() {
        let queries = vec![b"AC".to_vec()];
        let transposed = TQueries::new(&queries);
        let t = b"A";
        let mut results = Vec::new();
        mini_search_with_positions(&transposed, t, 1, Some(0.5), &mut results);
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
        mini_search_with_positions(&transposed, t, 4, Some(0.5), &mut results);
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
        mini_search_with_positions(&transposed, t, 4, Some(0.5), &mut results);
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
        let queries = vec![b"AAACCC".to_vec()];
        let transposed = TQueries::new(&queries);
        let t = b"GGGGGAAA";
        let mut results = Vec::new();
        mini_search_with_positions(&transposed, t, 4, Some(0.5), &mut results);
        // Get minimum cost match
        let _min_cost = results
            .iter()
            .map(|m| m.cost)
            .fold(f32::INFINITY, |a, b| a.min(b));
        println!("results: {:?}", results);
    }
}
