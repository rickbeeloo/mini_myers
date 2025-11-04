#![feature(portable_simd)]
use core::simd::Simd;
use std::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use std::simd::{LaneCount, SupportedLaneCount};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct TQueries {
    pub vectors: Vec<Simd<u8, 32>>,
    pub query_length: usize,
    pub n_queries: usize,
    pub peq_a: Vec<u32>,
    pub peq_c: Vec<u32>,
    pub peq_g: Vec<u32>,
    pub peq_t: Vec<u32>,
}

impl TQueries {
    pub fn new(queries: &[Vec<u8>]) -> Self {
        assert!(!queries.is_empty(), "No queries provided");
        let query_length = queries[0].len();
        assert!(
            queries.iter().all(|q| q.len() == query_length),
            "All queries must have the same length"
        );

        let n_queries = queries.len();
        assert!(n_queries <= 32, "Number of queries exceeds 32");

        let mut vectors: Vec<Simd<u8, 32>> = Vec::with_capacity(query_length);

        for i in 0..query_length {
            let mut lane: [u8; 32] = [0u8; 32];
            for (q_idx, q) in queries.iter().enumerate() {
                lane[q_idx] = q[i];
            }
            vectors.push(Simd::from_array(lane));
        }

        // Precompute peq bitvectors (independent of LANES, computed once)
        let mut peq_a = vec![0u32; n_queries];
        let mut peq_c = vec![0u32; n_queries];
        let mut peq_g = vec![0u32; n_queries];
        let mut peq_t = vec![0u32; n_queries];

        for pos in 0..query_length {
            let qv_simd = vectors[pos];
            let bit = 1u32 << pos;

            let a_mask = qv_simd.simd_eq(Simd::<u8, 32>::splat(b'A')).to_bitmask();
            let c_mask = qv_simd.simd_eq(Simd::<u8, 32>::splat(b'C')).to_bitmask();
            let g_mask = qv_simd.simd_eq(Simd::<u8, 32>::splat(b'G')).to_bitmask();
            let t_mask = qv_simd.simd_eq(Simd::<u8, 32>::splat(b'T')).to_bitmask();

            for qi in 0..n_queries {
                let lane_bit = 1u64 << qi;
                if (a_mask & lane_bit) != 0 {
                    peq_a[qi] |= bit;
                }
                if (c_mask & lane_bit) != 0 {
                    peq_c[qi] |= bit;
                }
                if (g_mask & lane_bit) != 0 {
                    peq_g[qi] |= bit;
                }
                if (t_mask & lane_bit) != 0 {
                    peq_t[qi] |= bit;
                }
            }
        }

        Self {
            vectors,
            query_length,
            n_queries,
            peq_a,
            peq_c,
            peq_g,
            peq_t,
        }
    }
}

/// Dispatcher: uses optimal lane count based on query length
#[inline(always)]
pub fn mini_search(transposed: &TQueries, target: &[u8], k: u8) -> Vec<i32> {
    let nq = transposed.n_queries;
    if nq == 0 {
        return Vec::new();
    }
    let m = transposed.query_length;
    assert!(m > 0 && m <= 32, "query length must be 1..=32");

    // Use i32 throughout (like backup.rs), but with more lanes for short queries
    if m <= 16 {
        search_generic::<16>(transposed, target, k)
    } else {
        search_generic::<8>(transposed, target, k)
    }
}

/// Build peqs vectors from peq arrays
#[inline(always)]
fn build_peqs_vectors<const LANES: usize>(
    peq_a: &[u32],
    peq_c: &[u32],
    peq_g: &[u32],
    peq_t: &[u32],
    nq: usize,
    vectors_in_block: usize,
) -> (
    Vec<Simd<i32, LANES>>,
    Vec<Simd<i32, LANES>>,
    Vec<Simd<i32, LANES>>,
    Vec<Simd<i32, LANES>>,
)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut peqs_a: Vec<Simd<i32, LANES>> = Vec::with_capacity(vectors_in_block);
    let mut peqs_c: Vec<Simd<i32, LANES>> = Vec::with_capacity(vectors_in_block);
    let mut peqs_g: Vec<Simd<i32, LANES>> = Vec::with_capacity(vectors_in_block);
    let mut peqs_t: Vec<Simd<i32, LANES>> = Vec::with_capacity(vectors_in_block);

    for v in 0..vectors_in_block {
        let base = v * LANES;
        let mut lane_a = [0i32; LANES];
        let mut lane_c = [0i32; LANES];
        let mut lane_g = [0i32; LANES];
        let mut lane_t = [0i32; LANES];

        for l in 0..LANES {
            let qi = base + l;
            if qi < nq {
                lane_a[l] = peq_a[qi] as i32;
                lane_c[l] = peq_c[qi] as i32;
                lane_g[l] = peq_g[qi] as i32;
                lane_t[l] = peq_t[qi] as i32;
            }
        }

        peqs_a.push(Simd::from_array(lane_a));
        peqs_c.push(Simd::from_array(lane_c));
        peqs_g.push(Simd::from_array(lane_g));
        peqs_t.push(Simd::from_array(lane_t));
    }

    (peqs_a, peqs_c, peqs_g, peqs_t)
}

#[inline(always)]
fn search_generic<const LANES: usize>(transposed: &TQueries, target: &[u8], k: u8) -> Vec<i32>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    // todo: would be nice to also extract this to transposedQueries so we only
    // have to build the peq table once
    let nq = transposed.n_queries;
    let m = transposed.query_length;

    let peq_a = &transposed.peq_a;
    let peq_c = &transposed.peq_c;
    let peq_g = &transposed.peq_g;
    let peq_t = &transposed.peq_t;

    let vectors_in_block = nq.div_ceil(LANES);
    //println!("Using {} vectors in block", vectors_in_block);

    let (peqs_a, peqs_c, peqs_g, peqs_t) =
        build_peqs_vectors(peq_a, peq_c, peq_g, peq_t, nq, vectors_in_block);

    let peq_table: [&[Simd<i32, LANES>]; 4] = [
        peqs_a.as_slice(),
        peqs_c.as_slice(),
        peqs_t.as_slice(),
        peqs_g.as_slice(),
    ];

    let all_ones = Simd::<i32, LANES>::splat(!0i32);
    let zero_v = Simd::<i32, LANES>::splat(0i32);
    let one_v = Simd::<i32, LANES>::splat(1i32);

    let mut pv_vec: Vec<Simd<i32, LANES>> = vec![all_ones; vectors_in_block];
    let mut mv_vec: Vec<Simd<i32, LANES>> = vec![zero_v; vectors_in_block];
    let mut scores_vec: Vec<Simd<i32, LANES>> =
        vec![Simd::<i32, LANES>::splat(m as i32); vectors_in_block];

    let max_possible = (m + target.len()) as i32;
    let mut min_scores_vec: Vec<Simd<i32, LANES>> =
        vec![Simd::<i32, LANES>::splat(max_possible); vectors_in_block];

    let high_bit: u32 = 1u32 << (m - 1);
    let mask_vec = Simd::<i32, LANES>::splat(high_bit as i32);

    let target_len = target.len();
    for t_idx in 0..target_len {
        let tb = unsafe { target.get_unchecked(t_idx) };
        let idx = (tb >> 1) & 3;
        let peq_slice = unsafe { peq_table.get_unchecked(idx as usize) };
        for v in 0..vectors_in_block {
            // Regular Myers
            let eq = unsafe { peq_slice.get_unchecked(v) };
            let pv = pv_vec[v];
            let mv = mv_vec[v];
            let score = unsafe { scores_vec.get_unchecked(v) };
            let xv = eq | mv;
            let xh = (((eq & pv) + pv) ^ pv) | eq;
            let ph = mv | !(xh | pv);
            let mh = pv & xh;
            // Track edit distance cost
            let ph_bit = (ph & mask_vec).simd_ne(zero_v).to_int() & one_v;
            let mh_bit = (mh & mask_vec).simd_ne(zero_v).to_int() & one_v;
            let new_score = score + (ph_bit - mh_bit);
            let ph_shift = ph << 1;
            let new_pv = (mh << 1) | !(xv | ph_shift);
            let new_mv = ph_shift & xv;

            unsafe {
                *pv_vec.get_unchecked_mut(v) = new_pv;
                *mv_vec.get_unchecked_mut(v) = new_mv;
                *scores_vec.get_unchecked_mut(v) = new_score;
                *min_scores_vec.get_unchecked_mut(v) =
                    min_scores_vec.get_unchecked(v).simd_min(new_score);
            }
        }
    }
    let k_v = Simd::<i32, LANES>::splat(k as i32);
    let neg1_v = Simd::<i32, LANES>::splat(-1);
    let mut result = vec![-1i32; nq];

    for v in 0..vectors_in_block {
        let min_score = unsafe { *min_scores_vec.get_unchecked(v) };
        // we check for leq k, all positions got init to max edits as t.len() + m, which then become -1 here
        let selected = min_score.simd_le(k_v).select(min_score, neg1_v);
        let base = v * LANES;
        let end = (base + LANES).min(nq);
        result[base..end].copy_from_slice(&selected.to_array()[..end - base]);
    }
    //println!("Time taken: {:?}", end_time.duration_since(start_time));
    result
}

mod tests {
    use super::*;

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
}
