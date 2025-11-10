use mini_myers::backend::U32;
use mini_myers::Searcher;

#[allow(dead_code)]
fn generate_random_dna(l: usize) -> Vec<u8> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    const DNA: &[u8; 4] = b"ACGT";
    let mut rng = StdRng::seed_from_u64(42);
    let mut dna = Vec::with_capacity(l);
    for _ in 0..l {
        let idx = rng.gen_range(0..4);
        dna.push(DNA[idx]);
    }
    dna
}

fn main() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::hint::black_box;

    let n_queries = 192;
    let query_len = 24;
    let mut queries = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        queries.push(generate_random_dna(query_len));
    }
    let mut searcher = Searcher::<U32>::new();
    let encoded = searcher.encode(&queries, false);
    let mut target = generate_random_dna(10_000_000);

    // Randomly insert 10 query matches into the target
    let mut rng = StdRng::seed_from_u64(123); // Different seed for insertion positions
                                              // let n_insertions = 10;
                                              // for _ in 0..n_insertions {
                                              //     // Randomly select a query
                                              //     let query_idx = rng.gen_range(0..n_queries);
                                              //     let query = &queries[query_idx];

    //     // Randomly select a position where the query can fit
    //     let max_pos = target.len() - query_len;
    //     let insert_pos = rng.gen_range(0..=max_pos);

    //     // Insert the query at the selected position
    //     target[insert_pos..insert_pos + query_len].copy_from_slice(query);
    // }

    let results = searcher.search(&encoded, &target, 1, None);
    println!("number of matches: {}", results.len());
    black_box(&results);
}
