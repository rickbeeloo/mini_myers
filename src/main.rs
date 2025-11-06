use mini_myers::{mini_search, MyersSearchState, TQueries};

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
    use std::hint::black_box;
    let n_queries = 192;
    let query_len = 24;
    let mut queries = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        queries.push(generate_random_dna(query_len));
    }
    let transposed = TQueries::new(&queries);
    let target = generate_random_dna(10_000_000);
    //let mut results = Vec::new();
    let mut state = MyersSearchState::new();
    let results = mini_search(&mut state, &transposed, &target, 4, None);
    black_box(&results);
}
