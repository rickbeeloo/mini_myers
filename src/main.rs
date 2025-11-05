use mini_myers::{mini_search, TQueries};

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
    use mini_myers::{mini_search_with_positions, TQueries};

    let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
    let transposed = TQueries::new(&queries);
    let target = b"CCCTCGCCCCCCATGCCCCC";
    let mut results = Vec::new();
    let result = mini_search_with_positions(&transposed, target, 1, &mut results);
    println!("Result: {:?}", results);
}
