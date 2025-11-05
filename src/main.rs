use mini_myers::{TQueries, mini_search};

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
    let target = generate_random_dna(1000000);
    let queries = vec![
        generate_random_dna(24),
        generate_random_dna(24),
        generate_random_dna(24),
    ];
    for query in queries {
        let transposed = TQueries::new(&[query]);
        let result = mini_search(&transposed, &target, 4);
        println!("Result: {:?}", result);
    }
}
