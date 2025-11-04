use mini_myers::{TQueries, mini_search};

fn main() {
    let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
    let transposed = TQueries::new(&queries);
    let target = b"CCCTTGCCCCCCATGCCCCC";
    let result = mini_search(&transposed, target, 4);
    println!("Result: {:?}", result);
}
