use mini_myers::backend::U32;
use mini_myers::search::Searcher as mini_searcher;
use mini_myers::TQueries;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use sassy::profiles::Iupac;
use sassy::Searcher;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

struct BenchResult {
    tool: &'static str,
    target_len: usize,
    query_len: usize,
    k: u8,
    iterations: usize,
    queries_per_iter: usize,
    num_matches: usize,
    avg_batch_ns: f64,
    avg_per_query_ns: f64,
}

impl BenchResult {
    fn avg_batch_ms(&self) -> f64 {
        self.avg_batch_ns / 1_000_000.0
    }

    fn avg_per_query_us(&self) -> f64 {
        self.avg_per_query_ns / 1_000.0
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("benchmark failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut results = Vec::new();

    let target_lens = vec![32, 64, 100, 1000, 10_000, 100_000];
    let query_lens = vec![32];
    let ks = vec![4];
    let iterations = 100;
    let n_queries = 96;

    // Regular search benchmarks
    for target_len in &target_lens {
        for query_len in &query_lens {
            for k in &ks {
                let (mini_search_result, mini_pos_result, sassy_result) =
                    run_bench_round(&mut rng, *target_len, *query_len, iterations, *k, n_queries);

                println!(
                    "target={:<7} query={:<2} k={} | mini_search: {:>8.4} ms/batch ({:>8.4} µs/query, {} matches), mini_search_with_positions: {:>8.4} ms/batch ({:>8.4} µs/query, {} matches), sassy: {:>8.4} ms/batch ({:>8.4} µs/query, {} matches)",
                    target_len,
                    query_len,
                    k,
                    mini_search_result.avg_batch_ms(),
                    mini_search_result.avg_per_query_us(),
                    mini_search_result.num_matches,
                    mini_pos_result.avg_batch_ms(),
                    mini_pos_result.avg_per_query_us(),
                    mini_pos_result.num_matches,
                    sassy_result.avg_batch_ms(),
                    sassy_result.avg_per_query_us(),
                    sassy_result.num_matches
                );

                results.push(mini_search_result);
                results.push(mini_pos_result);
                results.push(sassy_result);
            }
        }
    }

    let output_dir = Path::new("target/bench_results");
    fs::create_dir_all(output_dir)?;
    let output_path = output_dir.join("sassy_vs_mini.csv");
    write_results(&results, &output_path)?;
    println!("\nSaved to {}", output_path.display());

    Ok(())
}

fn run_bench_round(
    rng: &mut StdRng,
    target_len: usize,
    query_len: usize,
    iterations: usize,
    k: u8,
    n_queries: usize,
) -> (BenchResult, BenchResult, BenchResult) {
    let mut target = generate_random_dna(rng, target_len);
    let mut queries = Vec::new();
    for _ in 0..n_queries {
        queries.push(generate_random_dna(rng, query_len));
    }

    // Insert 1-4 matches of each query into the target
    insert_query_matches(rng, &mut target, &queries, k);

    let mut searcher = Searcher::<Iupac>::new_rc();

    // Create TQueries from the queries
    let t_queries = TQueries::<U32>::new(&queries, false);
    let mut mini_searcher = mini_searcher::<U32>::new(None);

    // Count matches for sassy (run once before timing)
    let mut sassy_match_count = 0;
    for q in &queries {
        let matches = searcher.search(q, &target, k as usize);
        sassy_match_count += matches.len();
    }

    // Count matches for mini_search (run once before timing)
    let mini_search_result_sample = mini_searcher.scan(&t_queries, &target, k as u32);
    let mini_search_match_count = mini_search_result_sample.iter().filter(|&&x| x).count();

    // Count matches for mini_search_with_positions (run once before timing)
    let mini_pos_result_sample = mini_searcher.scan(&t_queries, &target, k as u32);
    let mini_pos_match_count = mini_pos_result_sample.iter().filter(|&&x| x).count();

    // Benchmark sassy
    let sassy_total = time_iterations(iterations, || {
        for q in &queries {
            let matches = searcher.search(q, &target, k as usize);
            black_box(&matches);
        }
    });

    // Benchmark mini_search (same as with positions - new API only has search)
    let mini_search_total = time_iterations(iterations, || {
        let result = mini_searcher.scan(&t_queries, &target, k as u32);
        black_box(result);
    });

    // Benchmark mini_search_with_positions (same as above in new API)
    let mini_pos_total = time_iterations(iterations, || {
        let result = mini_searcher.scan(&t_queries, &target, k as u32);
        black_box(result);
    });

    let queries_per_iter = queries.len();

    let mini_search_result = BenchResult {
        tool: "mini_search",
        target_len,
        query_len,
        k,
        iterations,
        queries_per_iter,
        num_matches: mini_search_match_count,
        avg_batch_ns: average_per_batch(mini_search_total, iterations),
        avg_per_query_ns: average_per_query(mini_search_total, iterations, queries_per_iter),
    };

    let mini_pos_result = BenchResult {
        tool: "mini_search_with_positions",
        target_len,
        query_len,
        k,
        iterations,
        queries_per_iter,
        num_matches: mini_pos_match_count,
        avg_batch_ns: average_per_batch(mini_pos_total, iterations),
        avg_per_query_ns: average_per_query(mini_pos_total, iterations, queries_per_iter),
    };

    let sassy_result = BenchResult {
        tool: "sassy",
        target_len,
        query_len,
        k,
        iterations,
        queries_per_iter,
        num_matches: sassy_match_count,
        avg_batch_ns: average_per_batch(sassy_total, iterations),
        avg_per_query_ns: average_per_query(sassy_total, iterations, queries_per_iter),
    };

    (mini_search_result, mini_pos_result, sassy_result)
}

fn generate_random_dna(rng: &mut StdRng, len: usize) -> Vec<u8> {
    let bases = [b'A', b'T', b'G', b'C'];
    let mut dna = Vec::with_capacity(len);
    for _ in 0..len {
        let idx = rng.gen_range(0..bases.len());
        dna.push(bases[idx]);
    }
    dna
}

fn insert_query_matches(rng: &mut StdRng, target: &mut Vec<u8>, queries: &[Vec<u8>], k: u8) {
    let bases = [b'A', b'T', b'G', b'C'];

    for query in queries {
        // Randomly choose 1-4 matches to insert for this query
        let num_matches = rng.gen_range(1..=4);

        for _ in 0..num_matches {
            // Randomly choose position in target to insert
            let pos = rng.gen_range(0..=target.len());

            // Decide whether to insert exact match or match with edits (up to k/2 edits)
            let use_exact = rng.gen_bool(0.7); // 70% exact matches, 30% with edits

            let mut match_seq = if use_exact {
                query.clone()
            } else {
                // Create a variant with some edits (substitutions only, up to k/2)
                let max_edits = (k / 2).max(1) as usize;
                let num_edits = rng.gen_range(1..=max_edits.min(query.len()));
                let mut variant = query.clone();

                // Randomly select positions to mutate
                let mut positions: Vec<usize> = (0..query.len()).collect();
                positions.shuffle(rng);

                for &pos_idx in positions.iter().take(num_edits) {
                    // Change to a different base
                    let current_base = variant[pos_idx];
                    let mut new_base = bases[rng.gen_range(0..bases.len())];
                    while new_base == current_base {
                        new_base = bases[rng.gen_range(0..bases.len())];
                    }
                    variant[pos_idx] = new_base;
                }
                variant
            };

            // Insert the match sequence at the chosen position
            target.splice(pos..pos, match_seq.iter().copied());
        }
    }
}

fn time_iterations<F>(iterations: usize, mut f: F) -> Duration
where
    F: FnMut(),
{
    let mut total = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        total += start.elapsed();
    }
    total
}

fn average_per_batch(total: Duration, iterations: usize) -> f64 {
    total.as_secs_f64() * 1e9 / iterations as f64
}

fn average_per_query(total: Duration, iterations: usize, queries_per_iter: usize) -> f64 {
    total.as_secs_f64() * 1e9 / (iterations as f64 * queries_per_iter as f64)
}

fn write_results(results: &[BenchResult], output_path: &Path) -> std::io::Result<()> {
    let mut file = File::create(output_path)?;
    writeln!(
        file,
        "tool,target_len,query_len,k,iterations,queries_per_iter,num_matches,avg_batch_ns,avg_batch_ms,avg_per_query_ns,avg_per_query_us"
    )?;

    for result in results {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6}",
            result.tool,
            result.target_len,
            result.query_len,
            result.k,
            result.iterations,
            result.queries_per_iter,
            result.num_matches,
            result.avg_batch_ns,
            result.avg_batch_ms(),
            result.avg_per_query_ns,
            result.avg_per_query_us()
        )?;
    }

    Ok(())
}
