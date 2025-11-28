mod edlib_bench;

use edlib_bench::{get_edlib_config, run_edlib, sim_data::Alphabet};
use mini_myers::backend::{SimdBackend, U16, U32};
use mini_myers::search::Searcher as MiniSearcher;
use mini_myers::TQueries;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use sassy::profiles::Iupac;
use sassy::Searcher as SassySearcher;
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

    let target_lens = vec![16, 32, 100, 1000, 10_000, 100_000];
    let query_lens = vec![16, 32];
    let ks = vec![1, 4];
    let iterations = 100;
    let n_queries = 96;

    for target_len in &target_lens {
        for query_len in &query_lens {
            for k in &ks {
                let (mini_search, sassy, edlib) =
                    run_bench_round(&mut rng, *target_len, *query_len, iterations, *k, n_queries);

                println!(
                    "T={:<7} Q={:<2} K={} | \
                    Search: {:>7.3}ms ({:>7.3}µs/q) | \
                    Sassy:  {:>7.3}ms ({:>7.3}µs/q) | \
                    Edlib:  {:>7.3}ms ({:>7.3}µs/q)",
                    target_len,
                    query_len,
                    k,
                    mini_search.avg_batch_ms(),
                    mini_search.avg_per_query_us(),
                    sassy.avg_batch_ms(),
                    sassy.avg_per_query_us(),
                    edlib.avg_batch_ms(),
                    edlib.avg_per_query_us(),
                );

                results.push(mini_search);
                results.push(sassy);
                results.push(edlib);
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
    let target = generate_random_dna(rng, target_len);
    let mut queries = Vec::new();
    for _ in 0..n_queries {
        queries.push(generate_random_dna(rng, query_len));
    }

    let mut sassy_searcher = SassySearcher::<Iupac>::new_rc();

    // Setup edlib config
    let edlib_config = get_edlib_config(k as i32, &Alphabet::Dna);

    let mut sassy_matches = 0;
    for q in &queries {
        sassy_matches += sassy_searcher.search_all(q, &target, k as usize).len();
    }

    let (mini_search_matches, mini_search_time) = if query_len == 16 {
        run_mini_bench::<U16>(&queries, &target, k, iterations)
    } else {
        run_mini_bench::<U32>(&queries, &target, k, iterations)
    };

    // Edlib Count - count matches by checking edit distance
    let mut edlib_matches = 0;
    for q in &queries {
        let result = run_edlib(q, &target, &edlib_config);
        if result.editDistance >= 0 && result.editDistance <= k as i32 {
            edlib_matches += 1;
        }
    }

    // Mini Scan Count
    // let mini_scan_res = searcher_u32.trace_all_hits(&t_queries_u32, &target, k as u32);
    // let mini_scan_matches: usize = mini_scan_res.iter().filter(|&&b| b).count();

    // Time Sassy
    let sassy_time = time_iterations(iterations, || {
        for q in &queries {
            let m = sassy_searcher.search(q, &target, k as usize);
            black_box(m);
        }
    });

    // Time Edlib
    let edlib_time = time_iterations(iterations, || {
        for q in &queries {
            let result = run_edlib(q, &target, &edlib_config);
            black_box(result);
        }
    });

    // // Time Mini Scan
    // let mini_scan_time = time_iterations(iterations, || {
    //     let m = searcher_u32.scan(&t_queries_u32, &target, k as u32);
    //     black_box(m);
    // });

    let queries_per_iter = queries.len();

    let res_search = BenchResult {
        tool: "mini_search",
        target_len,
        query_len,
        k,
        iterations,
        queries_per_iter,
        num_matches: mini_search_matches,
        avg_batch_ns: average_per_batch(mini_search_time, iterations),
        avg_per_query_ns: average_per_query(mini_search_time, iterations, queries_per_iter),
    };

    // let res_scan = BenchResult {
    //     tool: "mini_scan",
    //     target_len,
    //     query_len,
    //     k,
    //     iterations,
    //     queries_per_iter,
    //     num_matches: mini_scan_matches,
    //     avg_batch_ns: average_per_batch(mini_scan_time, iterations),
    //     avg_per_query_ns: average_per_query(mini_scan_time, iterations, queries_per_iter),
    // };

    let res_sassy = BenchResult {
        tool: "sassy",
        target_len,
        query_len,
        k,
        iterations,
        queries_per_iter,
        num_matches: sassy_matches,
        avg_batch_ns: average_per_batch(sassy_time, iterations),
        avg_per_query_ns: average_per_query(sassy_time, iterations, queries_per_iter),
    };

    let res_edlib = BenchResult {
        tool: "edlib",
        target_len,
        query_len,
        k,
        iterations,
        queries_per_iter,
        num_matches: edlib_matches,
        avg_batch_ns: average_per_batch(edlib_time, iterations),
        avg_per_query_ns: average_per_query(edlib_time, iterations, queries_per_iter),
    };

    (res_search, res_sassy, res_edlib)
}

fn run_mini_bench<B: SimdBackend>(
    queries: &[Vec<u8>],
    target: &[u8],
    k: u8,
    iterations: usize,
) -> (usize, Duration) {
    let t_queries = TQueries::<B>::new(queries, true);
    let mut searcher = MiniSearcher::<B>::new(None);

    let mini_search_res = searcher.trace_all_hits(&t_queries, target, k as u32);
    let mini_search_matches: usize = mini_search_res.len();

    let mini_search_time = time_iterations(iterations, || {
        let m = searcher.trace_all_hits(&t_queries, target, k as u32);
        black_box(m);
    });

    (mini_search_matches, mini_search_time)
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

#[allow(dead_code)]
fn insert_query_matches(rng: &mut StdRng, target: &mut Vec<u8>, queries: &[Vec<u8>], k: u8) {
    let bases = [b'A', b'T', b'G', b'C'];

    for query in queries {
        let num_matches = rng.gen_range(1..=4);
        for _ in 0..num_matches {
            let pos = rng.gen_range(0..=target.len());
            let use_exact = rng.gen_bool(0.7);

            let match_seq = if use_exact {
                query.clone()
            } else {
                let max_edits = (k / 2).max(1) as usize;
                let num_edits = rng.gen_range(1..=max_edits.min(query.len()));
                let mut variant = query.clone();
                let mut positions: Vec<usize> = (0..query.len()).collect();
                positions.shuffle(rng);

                for &pos_idx in positions.iter().take(num_edits) {
                    let current_base = variant[pos_idx];
                    let mut new_base = bases[rng.gen_range(0..bases.len())];
                    while new_base == current_base {
                        new_base = bases[rng.gen_range(0..bases.len())];
                    }
                    variant[pos_idx] = new_base;
                }
                variant
            };

            // If pos is at end, just push, otherwise splice
            if pos >= target.len() {
                target.extend_from_slice(&match_seq);
            } else {
                target.splice(pos..pos, match_seq.iter().copied());
            }
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
