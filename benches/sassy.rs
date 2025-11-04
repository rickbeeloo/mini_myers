use mini_myers::{TQueries, mini_search};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sassy::Searcher;
use sassy::profiles::Dna;
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

    let target_lens = vec![30_000];
    let query_lens = vec![24, 32];
    let ks = vec![4];
    let iterations = 100;
    let n_queries = 32;

    for target_len in target_lens {
        for query_len in &query_lens {
            for k in &ks {
                let (mini, sassy) =
                    run_bench_round(&mut rng, target_len, *query_len, iterations, *k, n_queries);

                println!(
                    "target={:<7} query={:<2} k={} | mini_myers: {:>8.4} ms/batch ({:>8.4} µs/query), sassy: {:>8.4} ms/batch ({:>8.4} µs/query)",
                    target_len,
                    query_len,
                    k,
                    mini.avg_batch_ms(),
                    mini.avg_per_query_us(),
                    sassy.avg_batch_ms(),
                    sassy.avg_per_query_us()
                );

                results.push(mini);
                results.push(sassy);
            }
        }
    }

    let output_dir = Path::new("target/bench_results");
    fs::create_dir_all(output_dir)?;
    let output_path = output_dir.join("sassy_vs_mini.csv");
    write_results(&results, &output_path)?;
    println!("Saved to {}", output_path.display());

    Ok(())
}

fn run_bench_round(
    rng: &mut StdRng,
    target_len: usize,
    query_len: usize,
    iterations: usize,
    k: u8,
    n_queries: usize,
) -> (BenchResult, BenchResult) {
    let target = generate_random_dna(rng, target_len);
    let mut queries = Vec::new();
    for _ in 0..n_queries {
        queries.push(generate_random_dna(rng, query_len));
    }

    let mut searcher = Searcher::<Dna>::new_fwd();
    let transposed = TQueries::new(&queries);

    let sassy_total = time_iterations(iterations, || {
        for q in &queries {
            let matches = searcher.search(q, &target, k as usize);
            black_box(&matches);
        }
    });

    let mini_total = time_iterations(iterations, || {
        let result = mini_search(&transposed, &target, k);
        black_box(&result);
    });

    let queries_per_iter = queries.len();

    let sassy_result = BenchResult {
        tool: "sassy",
        target_len,
        query_len,
        k,
        iterations,
        queries_per_iter,
        avg_batch_ns: average_per_batch(sassy_total, iterations),
        avg_per_query_ns: average_per_query(sassy_total, iterations, queries_per_iter),
    };

    let mini_result = BenchResult {
        tool: "mini_myers",
        target_len,
        query_len,
        k,
        iterations,
        queries_per_iter,
        avg_batch_ns: average_per_batch(mini_total, iterations),
        avg_per_query_ns: average_per_query(mini_total, iterations, queries_per_iter),
    };

    (mini_result, sassy_result)
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
        "tool,target_len,query_len,k,iterations,queries_per_iter,avg_batch_ns,avg_batch_ms,avg_per_query_ns,avg_per_query_us"
    )?;

    for result in results {
        writeln!(
            file,
            "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6}",
            result.tool,
            result.target_len,
            result.query_len,
            result.k,
            result.iterations,
            result.queries_per_iter,
            result.avg_batch_ns,
            result.avg_batch_ms(),
            result.avg_per_query_ns,
            result.avg_per_query_us()
        )?;
    }

    Ok(())
}
