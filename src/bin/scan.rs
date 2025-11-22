use clap::Parser;
use mini_myers::backend::{U16, U32, U64};
use mini_myers::search::Searcher;
use mini_myers::tqueries::TQueries;
use needletail::parse_fastx_file;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::PathBuf;

// Ugly but fine for testing :)
enum AnySearcher {
    U16(Searcher<U16>),
    U32(Searcher<U32>),
    U64(Searcher<U64>),
}

enum AnyTQueries {
    U16(TQueries<U16>),
    U32(TQueries<U32>),
    U64(TQueries<U64>),
}

impl AnySearcher {
    fn scan(&mut self, t_queries: &AnyTQueries, target: &[u8], k: u32) -> Vec<bool> {
        match (self, t_queries) {
            (AnySearcher::U16(searcher), AnyTQueries::U16(tq)) => {
                searcher.scan(tq, target, k).to_vec()
            }
            (AnySearcher::U32(searcher), AnyTQueries::U32(tq)) => {
                searcher.scan(tq, target, k).to_vec()
            }
            (AnySearcher::U64(searcher), AnyTQueries::U64(tq)) => {
                searcher.scan(tq, target, k).to_vec()
            }
            _ => unreachable!("Backend mismatch"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "scan")]
#[command(about = "Search for queries in a target sequence using SIMD Myers algorithm")]
struct Args {
    /// Path to query FASTA file
    #[arg(short, long)]
    query: PathBuf,

    /// Path to target FASTA file
    #[arg(short, long)]
    target: PathBuf,

    /// Maximum edit distance (k)
    #[arg(short, long)]
    k: u32,

    /// Reduced edit cost for overhang
    #[arg(short, long, default_value_t = 1.0)]
    alpha: f32,

    /// Output file
    /// if provided, will write the results to a file
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn read_fasta_queries(path: &PathBuf) -> (Vec<Vec<u8>>, Vec<String>) {
    let mut sequences = Vec::new();
    let mut headers = Vec::new();
    let mut reader = parse_fastx_file(path).expect("valid path/file");
    while let Some(record) = reader.next() {
        let seqrec = record.expect("invalid record");
        sequences.push(seqrec.seq().to_vec());
        headers.push(String::from_utf8_lossy(seqrec.id()).to_string());
    }
    (sequences, headers)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Read queries from FASTA file
    let (queries, headers) = read_fasta_queries(&args.query);
    if queries.is_empty() {
        return Err("No queries found in query FASTA file".into());
    }

    let max_len = queries.iter().map(|q| q.len()).max().unwrap();

    // Select backend based on query length
    let (mut searcher, t_queries) = if max_len <= 16 {
        (
            AnySearcher::U16(Searcher::<U16>::new(Some(args.alpha))),
            AnyTQueries::U16(TQueries::<U16>::new(&queries, false)),
        )
    } else if max_len <= 32 {
        (
            AnySearcher::U32(Searcher::<U32>::new(Some(args.alpha))),
            AnyTQueries::U32(TQueries::<U32>::new(&queries, false)),
        )
    } else if max_len <= 64 {
        (
            AnySearcher::U64(Searcher::<U64>::new(Some(args.alpha))),
            AnyTQueries::U64(TQueries::<U64>::new(&queries, false)),
        )
    } else {
        return Err(format!(
            "Query length {} is too long. Maximum supported length is 64 nucleotides",
            max_len
        )
        .into());
    };

    let mut reader = parse_fastx_file(&args.target).expect("valid path/file");
    let mut buf_writer =
        BufWriter::new(File::create(args.output.unwrap()).expect("valid path/file"));
    while let Some(record) = reader.next() {
        let seqrec = record.expect("invalid record");
        let target = seqrec.seq().to_vec();
        let target_id = String::from_utf8_lossy(seqrec.id()).to_string();
        let results = searcher.scan(&t_queries, &target, args.k);
        let true_indices = results
            .iter()
            .enumerate()
            .filter(|(_, &x)| x)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        // We just write "if" a match is found
        for index in true_indices {
            writeln!(buf_writer, "{}\t{}", target_id, headers[index])?;
        }
    }

    Ok(())
}
