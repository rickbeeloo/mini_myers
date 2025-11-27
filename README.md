### 🤏 Mini Myers
SIMD Myers implementation to "check" whether "batches" of short patterns (<=64nt) are present in a longer 
text with at most `k` edits.

Mainly for [Barbell](https://github.com/rickbeeloo/barbell) as a faster pre-filter (2x) before running [sassy](https://github.com/RagnarGrootKoerkamp/sassy/).
Most likely you are looking for [sassy](https://github.com/RagnarGrootKoerkamp/sassy/) instead, unless you want to batch search short patterns.

---

### When to use
- Short queries (`<=16` for U16 backend, `<=32` for U32 backend, `<=64` for U64 backend)
- Search for multiple queries (at least 8 or so)

--- 

### How to use

#### Basic search (find hits)

This will return `SearchHit` results indicating where each query matches (within `k` edits).

```rust
use mini_myers::{Searcher, TQueries};
use mini_myers::backend::U32;

let mut searcher = Searcher::<U32>::new(None);
let queries = vec![b"ACGT".to_vec(), b"TGCA".to_vec()];
let t_queries = TQueries::<U32>::new(&queries, false); // false = only forward, true = forward + reverse complement
let text = b"AAACGTTTGCAAA";
let k = 0; // exact match only

let hits = searcher.search_with_hits(&t_queries, text, k);

// hits contains SearchHit { query_idx, end_position }
assert_eq!(hits.len(), 2);
assert_eq!(hits[0].query_idx, 0);
assert_eq!(hits[0].end_position, 5); // ACGT ends at position 5
```

#### Full alignment with traceback

This will return full `Alignment` results with CIGAR strings, edit counts, and start/end positions.

```rust
use mini_myers::{Searcher, TQueries};
use mini_myers::backend::U32;

let k = 2;
let mut searcher = Searcher::<U32>::new(None);
let queries = vec![b"ACGT".to_vec()];
let t_queries = TQueries::<U32>::new(&queries, false);
let text = b"AAATGTAAA"; // ATGT has 1 substitution from ACGT

let alignments = searcher.trace_all_hits(&t_queries, text, k);

for aln in alignments {
    println!(
        "query_idx: {}, edits: {}, cigar: {}, start: {}, end: {}",
        aln.query_idx,
        aln.edits,
        aln.operations.to_string(),
        aln.start,
        aln.end
    );
}
```

#### With overhang support

You can enable suffix/prefix overhang matching by providing an `alpha` parameter (cost per overhang character):

```rust
use mini_myers::{Searcher, TQueries};
use mini_myers::backend::U32;

// alpha=0.5 means each overhang character costs 0.5 edits
let mut searcher = Searcher::<U32>::new(Some(0.5));
let queries = vec![b"ACGT".to_vec()];
let t_queries = TQueries::<U32>::new(&queries, false);
let text = b"AC"; // Query can match with suffix overhang (GT)
let k = 2;

let alignments = searcher.trace_all_hits(&t_queries, text, k);
```


---

#### Little bench
Comparing mini search (orange) against [sassy](https://github.com/RagnarGrootKoerkamp/sassy.git)(green) `search_all`, 
and [edlib](https://github.com/Martinsos/edlib)(purple) for 96 queries of length [16, 32] at k=[1,4] in a range of targets (x-axis)

<img src="test_data/bench.png" alt="mini_scan vs sassy benchmark plot" width="800"/>


Run the same bench using `cargo bench --bench sassy`. 


#### Some dev stuff
see `justfile`, to get assembly for `scan`, disable inline `#[inline(never)]` and run `just search_asm`. To enable all permissions for flamegraph/perf `just perm`.