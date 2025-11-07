[![crates.io](https://img.shields.io/crates/v/mini_myers.svg)](https://crates.io/crates/mini_myers)

### 🤏 Mini Myers
SIMD Myers implementation to "check" whether "batches" of short patterns (<=32nt) are present in a longer 
text with at most `k` edits, can report positions at 5% overhead.

Mainly for [Barbell](https://github.com/rickbeeloo/barbell) as a faster pre-filter (2x) before running [sassy](https://github.com/RagnarGrootKoerkamp/sassy/).
Most likely you are looking for [sassy](https://github.com/RagnarGrootKoerkamp/sassy/) instead, unless you want to batch search very short patterns.

---

### When to use
- Short queries (`<=32`)
- Search for multiple of `8` queries


--- 

#### What it does
We compare each character of the queries against a single text character at the time using SIMD.
Then we track the lowest cost 
we see along the entire text and report the cost when below the cut-off `k`, or `-1` if above `k` for 
`mini_search`, at a ~5% overhead we can also track the positions in `mini_search_positions`. 

--- 

#### How to use:


#### Without position tracking
This will just return the lowest edits found (below `k`) for each query. 


```rust
use mini_myers::{Searcher, Scan, Positions};
use mini_myers::backend::{U32, U64};

// Create a searcher for scan mode with U32 backend
// that is 8 queries in parallel (8*32)
let mut searcher = Searcher::<U32, Scan>::new();
let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
let encoded = searcher.encode(&queries);
let target = b"CCCTCGCCCCCCATGCCCCC";

// Scan mode: get minimum cost per query
let results = searcher.search(&encoded, target, 4, None);
assert_eq!(results, vec![0.0, 1.0]);

// Positions mode: get all match positions
let mut pos_searcher = Searcher::<U32, Positions>::new();
let encoded = pos_searcher.encode(&queries);
let results = pos_searcher.search(&encoded, target, 4, None);
println!("Found {} matches", results.len());

// Use U64 backend for longer queries (up to 64 nucleotides)
// that is 4 queries in parlalel (4*64)
let mut searcher64 = Searcher::<U64, Scan>::new();
let encoded = searcher64.encode(&queries);
let results = searcher64.search(&encoded, target, 4, None);
```


#### With "overhang" enabled
Like in [sassy](https://github.com/RagnarGrootKoerkamp/sassy/) `mini_myers` can apply 
a reduced penalty for characters "hanging over" the target sequence, i.e.:
```
    q: AAACCC
          |||
    t:    CCCGGGGGGGGGGGGGG
```
You can enable overhang by changing `None` in the above commands to `Some(overhang_cost)`, i.e. `Some(0.5)`:
```rust
let results = searcher64.search(&encoded, target, 4, Some(0.5));
```
In the example above, this would give a cost of `3 * 0.5 = 1.5`


---

#### Benchmark
Note that `mini_myers` and `sassy` are not directly comparable. 
`mini_seaerch` just tells us if something is present below `k`, and
`mini_search_with_positions` does return the match locations similar
to `sassy` *but* without traceback and local minima scan. Here we search for 192 sequences, 
a multiple of 8, which is ideal for `mini_myers` as well.

| Profile | Query Len | Target Len | µs/query (mini_myers) | µs/query (sassy) | Function                    |
|---------|-----------|------------|-----------------------|------------------|-----------------------------|
| IUPAC   |   24      |   1,000    |  0.23                 |   0.85           | `mini_search`               |
| IUPAC   |   24      |  10,000    |  2.29                 |   5.46           | `mini_search`               |
| IUPAC   |   24      |  50,000    | 11.91                 |  25.22           | `mini_search`               |
| IUPAC   |   24      | 100,000    | 24.21                 |  52.67           | `mini_search`               |
| IUPAC   |   32      | 100,000    | 23.9                  | 51.3             | `mini_search`               |
| IUPAC   |   24      |   1,000    |  0.43                 |   0.83           | `mini_search_with_positions`|
| IUPAC   |   24      |  10,000    |  3.02                 |   5.47           | `mini_search_with_positions`|
| IUPAC   |   24      |  50,000    | 15.29                 |  25.11           | `mini_search_with_positions`|
| IUPAC   |   24      |1,000,000   |299.53                 | 496.75           | `mini_search_with_positions`|


Run the bench using `cargo bench --bench sassy`, now has `mini_search_with_positions` but you can replace 
the call with `mini_search` to bench without positions. 



