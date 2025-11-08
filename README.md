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


#### Basic usage
This will just return the lowest edits found (below `k`) for each query. 


```rust
use mini_myers::{Searcher, Scan, Positions};
use mini_myers::backend::{U32, U64};

// Create a searcher for scan mode with U32 backend
// that is 8 queries in parallel (8*32)
let mut searcher = Searcher::<U32, Scan>::new();
let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
let encoded = searcher.encode(&queries, false); //true = also search rc
let target = b"CCCTCGCCCCCCATGCCCCC";

// Scan mode: get minimum cost per query
let results = searcher.search(&encoded, target, 4, None);
assert_eq!(results, vec![0.0, 1.0]);

// Positions mode: get all match positions
let mut pos_searcher = Searcher::<U32, Positions>::new();
let encoded = pos_searcher.encode(&queries, false);
let results = pos_searcher.search(&encoded, target, 4, None);
println!("Found {} matches", results.len());

// Use U64 backend for longer queries (up to 64 nucleotides)
// that is 4 queries in parlalel (4*64)
let mut searcher64 = Searcher::<U64, Scan>::new();
let encoded = searcher64.encode(&queries, false);
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

| Query length | Target length | Sassy (µs/query) | Mini search (µs/query) | Mini search (pos) (µs/query) | Speedup (Pos × Sassy)[^1] |
|--------------|--------------|------------------|------------------------|------------------------------|-------------------------------|
| 32           | 50           |   0.9763         |   0.0312               |   0.0359                     | 27.2×                         |
| 32           | 100          |   1.1464         |   0.0506               |   0.1240                     | 9.2×                          |
| 32           | 1_000         |   2.2656         |   0.5051               |   0.6086                     | 3.7×                          |
| 32           | 10_000        |  14.5188         |   4.8694               |   5.9998                     | 2.4×                          |
| 32           | 50_000        |  67.0511         |  23.8459               |  30.3187                     | 2.2×                          |

[^1]: Sassy also does traceback which costs time

Run the bench using `cargo bench --bench sassy`, now has `mini_search_with_positions` but you can replace 
the call with `mini_search` to bench without positions. 



