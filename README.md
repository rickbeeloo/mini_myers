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
The `mini_myers` "scan" mode just returns whether a match is present `<=k`, whereas sassy 
finds the positions and preforms traceback (of course much more compute).

| Query length | Target length | Sassy (µs/query) | Mini scan (µs/query) | Speedup (Mini × Sassy) |
|--------------|--------------|------------------|----------------------|------------------------|
| 32           | 32           |   11.52          |   0.52               | 22.1×                  |
| 32           | 64           |   10.84          |   0.50               | 21.7×                  |
| 32           | 100          |   10.53          |   0.47               | 22.4×                  |
| 32           | 1,000        |   12.20          |   0.57               | 21.4×                  |
| 32           | 10,000       |   21.16          |   1.16               | 18.3×                  |
| 32           | 100,000      |  113.74          |   6.61               | 17.2×                  |

Run the bench using `cargo bench --bench sassy`. 
