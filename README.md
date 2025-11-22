[![crates.io](https://img.shields.io/crates/v/mini_myers.svg)](https://crates.io/crates/mini_myers)

### 🤏 Mini Myers
SIMD Myers implementation to "check" whether "batches" of short patterns (<=32nt) are present in a longer 
text with at most `k` edits.

Mainly for [Barbell](https://github.com/rickbeeloo/barbell) as a faster pre-filter (2x) before running [sassy](https://github.com/RagnarGrootKoerkamp/sassy/).
Most likely you are looking for [sassy](https://github.com/RagnarGrootKoerkamp/sassy/) instead, unless you want to batch search very short patterns.

---

### When to use
- Short queries (`<=32`)
- Search for multiple of `8` queries


--- 

#### What it does
We compare each character of the queries against a single text character at the time using SIMD.
Then we track the edit distance along the entire text and report whether each query matches (within `k` edits). 

--- 

#### How to use:


#### Basic usage
This will return boolean results indicating whether each query matches (within `k` edits). 


```rust
use mini_myers::{Searcher, TQueries};
use mini_myers::backend::{U32, U64};

// Create a searcher with U32 backend
// that is 8 queries in parallel (8*32)
let mut searcher = Searcher::<U32>::new();
let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
let encoded = TQueries::<U32>::new(&queries, false); //true = also search rc
let target = b"CCCTCGCCCCCCATGCCCCC";

// Scan mode: get boolean results indicating if each query matches (within k edits)
let results = searcher.scan(&encoded, target, 4, None);
assert!(results[0]); // "ATG" matches
assert!(!results[1]); // "TTG" doesn't match

// Use U64 backend for longer queries (up to 64 nucleotides)
// that is 4 queries in parallel (4*64)
let mut searcher64 = Searcher::<U64>::new();
let encoded = TQueries::<U64>::new(&queries, false);
let results = searcher64.scan(&encoded, target, 4, None);
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
let results = searcher64.scan(&encoded, target, 4, Some(0.5));
```
In the example above, this would give a cost of `3 * 0.5 = 1.5`


---

#### Benchmark
Note that `mini_myers` and `sassy` are not directly comparable. 
The `mini_myers` "scan" mode just returns whether a match is present `<=k`, whereas sassy 
finds the positions and performs traceback (of course much more compute).

| Query length | Target length | Mini scan (µs/query) | Sassy (µs/query) | Speedup (Mini × Sassy) |
|--------------|--------------|----------------------|------------------|------------------------|
| 16           | 32           |    0.0077            |   0.6410         | 83×                    |
| 16           | 64           |    0.0148            |   0.6760         | 46×                    |
| 16           | 100          |    0.0231            |   0.7857         | 34×                    |
| 16           | 1,000        |    0.2228            |   1.9672         | 9×                     |
| 16           | 10,000       |    2.2677            |  15.3632         | 6.8×                   |
| 16           | 100,000      |   23.3472            | 143.1886         | 6.1×                   |



Run the bench using `cargo bench --bench sassy`. 
