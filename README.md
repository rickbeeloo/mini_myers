[![crates.io](https://img.shields.io/crates/v/mini_myers.svg)](https://crates.io/crates/mini_myers)

### 🤏 Mini Myers
SIMD Myers implementation to "check" whether "batches" of short patterns (<=32nt) are present in a longer 
text with at most `k` edits. 

Mainly for [Barbell](https://github.com/rickbeeloo/barbell) as a faster pre-filter before running [sassy](https://github.com/RagnarGrootKoerkamp/sassy/),
so most likely you are looking for [sassy](https://github.com/RagnarGrootKoerkamp/sassy/) instead.

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
use mini_myers::{TQueries, mini_search};

let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
let transposed = TQueries::new(&queries);
let target = b"CCCTCGCCCCCCATGCCCCC";
let result = mini_search(&transposed, target, 4, None);
println!("Result: {:?}", result); 
// [0,1] (ATG = 0 edits, TTG = 1 edit)
```

#### With position tracking

```rust
use mini_myers::{TQueries, mini_search_with_positions};

let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
let transposed = TQueries::new(&queries);
let target = b"CCCTCGCCCCCCATGCCCCC";
let mut results = Vec::new();
let result = mini_search_with_positions(&transposed, target, 4, &mut results, None);
println!("Result: {:?}", result); 
// Result: [MatchInfo { query_idx: 1, cost: 1, pos: 5 }, MatchInfo { query_idx: 0, cost: 1, pos: 13 }, MatchInfo { query_idx: 0, cost: 0, pos: 14 }, MatchInfo { query_idx: 1, cost: 1, pos: 14 }, MatchInfo { query_idx: 0, cost: 1, pos: 15 }]
```
This returns *all* positions, which is not ideal perhaps, sassy returns the local minima position.

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
let result = mini_search_with_positions(&transposed, target, 4, &mut results, Some(0.5));
```
In the example above, this would give a cost of `3 * 0.5 = 1.5`


---

#### Benchmark
Note that `mini_myers` and `sassy` are not directly comparable. 
`mini_seaerch` just tells us if something is present below `k`,
`mini_search_with_positions` does return the match locations similar
to `sassy` though without traceback. Here we have a batch of 32 which 
is ideal for `mini_myers` as well.

| Profile | Query Len | µs/query (mini_myers) | µs/query (sassy) | Function                      |
|---------|-----------|-----------------------|------------------|-------------------------------|
| IUPAC   |   24      | 23.5                  | 50.7             | `mini_search`                   |
| IUPAC   |   32      | 23.9                  | 51.3             | `mini_search`                   |
| DNA     |   24      | 23.2                  | 52.2             | `mini_search`                   |
| DNA     |   32      | 23.1                  | 50.4             | `mini_search`                   |
| IUPAC   |   24      | 32.3                  | 54.0             | `mini_search_with_positions`    |
| IUPAC   |   32      | 32.8                  | 53.2             | `mini_search_with_positions`    |

_Searching for 32 queries in a 100K DNA string with k=4._

Run the bench using `cargo bench --bench sassy`.


