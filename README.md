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

#### How to use:


#### Basic usage
This will return boolean results indicating whether each query matches (within `k` edits). 


### Presence of queries in texts

```rust
use mini_myers::{Searcher, TQueries};
use mini_myers::backend::{U32, U64};

let mut scanner = Searcher::<U32>::new(None);
let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
let encoded = TQueries::<U32>::new(&queries, false); //false = only fwd, true = fwd + rc
let target = b"CCCTCGCCCCCCATGCCCCC";
let results = scanner.scan(&encoded, target, 0);
// results = [true, false]
assert!(results[0]); // "ATG" matches
assert!(!results[1]); // "TTG" doesn't match
```

### End locations of queries in texts

```rust
use mini_myers::{Searcher, TQueries};
use mini_myers::backend::{U32, U64};

let mut searcher = Searcher::<U32>::new(None);
let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
let encoded = TQueries::<U32>::new(&queries, false); 
let target = b"CCCTCGCCCCCCATGCCCCC";
let results = searcher.search(&encoded, target, 0);
// results = [[14], []]
```

### Presence or locations with overhang
We use the `alpha` parameter to reduce the penalty for "overhanging" query sequence
at either end of the text: `overhang_length * alpha`:
```
    q: AAAACCC
           |||
    t:     CCCGGGGGGGGGGGGGG
       ^^^^ overhang (AAAA)
```
```rust
use mini_myers::{Searcher, TQueries};
use mini_myers::backend::{U32, U64};
use mini_myers::{Searcher, TQueries};
let mut searcher = Searcher::<U32>::new(Some(0.5));
let queries = vec![b"AAAACCC".to_vec()];
let encoded = TQueries::<U32>::new(&queries, false);
let target = b"CCCGGGGGGGGGGGGGG";
// 4 * 0.5(alpha) = 2
let results = searcher.search(&encoded, target, 2);
// results = [[0,1]]
```

### Multi-text scan (presence)
If you first run a search using a prefix, then want to compare the
entire sequence you can use `multi_text_scan` to compare a list 
of queries to a list of texts, where queries[0] is compared to texts[0], 
and so on:

```rust
let queries = vec![b"GGCC".to_vec(), b"AAAA".to_vec(), b"CCGG".to_vec()];
let texts: Vec<&[u8]> = vec![b"AAAAAAAAAAAAAAAAAAAAAAAAAAGG", 
                             b"GGGGGGGG", 
                             b"AAAAAAAAAACC"];

let transposed = TQueries::<U32>::new(&queries, false);
let mut searcher = Searcher::<U32>::new(Some(0.5));
let matches = searcher.multi_text_scan(&transposed, &texts, 1);
assert_eq!(matches, vec![true, false, true]);

```


---

#### Little bench
Note that `mini_myers` and `sassy` are not directly comparable.
- `sassy`: also preforms a traceback 
- `mini scan`: only returns presence
- `mini search`: returns end position (but not traceback)

Search for 96 queries at k=[1,4] in a range of targets (x-axis)

<img src="test_data/bench.png" alt="mini_scan vs sassy benchmark plot" width="500"/>


Run the same bench using `cargo bench --bench sassy`. 


#### Some dev stuff
see `justfile`, to get assembly for `scan`, disable inline `#[inline(never)]` and run `just search_asm`. To enable all permissions for flamegraph/perf `just perm`.