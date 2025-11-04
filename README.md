### Mini Myers
Simple Myers implementation to "check" whether short patterns (<=32nt) are present in a longer 
text with at most `k` edits. 

### When to use
- Short queries (`<=32`)
- Search for multiple of `8` queries
- Don't need positions
- Only have DNA chars (no IUPAC)

Most likely you want something like [sassy](https://github.com/RagnarGrootKoerkamp/sassy/), this is faster prefilter to run before sassy.


#### What it does
We compare each character of the queries against a single text character at the time using SIMD.
Then we track the lowest cost 
we observe along the entire text and report the cost when below the cut-off `k`, or `-1` if above `k`. 

#### How to use:
```rust
use mini_myers::{TQueries, mini_search};

let queries = vec![b"ATG".to_vec(), b"TTG".to_vec()];
let transposed = TQueries::new(&queries);
let target = b"CCCTCGCCCCCCATGCCCCC";
let result = mini_search(&transposed, target, 4);
println!("Result: {:?}", result); 
// [0,1] (ATG = 0 edits, TTG = 1 edit)
```

#### Benchmark
Searching for 32 queries of length 24 in a 100K DNA string:
```bash
target=100000  query=24 k=4 | mini_myers:   0.7414 ms/batch ( 23.1689 µs/query), sassy:   1.6700 ms/batch ( 52.1862 µs/query)
target=100000  query=32 k=4 | mini_myers:   0.7387 ms/batch ( 23.0834 µs/query), sassy:   1.6135 ms/batch ( 50.4230 µs/query)
```


