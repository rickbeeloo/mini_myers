### 🤏 Mini Myers
Simple SIMD Myers implementation to "check" whether short patterns (<=32nt) are present in a longer 
text with at most `k` edits. 

---

### When to use
- Short queries (`<=32`)
- Search for multiple of `8` queries
- Don't need positions
- Supports DNA IUPAC characters (A, C, G, T, N, ...)

**Most likely you want something like [sassy](https://github.com/RagnarGrootKoerkamp/sassy/)**, this is a faster prefilter to run before sassy but does not return the positions, nor works for longer queries.

--- 

#### What it does
We compare each character of the queries against a single text character at the time using SIMD.
Then we track the lowest cost 
we see along the entire text and report the cost when below the cut-off `k`, or `-1` if above `k`. 

--- 

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
---

#### Benchmark
Searching for 32 queries of length 24 in a 100K DNA string:
```bash
IUPAC, note a lot goes intro tracing for Sassy for |q|=24;k=4
target=100000  query=24 k=4 | 
mini_myers:   0.7652 ms/batch ( 23.9111 µs/query) 
sassy:  41.2407 ms/batch (1288.7725 µs/query)
target=100000  query=32 k=4 | 
mini_myers:   0.7681 ms/batch ( 24.0024 µs/query)
sassy:   4.1853 ms/batch (130.7896 µs/query)


DNA
target=100000  query=24 k=4 
mini_myers:   0.7414 ms/batch ( 23.1689 µs/query)
sassy:   1.6700 ms/batch ( 52.1862 µs/query)

target=100000  query=32 k=4
mini_myers:   0.7387 ms/batch ( 23.0834 µs/query)
sassy:   1.6135 ms/batch ( 50.4230 µs/query)
```


