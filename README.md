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

| Profile | Query Len | µs/query (mini_myers) | µs/query (sassy) |
|---------|-----------|-----------------------|------------------|
| IUPAC   |   24      | 23.5                  | 50.7             |
| IUPAC   |   32      | 23.9                  | 51.3             |
| DNA     |   24      | 23.2                  | 52.2             |
| DNA     |   32      | 23.1                  | 50.4             |

_Searching for 32 queries in a 100K DNA string with k=4._



