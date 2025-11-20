doc:
    cargo doc --no-deps --open

perm:
    sudo sysctl -w kernel.perf_event_paranoid=-1
    sudo sh -c 'echo 0 > /proc/sys/kernel/kptr_restrict'
    sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'

asm:
    cargo rustc --release --bin mini_myers -- --emit=asm && find target/release/deps -name "mini_myers*.s"

search_asm:
    cargo build --release --bin mini_myers
    objdump -d target/release/mini_myers | awk '/<.*Searcher.*scan.*>:/,/^$/' > search_asm.s








