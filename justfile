doc:
    cargo doc --no-deps --open

perm:
    sudo sysctl -w kernel.perf_event_paranoid=-1

asm:
    cargo rustc --release --bin mini_myers -- --emit=asm && find target/release/deps -name "mini_myers*.s"









