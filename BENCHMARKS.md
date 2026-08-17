# Performance baseline

Benchmarks use Criterion and are reproducible with:

```text
cargo bench --all-features --bench core
```

## TreeSHAP batching baseline

Measured on 2026-08-17 with Rust 1.97.1, Windows x86-64, and an Intel
Core i7-1355U (10 cores, 12 logical processors). The workload explains 1,024
rows with an ensemble of 128 shallow trees.

| Execution | Median range |
|---|---:|
| Sequential batch | 34.337–35.165 ms |
| Rayon, 64-row chunks | 8.306–8.472 ms |

The measured parallel speedup is approximately 4.1x. This supports retaining
sample-level Rayon parallelism. No explicit SIMD implementation is included:
TreeSHAP is branch- and path-state-heavy, so a SIMD-specific representation
should only be added after lower-level profiling demonstrates a vectorizable
hot loop and benchmarks show a gain beyond compiler auto-vectorization.

These figures are a local baseline, not a cross-platform performance promise.
