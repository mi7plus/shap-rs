# Contributing

Thank you for improving `shap-rs`. Open an issue before undertaking a large
API change so its semantics and compatibility fixtures can be agreed first.

## Development

The minimum supported Rust version is 1.80. Before submitting a pull request,
run:

```text
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
cargo package
```

New algorithms should include deterministic tests, local-accuracy checks where
applicable, documented numerical tolerances, and reference fixtures when a
compatible external implementation exists. Public constructors must reject
malformed dimensions and non-finite configuration values.

Contributions are licensed under the repository's MIT license.
