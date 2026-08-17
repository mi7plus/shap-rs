# Release checklist

- [x] Review the public API and intended SemVer compatibility.
- [ ] Update `CHANGELOG.md`, version, and release date.
- [ ] Run formatting, strict Clippy, all-feature tests, and warning-free docs.
- [ ] Verify Linux, Windows, macOS, Rust 1.80, and current stable CI.
- [ ] Run compatibility fixtures and review algorithm tolerances.
- [ ] Run `cargo package` and inspect the packaged file list.
- [ ] Run `cargo publish --dry-run` with registry access.
- [ ] Tag the reviewed commit as `vX.Y.Z` and publish the crate.
- [ ] Create release notes and verify docs.rs output.
