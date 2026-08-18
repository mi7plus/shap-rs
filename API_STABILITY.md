# Public API and SemVer policy

This document records the pre-1.0 public-API review for `shap-rs` 0.1.0.

## Supported surface

The crate-root re-exports are the preferred API: model and explainer traits,
explanation types, metadata, maskers, execution configuration, and tree model
types. Named explainer, interaction, plotting, analysis, and adapter modules are
also intentional public extension points. `coalition` is public deliberately so
custom explainers can share the crate's mask representation and Kernel SHAP
weights.

All public constructors that can reject input return `Result`; serialized types
validate during deserialization, and public selection/indexing operations return
typed errors. Read-only accessors expose views or slices rather than mutable
representation internals.

## Stability boundaries

- `Predict`, `Explainer`, `Masker`, `DifferentiablePredict`, and
  `DeepAttribution` are the core dense integration contracts. Their owned-batch
  and streaming methods have default implementations so existing adapters do
  not need to implement transfer or out-of-core optimizations.
- `SparseMatrix`, `SparsePredict`, and `SparseIndependentMasker` are the native
  CSR integration boundary. CSR representation fields remain private and all
  construction and deserialization paths validate canonical row structure.
- `Explanation`, its axis ordering, metadata, and attribution semantics are the
  interchange contract. Serialized payloads carry an explicit schema version.
- Tree adapter input schemas track upstream exporters and may add support for
  new variants in minor releases. Unsupported variants return errors.
- Burn, JSON tree import, Rayon, and their transitive types are feature-gated.
- The crate follows Cargo SemVer. While the version is below 1.0, a minor
  release may contain breaking API changes; patch releases will remain
  compatible except where a soundness or panic fix requires otherwise.

## Review outcome

The 0.1.0 audit found no accidental mutable representation exposure or public
implementation-only type requiring removal. Public modules and root re-exports
are retained intentionally. Future releases should compare generated rustdoc or
`cargo public-api` output against the preceding release before publication and
classify every removal, signature change, and newly exhaustive enum variant.

