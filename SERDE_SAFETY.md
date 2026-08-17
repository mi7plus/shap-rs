# Serde safety audit

Public deserializable types were audited for constructor bypasses. Types whose
invariants protect indexing, dimensions, graph traversal, or allocation use
validated deserialization and reject malformed payloads immediately:

- `Background`, `EvaluationConfig`, `FeatureMetadata`, and `OutputMetadata`
- `Explanation`, `UncertainExplanation`, and `InteractionExplanation`
- `FixedMasker`, `TextMasker`, and `ImageMasker`
- `CausalGraph`, `FeaturePartition`, and `PartitionTree`
- `Tree` and `TreeEnsemble`

`IndependentMasker` contains a `Background`, whose custom deserializer performs
the required validation. `TreeArrays`, `Node`, and `PartitionNode` are inert
interchange DTOs; conversion into an executable tree or hierarchy goes through
validated constructors. Configuration values such as `AdditivityTolerance`
are revalidated by every consuming operation.

Enums, errors, analysis results, and plot-data records have no private
cross-field invariants and do not index or allocate based on their contents.
They retain derived deserialization. Executable tree prediction also performs
defensive validation even though invalid trees can no longer be deserialized.

Fuzz targets exercise the validated types and their use-time boundaries. New
public serde types must be added to this audit and to `serde_payloads` or a more
specific fuzz target.
