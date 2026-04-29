# Artifacts

Generated or staged artifacts that are not canonical source code.

- `figures/`: reserved for generated cross-paper figures.
- `tables/`: reserved for generated cross-paper tables.
- `generated/`: staging snapshots, generated Chroma stores, and deployment build outputs.

Large generated files under `artifacts/generated/` are ignored by git unless a
specific release process says otherwise.
