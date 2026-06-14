# Future Work

## FrameExtensions - Data-Prep

Future frame-level data-prep helpers:

- `Split(double trainRatio, int? seed = null)` returns train and test frames using a deterministic Fisher-Yates shuffle when a seed is supplied.
- `Normalize(params string[] columnNames)` applies z-score normalization to selected numeric columns and skips null values.
- `Standardize(params string[] columnNames)` is an alias for `Normalize`.

These helpers belong to the frame/data-prep surface, not AutoDiff or LINQ integration. They should be planned and tested as preprocessing conveniences, with no use in AutoDiff training hot paths.
