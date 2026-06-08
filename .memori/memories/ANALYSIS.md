# Memories

- Current usage analysis: All frame-level math wrappers (Dot, CosineSimilarity, ColumnNorms, RowNorms) called only from test code. Series extensions (DotProduct, Norm, SumTensor, AddTensor, MultiplyTensor, TransformTensor) also only from test code. MatrixMultiply completely unused. ToTensor/FromTensor have production callers including AutoDiff and MLNet paths. ToTensorSpan/FromTensorSpan etc only from tests. <!-- id=b02e1cec6ad3413f941d4b176b6a545b entity=default type=analysis ts=2026-06-08T05:58:49.9178922+00:00 v=1 tags=usage,analysis,callers -->
