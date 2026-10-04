# Negative control fixture — not documentation

This file is not a Nivara document and is deliberately **not** reachable from the gate's
document scan, so it never counts toward coverage. It exists so that a green run of the
gate is known to mean something: the extractor must find this block and the compiler
must reject it. If the block below ever compiles, the gate has stopped gating.

<!-- gate
preamble: streaming-as-stream
locals: NivaraColumn<int> column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
-->
```csharp
var total = column.NoSuchMethod();
```