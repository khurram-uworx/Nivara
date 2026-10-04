# Agent Code Examples

Code patterns for AI agents when generating tensor-aware code in Nivara.

## Zero-copy tensor kernel (safe path)

<!-- gate
preamble: agent-tensor-kernel
locals: NivaraColumn<float> column = NivaraColumn<float>.Create(new[] { 1f, 2f, 3f }); Span<float> otherSpan = new[] { 1f, 1f, 1f }; Span<float> destinationSpan = new float[3];
-->
```csharp
// Precondition: column.HasNulls == false - TryGetSpan returns false otherwise.
column.TryGetSpan(out ReadOnlySpan<float> span);   // zero-copy read-only view

// Hand the span straight to a kernel - no ColumnStorage round-trip.
TensorPrimitives.Add(span, otherSpan, destinationSpan);
```

## Safe tensor creation from nullable values

<!-- gate
mode: GenericLocal
preamble: agent-null-tensor
locals: int len = 4; T?[] values = new T?[] { null, null, null, null }; bool hasNulls = true;
-->
```csharp
var data = new T[len];
var nullMask = new bool[len];
for (int i=0;i<len;i++) {
  if (values[i].HasValue) data[i] = values[i].Value; else { data[i] = default; nullMask[i] = true; }
}
var tensor = Tensor.Create(data, new nint[] { len });
var nullTensor = hasNulls ? Tensor.Create(nullMask, new nint[] { len }) : null;
```

## Null handling test pattern

<!-- gate
mode: MemberDecl
-->
```csharp
[Test]
public void NullMaskMaintenance_ArithmeticOperations_PreservesNullPositions()
{
    var testCases = new[] { new int?[] { 1, null, 3 } };
    foreach (var values in testCases)
    {
        var column = NivaraColumnFactory.CreateFromNullable(values);
        var result = column.Multiply(5);
        for (int i = 0; i < values.Length; i++)
            Assert.That(result.IsNull(i), Is.EqualTo(values[i] == null));
    }
}
```


