# Agent Code Examples

Code patterns for AI agents when generating tensor-aware code in Nivara.

## Zero-copy tensor kernel (safe path)

```csharp
// Precondition: tensorStorage.HasNulls == false
var span = tensorStorage.AsTensorSpan(); // returns TensorSpan<T>
// Call kernel that accepts TensorSpan<T> directly
MyKernels.AddTensorSpan(span, otherSpan, destinationSpan);
```

## Safe tensor creation from nullable values

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

```csharp
[Test]
public void NullMaskMaintenance_ArithmeticOperations_PreservesNullPositions()
{
    var testCases = new[] { new int?[] { 1, null, 3 } };
    foreach (var values in testCases)
    {
        var column = NivaraColumn<int>.CreateFromNullable(values);
        var result = column.Multiply(5);
        for (int i = 0; i < values.Length; i++)
            Assert.That(result.IsNull(i), Is.EqualTo(values[i] == null));
    }
}
```


