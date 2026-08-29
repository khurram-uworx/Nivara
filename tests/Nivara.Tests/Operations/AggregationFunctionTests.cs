using Nivara;
using NUnit.Framework;

namespace Nivara.Tests.Operations;

[TestFixture]
public class AggregationFunctionTests
{
    [TestFixture]
    public class CountAggregationTests
    {
        [Test]
        public void Apply_WithValidValues_ReturnsCorrectCount()
        {
            // Arrange
            var values = new[] { 1, 2, 3, 4, 5 };
            var column = NivaraColumn<int>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Count();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(5L));
        }

        [Test]
        public void Apply_WithNullValues_CountsOnlyNonNulls()
        {
            // Arrange
            var values = new int?[] { 1, null, 3, null, 5 };
            var column = NivaraColumn<int?>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Count();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(3L)); // Only non-null values
        }

        [Test]
        public void Apply_WithEmptyIndices_ReturnsZero()
        {
            // Arrange
            var values = new[] { 1, 2, 3 };
            var column = NivaraColumn<int>.Create(values);
            var indices = new List<int>();
            var aggregation = AggregationFunctions.Count();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(0L));
        }

        [Test]
        public void GetResultType_ReturnsLong()
        {
            // Arrange
            var aggregation = AggregationFunctions.Count();

            // Act
            var resultType = aggregation.GetResultType(typeof(int));

            // Assert
            Assert.That(resultType, Is.EqualTo(typeof(long)));
        }
    }

    [TestFixture]
    public class SumAggregationTests
    {
        [Test]
        public void Apply_WithIntegerValues_ReturnsCorrectSum()
        {
            // Arrange
            var values = new[] { 1, 2, 3, 4, 5 };
            var column = NivaraColumn<int>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Sum();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(15L)); // Sum as long
        }

        [Test]
        public void Apply_WithByteAndShortValues_ReturnsWidenedLongSum()
        {
            // Arrange
            var byteColumn = NivaraColumn<byte>.Create(new byte[] { 200, 100 });
            var shortColumn = NivaraColumn<short>.Create(new short[] { -100, 500 });
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Sum();

            // Act
            var byteResult = aggregation.Apply(byteColumn, indices);
            var shortResult = aggregation.Apply(shortColumn, indices);

            // Assert
            Assert.That(byteResult, Is.EqualTo(300L));
            Assert.That(shortResult, Is.EqualTo(400L));
        }

        [Test]
        public void Apply_WithDoubleValues_ReturnsCorrectSum()
        {
            // Arrange
            var values = new[] { 1.5, 2.5, 3.0 };
            var column = NivaraColumn<double>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Sum();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(7.0).Within(0.001));
        }

        [Test]
        public void Apply_WithFloatValues_ReturnsCorrectSum()
        {
            // Arrange
            var values = new[] { 1.5f, 2.5f, 3.0f };
            var column = NivaraColumn<float>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Sum();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(7.0).Within(0.001));
        }

        [Test]
        public void Apply_WithNullValues_IgnoresNulls()
        {
            // Arrange
            var values = new int?[] { 1, null, 3, null, 5 };
            var column = NivaraColumn<int?>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Sum();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(9L)); // 1 + 3 + 5
        }

        [Test]
        public void Apply_WithStringValues_ThrowsArgumentException()
        {
            // Arrange
            var values = new[] { "a", "b", "c" };
            var column = NivaraColumn<string>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Sum();

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => aggregation.Apply(column, indices));
            Assert.That(ex.Message, Contains.Substring("Sum aggregation requires numeric type"));
        }

        [Test]
        public void GetResultType_WithIntegerInput_ReturnsLong()
        {
            // Arrange
            var aggregation = AggregationFunctions.Sum();

            // Act
            var resultType = aggregation.GetResultType(typeof(int));

            // Assert
            Assert.That(resultType, Is.EqualTo(typeof(long)));
        }

        [Test]
        public void GetResultType_WithFloatInput_ReturnsDouble()
        {
            // Arrange
            var aggregation = AggregationFunctions.Sum();

            // Act
            var resultType = aggregation.GetResultType(typeof(float));

            // Assert
            Assert.That(resultType, Is.EqualTo(typeof(double)));
        }

        [Test]
        public void Apply_WithUIntValues_ReturnsWidenedLongSum()
        {
            var values = new uint[] { 4_000_000_000, 1_000_000_000 };
            var column = NivaraColumn<uint>.Create(values);
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Sum();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(5_000_000_000L));
        }

        [Test]
        public void Apply_WithUShortValues_ReturnsWidenedLongSum()
        {
            var values = new ushort[] { 65535, 1 };
            var column = NivaraColumn<ushort>.Create(values);
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Sum();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(65536L));
        }

        [Test]
        public void Apply_WithSByteValues_ReturnsWidenedLongSum()
        {
            var values = new sbyte[] { -128, 127 };
            var column = NivaraColumn<sbyte>.Create(values);
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Sum();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(-1L));
        }

        [Test]
        public void Apply_WithCharValues_ReturnsWidenedLongSum()
        {
            var values = new char[] { 'a', 'b' };
            var column = NivaraColumn<char>.Create(values);
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Sum();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(195L)); // 97 + 98
        }

        [Test]
        public void Apply_WithBoolValues_ReturnsTrueCount()
        {
            var values = new bool[] { true, false, true, true };
            var column = NivaraColumn<bool>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3 };
            var aggregation = AggregationFunctions.Sum();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(3L));
        }

        [Test]
        public void Apply_WithUInt64Values_ReturnsUInt64Sum()
        {
            var values = new ulong[] { 10_000_000_000, 20_000_000_000 };
            var column = NivaraColumn<ulong>.Create(values);
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Sum();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(30_000_000_000UL));
        }

        [Test]
        public void Apply_WithNIntValues_ReturnsInt128Sum()
        {
            var values = new nint[] { 5, 10 };
            var column = NivaraColumn<nint>.Create(values);
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Sum();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo((Int128)15));
        }

        [Test]
        public void Apply_WithNUIntValues_ReturnsUInt128Sum()
        {
            var values = new nuint[] { 5, 10 };
            var column = NivaraColumn<nuint>.Create(values);
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Sum();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo((UInt128)15));
        }

        [Test]
        public void Apply_WithInt128Values_ReturnsInt128Sum()
        {
            var values = new Int128[] { 5, 10 };
            var column = NivaraColumn<Int128>.Create(values);
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Sum();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo((Int128)15));
        }

        [Test]
        public void Apply_WithUInt128Values_ReturnsUInt128Sum()
        {
            var values = new UInt128[] { 5, 10 };
            var column = NivaraColumn<UInt128>.Create(values);
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Sum();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo((UInt128)15));
        }

        [Test]
        public void Apply_WithHalfValues_ReturnsWidenedDoubleSum()
        {
            var values = new Half[] { (Half)1.5, (Half)2.5, (Half)3.0 };
            var column = NivaraColumn<Half>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Sum();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(7.0).Within(0.001));
        }

        [Test]
        public void Apply_WithNullableUIntValues_IgnoresNulls()
        {
            var values = new uint?[] { 4_000_000_000, null, 1_000_000_000 };
            var column = NivaraColumn<uint?>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Sum();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(5_000_000_000L));
        }

        [Test]
        public void Apply_WithEmptyIndices_ReturnsTypedZeroForNewTypes()
        {
            var ulongColumn = NivaraColumn<ulong>.Create(new ulong[] { 1, 2 });
            var int128Column = NivaraColumn<Int128>.Create(new Int128[] { 1, 2 });
            var uint128Column = NivaraColumn<UInt128>.Create(new UInt128[] { 1, 2 });
            var aggregation = AggregationFunctions.Sum();
            var indices = new List<int>();

            Assert.That(aggregation.Apply(ulongColumn, indices), Is.EqualTo(0UL));
            Assert.That(aggregation.Apply(int128Column, indices), Is.EqualTo(Int128.Zero));
            Assert.That(aggregation.Apply(uint128Column, indices), Is.EqualTo(UInt128.Zero));
        }

        [Test]
        public void GetResultType_WithNewNumericTypes_ReturnsPromotedTypes()
        {
            var aggregation = AggregationFunctions.Sum();

            Assert.That(aggregation.GetResultType(typeof(uint)), Is.EqualTo(typeof(long)));
            Assert.That(aggregation.GetResultType(typeof(ushort)), Is.EqualTo(typeof(long)));
            Assert.That(aggregation.GetResultType(typeof(sbyte)), Is.EqualTo(typeof(long)));
            Assert.That(aggregation.GetResultType(typeof(char)), Is.EqualTo(typeof(long)));
            Assert.That(aggregation.GetResultType(typeof(bool)), Is.EqualTo(typeof(long)));
            Assert.That(aggregation.GetResultType(typeof(ulong)), Is.EqualTo(typeof(ulong)));
            Assert.That(aggregation.GetResultType(typeof(nint)), Is.EqualTo(typeof(Int128)));
            Assert.That(aggregation.GetResultType(typeof(nuint)), Is.EqualTo(typeof(UInt128)));
            Assert.That(aggregation.GetResultType(typeof(Int128)), Is.EqualTo(typeof(Int128)));
            Assert.That(aggregation.GetResultType(typeof(UInt128)), Is.EqualTo(typeof(UInt128)));
            Assert.That(aggregation.GetResultType(typeof(Half)), Is.EqualTo(typeof(double)));
        }
    }

    [TestFixture]
    public class MinAggregationTests
    {
        [Test]
        public void Apply_WithIntegerValues_ReturnsMinimum()
        {
            // Arrange
            var values = new[] { 5, 2, 8, 1, 9 };
            var column = NivaraColumn<int>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Min();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(1));
        }

        [Test]
        public void Apply_WithDoubleValues_ReturnsMinimum()
        {
            // Arrange
            var values = new[] { 5.5, 2.2, 8.8, 1.1, 9.9 };
            var column = NivaraColumn<double>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Min();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(1.1).Within(0.001));
        }

        [Test]
        public void Apply_WithStringValues_ReturnsLexicographicMinimum()
        {
            // Arrange
            var values = new[] { "zebra", "apple", "banana", "cherry" };
            var column = NivaraColumn<string>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3 };
            var aggregation = AggregationFunctions.Min();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo("apple"));
        }

        [Test]
        public void Apply_WithDecimalValues_ReturnsMinimum()
        {
            // Arrange
            var values = new[] { 5.5m, 2.2m, 8.8m, 1.1m, 9.9m };
            var column = NivaraColumn<decimal>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Min();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(1.1m));
        }

        [Test]
        public void Apply_WithNullValues_IgnoresNulls()
        {
            // Arrange
            var values = new int?[] { 5, null, 2, null, 1 };
            var column = NivaraColumn<int?>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Min();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(1));
        }

        [Test]
        public void Apply_WithAllNullValues_ReturnsNull()
        {
            // Arrange
            var values = new int?[] { null, null, null };
            var column = NivaraColumn<int?>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Min();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.Null);
        }
    }

    [TestFixture]
    public class MaxAggregationTests
    {
        [Test]
        public void Apply_WithIntegerValues_ReturnsMaximum()
        {
            // Arrange
            var values = new[] { 5, 2, 8, 1, 9 };
            var column = NivaraColumn<int>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Max();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(9));
        }

        [Test]
        public void Apply_WithNullValues_IgnoresNulls()
        {
            // Arranged via the nullable-element column type (NivaraColumn<int?>) so the typed
            // extraction must read through IColumn<int?>, not IColumn<int>.
            // Arrange
            var values = new int?[] { 5, null, 2, null, 1 };
            var column = NivaraColumn<int?>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Max();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(5));
        }

        [Test]
        public void Apply_WithAllNullValues_ReturnsNull()
        {
            // Arrange
            var values = new int?[] { null, null, null };
            var column = NivaraColumn<int?>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Max();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.Null);
        }

        [Test]
        public void Apply_WithDoubleValues_ReturnsMaximum()
        {
            // Arrange
            var values = new[] { 5.5, 2.2, 8.8, 1.1, 9.9 };
            var column = NivaraColumn<double>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Max();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(9.9).Within(0.001));
        }

        [Test]
        public void Apply_WithStringValues_ReturnsLexicographicMaximum()
        {
            // Arrange
            var values = new[] { "zebra", "apple", "banana", "cherry" };
            var column = NivaraColumn<string>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3 };
            var aggregation = AggregationFunctions.Max();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo("zebra"));
        }

        [Test]
        public void Apply_WithDecimalValues_ReturnsMaximum()
        {
            // Arrange
            var values = new[] { 5.5m, 2.2m, 8.8m, 1.1m, 9.9m };
            var column = NivaraColumn<decimal>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Max();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(9.9m));
        }
    }

    [TestFixture]
    public class MeanAggregationTests
    {
        [Test]
        public void Apply_WithIntegerValues_ReturnsCorrectMean()
        {
            // Arrange
            var values = new[] { 1, 2, 3, 4, 5 };
            var column = NivaraColumn<int>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Mean();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(3.0).Within(0.001));
        }

        [Test]
        public void Apply_WithDoubleValues_ReturnsCorrectMean()
        {
            // Arrange
            var values = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
            var column = NivaraColumn<double>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Mean();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(3.0).Within(0.001));
        }

        [Test]
        public void Apply_WithNullValues_IgnoresNulls()
        {
            // Arrange
            var values = new int?[] { 1, null, 3, null, 5 };
            var column = NivaraColumn<int?>.Create(values);
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Mean();

            // Act
            var result = aggregation.Apply(column, indices);

            // Assert
            Assert.That(result, Is.EqualTo(3.0).Within(0.001)); // (1 + 3 + 5) / 3
        }

        [Test]
        public void Apply_WithStringValues_ThrowsArgumentException()
        {
            // Arrange
            var values = new[] { "a", "b", "c" };
            var column = NivaraColumn<string>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Mean();

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => aggregation.Apply(column, indices));
            Assert.That(ex.Message, Contains.Substring("Mean aggregation requires numeric type"));
        }

        [Test]
        public void GetResultType_ReturnsDouble()
        {
            // Arrange
            var aggregation = AggregationFunctions.Mean();

            // Act
            var resultType = aggregation.GetResultType(typeof(int));

            // Assert
            Assert.That(resultType, Is.EqualTo(typeof(double)));
        }

        [Test]
        public void Apply_WithUIntValues_ReturnsCorrectMean()
        {
            var values = new uint[] { 2, 4, 6 };
            var column = NivaraColumn<uint>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Mean();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(4.0).Within(0.001));
        }

        [Test]
        public void Apply_WithUInt64Values_ReturnsCorrectMean()
        {
            var values = new ulong[] { 2, 4, 6 };
            var column = NivaraColumn<ulong>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Mean();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(4.0).Within(0.001));
        }

        [Test]
        public void Apply_WithInt128Values_ReturnsCorrectMean()
        {
            var values = new Int128[] { 2, 4, 6 };
            var column = NivaraColumn<Int128>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Mean();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(4.0).Within(0.001));
        }

        [Test]
        public void Apply_WithUInt128Values_ReturnsCorrectMean()
        {
            var values = new UInt128[] { 2, 4, 6 };
            var column = NivaraColumn<UInt128>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Mean();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(4.0).Within(0.001));
        }

        [Test]
        public void Apply_WithHalfValues_ReturnsCorrectMean()
        {
            var values = new Half[] { (Half)1.5, (Half)2.5, (Half)3.5 };
            var column = NivaraColumn<Half>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Mean();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(2.5).Within(0.001));
        }

        [Test]
        public void Apply_WithBoolValues_ReturnsTrueProportion()
        {
            var values = new bool[] { true, false, true };
            var column = NivaraColumn<bool>.Create(values);
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Mean();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(2.0 / 3.0).Within(0.001));
        }

        [Test]
        public void Apply_WithCharValues_ReturnsCorrectMean()
        {
            var values = new char[] { 'a', 'c' };
            var column = NivaraColumn<char>.Create(values);
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Mean();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(98.0).Within(0.001)); // (97 + 99) / 2
        }
    }

    [TestFixture]
    public class QuantileAggregationTests
    {
        [Test]
        public void Apply_WithIntegerValues_ReturnsCorrectQuantile()
        {
            var column = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 });
            var indices = new List<int> { 0, 1, 2, 3 };
            var aggregation = AggregationFunctions.Quantile(0.5);

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(2.5).Within(1e-9));
        }

        [Test]
        public void Apply_WithFloatValues_InterpolatesLinearly()
        {
            var column = NivaraColumn<double>.Create(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Quantile(0.9);

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(46.0).Within(1e-9));
        }

        [Test]
        public void Apply_WithMinQuantile_ReturnsMinimum()
        {
            var column = NivaraColumn<int>.Create(new[] { 3, 9, 6 });
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Quantile(0.0);

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(3.0));
        }

        [Test]
        public void Apply_WithMaxQuantile_ReturnsMaximum()
        {
            var column = NivaraColumn<int>.Create(new[] { 3, 9, 6 });
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Quantile(1.0);

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(9.0));
        }

        [Test]
        public void Apply_WithNullValues_IgnoresNulls()
        {
            var column = NivaraColumn.CreateFromNullable(new double?[] { 5, null, 3, 1, 4 });
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Quantile(0.25);

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(2.5).Within(1e-9));
        }

        [Test]
        public void Apply_WithNullableElementColumn_IgnoresNulls()
        {
            // Uses the nullable-element column type (NivaraColumn<double?>) so the typed
            // extraction must read through IColumn<double?>, not IColumn<double>.
            var column = NivaraColumn<double?>.Create(new double?[] { 5, null, 3, 1, 4 });
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Quantile(0.25);

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(2.5).Within(1e-9));
        }

        [Test]
        public void Apply_WithAllNullValues_ReturnsNull()
        {
            var column = NivaraColumn.CreateFromNullable(new double?[] { null, null });
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Quantile(0.5);

            Assert.That(aggregation.Apply(column, indices), Is.Null);
        }

        [Test]
        public void Apply_WithEmptyIndices_ReturnsNull()
        {
            var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
            var aggregation = AggregationFunctions.Quantile(0.5);

            Assert.That(aggregation.Apply(column, new List<int>()), Is.Null);
        }

        [Test]
        public void Apply_WithInt128Values_ReturnsCorrectQuantile()
        {
            var column = NivaraColumn<Int128>.Create(new Int128[] { 1, 2, 3, 4 });
            var indices = new List<int> { 0, 1, 2, 3 };
            var aggregation = AggregationFunctions.Quantile(0.5);

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(2.5).Within(1e-9));
        }

        [Test]
        public void Apply_WithHalfValues_ReturnsCorrectQuantile()
        {
            var column = NivaraColumn<Half>.Create(new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 });
            var indices = new List<int> { 0, 1, 2, 3 };
            var aggregation = AggregationFunctions.Quantile(0.5);

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(2.5).Within(1e-3));
        }

        [Test]
        public void Apply_WithStringValues_ThrowsArgumentException()
        {
            var column = NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b" });
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Quantile(0.5);

            Assert.Throws<ArgumentException>(() => aggregation.Apply(column, indices));
        }

        [Test]
        public void Constructor_InvalidQuantile_ThrowsArgumentOutOfRangeException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => AggregationFunctions.Quantile(-0.1));
            Assert.Throws<ArgumentOutOfRangeException>(() => AggregationFunctions.Quantile(1.1));
            Assert.Throws<ArgumentOutOfRangeException>(() => AggregationFunctions.Quantile(double.NaN));
        }

        [Test]
        public void Name_ReflectsQuantileArgument()
        {
            Assert.That(AggregationFunctions.Quantile(0.25).Name, Is.EqualTo("Quantile(0.25)"));
        }

        [Test]
        public void GetResultType_ReturnsDouble()
        {
            Assert.That(AggregationFunctions.Quantile(0.5).GetResultType(typeof(int)), Is.EqualTo(typeof(double)));
            Assert.That(AggregationFunctions.Quantile(0.5).GetResultType(typeof(double)), Is.EqualTo(typeof(double)));
        }
    }

    [TestFixture]
    public class MedianAggregationTests
    {
        [Test]
        public void Apply_WithOddLengthValues_ReturnsMiddleValue()
        {
            var column = NivaraColumn<int>.Create(new[] { 3, 1, 2 });
            var indices = new List<int> { 0, 1, 2 };
            var aggregation = AggregationFunctions.Median();

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(2.0));
        }

        [Test]
        public void Apply_WithEvenLengthValues_AveragesMiddleTwo()
        {
            var column = NivaraColumn<int>.Create(new[] { 1, 3, 2, 4 });
            var indices = new List<int> { 0, 1, 2, 3 };
            var aggregation = AggregationFunctions.Median();

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(2.5).Within(1e-9));
        }

        [Test]
        public void Apply_WithNullValues_IgnoresNulls()
        {
            var column = NivaraColumn.CreateFromNullable(new double?[] { 5, null, 3, 1, 4 });
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Median();

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(3.5).Within(1e-9));
        }

        [Test]
        public void Apply_WithNullableElementColumn_IgnoresNulls()
        {
            var column = NivaraColumn<double?>.Create(new double?[] { 5, null, 3, 1, 4 });
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Median();

            var result = aggregation.Apply(column, indices);

            Assert.That(result, Is.EqualTo(3.5).Within(1e-9));
        }

        [Test]
        public void Apply_WithAllNullValues_ReturnsNull()
        {
            var column = NivaraColumn.CreateFromNullable(new double?[] { null, null });
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Median();

            Assert.That(aggregation.Apply(column, indices), Is.Null);
        }

        [Test]
        public void Name_IsMedian()
        {
            Assert.That(AggregationFunctions.Median().Name, Is.EqualTo("Median"));
        }

        [Test]
        public void GetResultType_ReturnsDouble()
        {
            Assert.That(AggregationFunctions.Median().GetResultType(typeof(long)), Is.EqualTo(typeof(double)));
        }
    }

    [TestFixture]
    public class StdDevAggregationTests
    {
        [Test]
        public void Apply_WithValues_ReturnsPopulationStdDev()
        {
            var column = NivaraColumn<int>.Create(new[] { 2, 4, 4, 4, 5, 5, 7, 9 });
            var indices = new List<int> { 0, 1, 2, 3, 4, 5, 6, 7 };
            var aggregation = AggregationFunctions.StdDev();

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(2.0).Within(1e-9));
        }

        [Test]
        public void Apply_WithSampleDdof_ReturnsSampleStdDev()
        {
            var column = NivaraColumn<int>.Create(new[] { 2, 4, 4, 4, 5, 5, 7, 9 });
            var indices = new List<int> { 0, 1, 2, 3, 4, 5, 6, 7 };
            var aggregation = AggregationFunctions.StdDev(1);

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(2.138089935299395).Within(1e-9));
        }

        [Test]
        public void Apply_WithNullValues_IgnoresNulls()
        {
            var column = NivaraColumn.CreateFromNullable(new double?[] { 5, null, 3, 1, 4 });
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.StdDev();

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(1.479019945774904).Within(1e-9));
        }

        [Test]
        public void Apply_WithSingleValuePopulation_ReturnsZero()
        {
            var column = NivaraColumn<int>.Create(new[] { 7 });
            var indices = new List<int> { 0 };
            var aggregation = AggregationFunctions.StdDev();

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(0.0));
        }

        [Test]
        public void Apply_WithAllNullValues_ReturnsNull()
        {
            var column = NivaraColumn.CreateFromNullable(new double?[] { null, null });
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.StdDev();

            Assert.That(aggregation.Apply(column, indices), Is.Null);
        }

        [Test]
        public void Apply_WithEmptyIndices_ReturnsNull()
        {
            var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
            var aggregation = AggregationFunctions.StdDev();

            Assert.That(aggregation.Apply(column, new List<int>()), Is.Null);
        }

        [Test]
        public void Apply_WithStringValues_ThrowsArgumentException()
        {
            var column = NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b" });
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.StdDev();

            Assert.Throws<ArgumentException>(() => aggregation.Apply(column, indices));
        }

        [Test]
        public void Constructor_NegativeDdof_ThrowsArgumentOutOfRangeException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => AggregationFunctions.StdDev(-1));
        }

        [Test]
        public void Apply_SampleOverSingleValue_ThrowsInvalidOperationException()
        {
            var column = NivaraColumn<int>.Create(new[] { 7 });
            var indices = new List<int> { 0 };
            var aggregation = AggregationFunctions.StdDev(1);

            Assert.Throws<InvalidOperationException>(() => aggregation.Apply(column, indices));
        }

        [Test]
        public void Name_ReflectsDdof()
        {
            Assert.That(AggregationFunctions.StdDev().Name, Is.EqualTo("StdDev"));
            Assert.That(AggregationFunctions.StdDev(1).Name, Is.EqualTo("StdDev(1)"));
        }

        [Test]
        public void GetResultType_ReturnsDouble()
        {
            Assert.That(AggregationFunctions.StdDev().GetResultType(typeof(int)), Is.EqualTo(typeof(double)));
        }
    }

    [TestFixture]
    public class VarianceAggregationTests
    {
        [Test]
        public void Apply_WithValues_ReturnsPopulationVariance()
        {
            var column = NivaraColumn<int>.Create(new[] { 2, 4, 4, 4, 5, 5, 7, 9 });
            var indices = new List<int> { 0, 1, 2, 3, 4, 5, 6, 7 };
            var aggregation = AggregationFunctions.Variance();

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(4.0).Within(1e-9));
        }

        [Test]
        public void Apply_WithSampleDdof_ReturnsSampleVariance()
        {
            var column = NivaraColumn<int>.Create(new[] { 2, 4, 4, 4, 5, 5, 7, 9 });
            var indices = new List<int> { 0, 1, 2, 3, 4, 5, 6, 7 };
            var aggregation = AggregationFunctions.Variance(1);

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(4.571428571428571).Within(1e-9));
        }

        [Test]
        public void Apply_WithNullValues_IgnoresNulls()
        {
            var column = NivaraColumn.CreateFromNullable(new double?[] { 5, null, 3, 1, 4 });
            var indices = new List<int> { 0, 1, 2, 3, 4 };
            var aggregation = AggregationFunctions.Variance();

            Assert.That(aggregation.Apply(column, indices), Is.EqualTo(2.1875).Within(1e-9));
        }

        [Test]
        public void Apply_WithAllNullValues_ReturnsNull()
        {
            var column = NivaraColumn.CreateFromNullable(new double?[] { null, null });
            var indices = new List<int> { 0, 1 };
            var aggregation = AggregationFunctions.Variance();

            Assert.That(aggregation.Apply(column, indices), Is.Null);
        }

        [Test]
        public void Constructor_NegativeDdof_ThrowsArgumentOutOfRangeException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => AggregationFunctions.Variance(-1));
        }

        [Test]
        public void Name_ReflectsDdof()
        {
            Assert.That(AggregationFunctions.Variance().Name, Is.EqualTo("Variance"));
            Assert.That(AggregationFunctions.Variance(1).Name, Is.EqualTo("Variance(1)"));
        }

        [Test]
        public void GetResultType_ReturnsDouble()
        {
            Assert.That(AggregationFunctions.Variance().GetResultType(typeof(long)), Is.EqualTo(typeof(double)));
        }
    }

    [TestFixture]
    public class AggregationFactoryTests
    {
        [Test]
        public void GetStandardFunctions_ReturnsAllStandardFunctions()
        {
            // Act
            var functions = AggregationFunctions.GetStandardFunctions();

            // Assert
            Assert.That(functions, Has.Count.EqualTo(9));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Count"));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Sum"));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Min"));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Max"));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Mean"));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Median"));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Quantile(0.25)"));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Quantile(0.5)"));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Quantile(0.75)"));
        }

        [Test]
        public void FactoryMethods_CreateCorrectInstances()
        {
            // Act & Assert
            Assert.That(AggregationFunctions.Count(), Is.InstanceOf<CountAggregation>());
            Assert.That(AggregationFunctions.Sum(), Is.InstanceOf<SumAggregation>());
            Assert.That(AggregationFunctions.Min(), Is.InstanceOf<MinAggregation>());
            Assert.That(AggregationFunctions.Max(), Is.InstanceOf<MaxAggregation>());
            Assert.That(AggregationFunctions.Mean(), Is.InstanceOf<MeanAggregation>());
            Assert.That(AggregationFunctions.Median(), Is.InstanceOf<MedianAggregation>());
            Assert.That(AggregationFunctions.Quantile(0.5), Is.InstanceOf<QuantileAggregation>());
        }
    }

    [TestFixture]
    public class ApplyToGroupsTests
    {
        [Test]
        public void ApplyToGroups_WithMultipleGroups_ReturnsCorrectResults()
        {
            // Arrange
            var values = new[] { 10, 20, 30, 40, 50, 60 };
            var column = NivaraColumn<int>.Create(values);

            var groups = new[]
            {
                (GroupKey.FromValues(new object[] { "A" }), (IReadOnlyList<int>)new List<int> { 0, 1, 2 }),
                (GroupKey.FromValues(new object[] { "B" }), (IReadOnlyList<int>)new List<int> { 3, 4, 5 })
            };

            var sumAggregation = AggregationFunctions.Sum();

            // Act
            var result = sumAggregation.ApplyToGroups(column, groups);

            // Assert
            Assert.That(result.Length, Is.EqualTo(2));
            Assert.That(result.GetValue(0), Is.EqualTo(60L)); // 10 + 20 + 30
            Assert.That(result.GetValue(1), Is.EqualTo(150L)); // 40 + 50 + 60
        }

        [Test]
        public void ApplyToGroups_WithUInt64Values_ReturnsTypedColumn()
        {
            var column = NivaraColumn<ulong>.Create(new ulong[] { 10, 20, 30 });
            var groups = new[]
            {
                (GroupKey.FromValues(new object[] { "A" }), (IReadOnlyList<int>)new List<int> { 0, 1 }),
                (GroupKey.FromValues(new object[] { "B" }), (IReadOnlyList<int>)new List<int> { 2 })
            };

            var result = AggregationFunctions.Sum().ApplyToGroups(column, groups);

            Assert.That(result, Is.InstanceOf<NivaraColumn<ulong>>());
            Assert.That(result.GetValue(0), Is.EqualTo(30UL));
            Assert.That(result.GetValue(1), Is.EqualTo(30UL));
        }

        [Test]
        public void ApplyToGroups_WithInt128Values_ReturnsTypedColumn()
        {
            var column = NivaraColumn<Int128>.Create(new Int128[] { 10, 20, 30 });
            var groups = new[]
            {
                (GroupKey.FromValues(new object[] { "A" }), (IReadOnlyList<int>)new List<int> { 0, 1 }),
                (GroupKey.FromValues(new object[] { "B" }), (IReadOnlyList<int>)new List<int> { 2 })
            };

            var result = AggregationFunctions.Sum().ApplyToGroups(column, groups);

            Assert.That(result, Is.InstanceOf<NivaraColumn<Int128>>());
            Assert.That(result.GetValue(0), Is.EqualTo((Int128)30));
            Assert.That(result.GetValue(1), Is.EqualTo((Int128)30));
        }

        [Test]
        public void ApplyToGroups_WithUInt128Values_ReturnsTypedColumn()
        {
            var column = NivaraColumn<UInt128>.Create(new UInt128[] { 10, 20, 30 });
            var groups = new[]
            {
                (GroupKey.FromValues(new object[] { "A" }), (IReadOnlyList<int>)new List<int> { 0, 1 }),
                (GroupKey.FromValues(new object[] { "B" }), (IReadOnlyList<int>)new List<int> { 2 })
            };

            var result = AggregationFunctions.Sum().ApplyToGroups(column, groups);

            Assert.That(result, Is.InstanceOf<NivaraColumn<UInt128>>());
            Assert.That(result.GetValue(0), Is.EqualTo((UInt128)30));
            Assert.That(result.GetValue(1), Is.EqualTo((UInt128)30));
        }

        [Test]
        public void ApplyToGroups_WithHalfValues_ReturnsTypedColumn()
        {
            var column = NivaraColumn<Half>.Create(new Half[] { (Half)10, (Half)20, (Half)30 });
            var groups = new[]
            {
                (GroupKey.FromValues(new object[] { "A" }), (IReadOnlyList<int>)new List<int> { 0, 1 }),
                (GroupKey.FromValues(new object[] { "B" }), (IReadOnlyList<int>)new List<int> { 2 })
            };

            var result = AggregationFunctions.Min().ApplyToGroups(column, groups);

            Assert.That(result, Is.InstanceOf<NivaraColumn<Half>>());
            Assert.That(result.GetValue(0), Is.EqualTo((Half)10));
            Assert.That(result.GetValue(1), Is.EqualTo((Half)30));
        }

        [Test]
        public void ApplyToGroups_WithNIntValues_ReturnsTypedColumn()
        {
            var column = NivaraColumn<nint>.Create(new nint[] { 10, 20, 30 });
            var groups = new[]
            {
                (GroupKey.FromValues(new object[] { "A" }), (IReadOnlyList<int>)new List<int> { 0, 1 }),
                (GroupKey.FromValues(new object[] { "B" }), (IReadOnlyList<int>)new List<int> { 2 })
            };

            var result = AggregationFunctions.Max().ApplyToGroups(column, groups);

            Assert.That(result, Is.InstanceOf<NivaraColumn<nint>>());
            Assert.That(result.GetValue(0), Is.EqualTo((nint)20));
            Assert.That(result.GetValue(1), Is.EqualTo((nint)30));
        }

        [Test]
        public void ApplyToGroups_WithCharValues_ReturnsTypedColumn()
        {
            var column = NivaraColumn<char>.Create(new char[] { 'c', 'a', 'b' });
            var groups = new[]
            {
                (GroupKey.FromValues(new object[] { "A" }), (IReadOnlyList<int>)new List<int> { 0, 1 }),
                (GroupKey.FromValues(new object[] { "B" }), (IReadOnlyList<int>)new List<int> { 2 })
            };

            var result = AggregationFunctions.Min().ApplyToGroups(column, groups);

            Assert.That(result, Is.InstanceOf<NivaraColumn<char>>());
            Assert.That(result.GetValue(0), Is.EqualTo('a'));
            Assert.That(result.GetValue(1), Is.EqualTo('b'));
        }

        [Test]
        public void ApplyToGroups_WithSByteValues_ReturnsTypedColumn()
        {
            var column = NivaraColumn<sbyte>.Create(new sbyte[] { -5, 3, 7 });
            var groups = new[]
            {
                (GroupKey.FromValues(new object[] { "A" }), (IReadOnlyList<int>)new List<int> { 0, 1 }),
                (GroupKey.FromValues(new object[] { "B" }), (IReadOnlyList<int>)new List<int> { 2 })
            };

            var result = AggregationFunctions.Min().ApplyToGroups(column, groups);

            Assert.That(result, Is.InstanceOf<NivaraColumn<sbyte>>());
            Assert.That(result.GetValue(0), Is.EqualTo((sbyte)-5));
            Assert.That(result.GetValue(1), Is.EqualTo((sbyte)7));
        }

        [Test]
        public void ApplyToGroups_WithUShortValues_ReturnsTypedColumn()
        {
            var column = NivaraColumn<ushort>.Create(new ushort[] { 10, 20, 30 });
            var groups = new[]
            {
                (GroupKey.FromValues(new object[] { "A" }), (IReadOnlyList<int>)new List<int> { 0, 1 }),
                (GroupKey.FromValues(new object[] { "B" }), (IReadOnlyList<int>)new List<int> { 2 })
            };

            var result = AggregationFunctions.Max().ApplyToGroups(column, groups);

            Assert.That(result, Is.InstanceOf<NivaraColumn<ushort>>());
            Assert.That(result.GetValue(0), Is.EqualTo((ushort)20));
            Assert.That(result.GetValue(1), Is.EqualTo((ushort)30));
        }

        [Test]
        public void ApplyToGroups_WithUIntValues_ReturnsTypedColumn()
        {
            var column = NivaraColumn<uint>.Create(new uint[] { 10, 20, 30 });
            var groups = new[]
            {
                (GroupKey.FromValues(new object[] { "A" }), (IReadOnlyList<int>)new List<int> { 0, 1 }),
                (GroupKey.FromValues(new object[] { "B" }), (IReadOnlyList<int>)new List<int> { 2 })
            };

            var result = AggregationFunctions.Min().ApplyToGroups(column, groups);

            Assert.That(result, Is.InstanceOf<NivaraColumn<uint>>());
            Assert.That(result.GetValue(0), Is.EqualTo(10u));
            Assert.That(result.GetValue(1), Is.EqualTo(30u));
        }

        [Test]
        public void ApplyToGroups_WithDateTimeOffsetValues_ReturnsTypedColumn()
        {
            var first = new DateTimeOffset(2024, 1, 1, 0, 0, 0, TimeSpan.Zero);
            var second = new DateTimeOffset(2024, 1, 2, 0, 0, 0, TimeSpan.Zero);
            var third = new DateTimeOffset(2024, 1, 3, 0, 0, 0, TimeSpan.Zero);
            var column = NivaraColumn<DateTimeOffset>.Create(new[] { first, second, third });
            var groups = new[]
            {
                (GroupKey.FromValues(new object[] { "A" }), (IReadOnlyList<int>)new List<int> { 0, 1 }),
                (GroupKey.FromValues(new object[] { "B" }), (IReadOnlyList<int>)new List<int> { 2 })
            };

            var result = AggregationFunctions.Min().ApplyToGroups(column, groups);

            Assert.That(result, Is.InstanceOf<NivaraColumn<DateTimeOffset>>());
            Assert.That(result.GetValue(0), Is.EqualTo(first));
            Assert.That(result.GetValue(1), Is.EqualTo(third));
        }
    }
}
