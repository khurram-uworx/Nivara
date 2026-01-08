using NUnit.Framework;
using Nivara;
using System;
using System.Collections.Generic;
using System.Linq;

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
            Assert.That(functions, Has.Count.EqualTo(5));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Count"));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Sum"));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Min"));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Max"));
            Assert.That(functions.Select(f => f.Name), Contains.Item("Mean"));
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
                (new GroupKey("A"), (IReadOnlyList<int>)new List<int> { 0, 1, 2 }),
                (new GroupKey("B"), (IReadOnlyList<int>)new List<int> { 3, 4, 5 })
            };

            var sumAggregation = AggregationFunctions.Sum();

            // Act
            var result = sumAggregation.ApplyToGroups(column, groups);

            // Assert
            Assert.That(result.Length, Is.EqualTo(2));
            Assert.That(result.GetValue(0), Is.EqualTo(60L)); // 10 + 20 + 30
            Assert.That(result.GetValue(1), Is.EqualTo(150L)); // 40 + 50 + 60
        }
    }
}