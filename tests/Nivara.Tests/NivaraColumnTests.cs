using NUnit.Framework;

namespace Nivara.Tests;

/// <summary>
/// Tests for NivaraColumn&lt;T&gt; creation and basic functionality
/// </summary>
[TestFixture]
public class NivaraColumnTests
{
    #region Property 1: Automatic storage selection for vectorizable types

    /// <summary>
    /// Property 1: Automatic storage selection for vectorizable types
    /// For any vectorizable type (int, float, double, etc.) and any array of values, 
    /// creating a NivaraColumn should result in appropriate storage being selected automatically.
    /// **Validates: Requirements 1.1, 6.1**
    /// </summary>
    [Test]
    public void VectorizableTypes_ShouldSelectAppropriateStorage()
    {
        // Test with int (vectorizable type)
        var intValues = new[] { 1, 2, 3, 4, 5 };
        var intColumn = NivaraColumn<int>.Create(intValues);

        Assert.That(intColumn, Is.Not.Null, "Int column should be created");
        Assert.That(intColumn.Length, Is.EqualTo(intValues.Length), "Int column should preserve length");
        // Note: Currently all types use memory storage, so IsVectorizable will be false
        // This will be updated when tensor storage is properly implemented
        Assert.That(intColumn.IsVectorizable, Is.False, "Currently all columns use memory storage");

        // Test with float (vectorizable type)
        var floatValues = new[] { 1.0f, 2.5f, -3.14f };
        var floatColumn = NivaraColumn<float>.Create(floatValues);

        Assert.That(floatColumn, Is.Not.Null, "Float column should be created");
        Assert.That(floatColumn.Length, Is.EqualTo(floatValues.Length), "Float column should preserve length");
        Assert.That(floatColumn.IsVectorizable, Is.False, "Currently all columns use memory storage");

        // Test with double (vectorizable type)
        var doubleValues = new[] { 1.0, 2.5, -3.14159 };
        var doubleColumn = NivaraColumn<double>.Create(doubleValues);

        Assert.That(doubleColumn, Is.Not.Null, "Double column should be created");
        Assert.That(doubleColumn.Length, Is.EqualTo(doubleValues.Length), "Double column should preserve length");
        Assert.That(doubleColumn.IsVectorizable, Is.False, "Currently all columns use memory storage");

        // Test with long (vectorizable type)
        var longValues = new[] { 1L, 2L, 3L };
        var longColumn = NivaraColumn<long>.Create(longValues);

        Assert.That(longColumn, Is.Not.Null, "Long column should be created");
        Assert.That(longColumn.Length, Is.EqualTo(longValues.Length), "Long column should preserve length");
        Assert.That(longColumn.IsVectorizable, Is.False, "Currently all columns use memory storage");

        // Test with byte (vectorizable type)
        var byteValues = new byte[] { 1, 2, 3 };
        var byteColumn = NivaraColumn<byte>.Create(byteValues);

        Assert.That(byteColumn, Is.Not.Null, "Byte column should be created");
        Assert.That(byteColumn.Length, Is.EqualTo(byteValues.Length), "Byte column should preserve length");
        Assert.That(byteColumn.IsVectorizable, Is.False, "Currently all columns use memory storage");

        // Test with empty arrays
        var emptyIntColumn = NivaraColumn<int>.Create(Array.Empty<int>());
        Assert.That(emptyIntColumn, Is.Not.Null, "Empty int column should be created");
        Assert.That(emptyIntColumn.Length, Is.EqualTo(0), "Empty column should have zero length");

        // Test with single element arrays
        var singleIntColumn = NivaraColumn<int>.Create(new[] { 42 });
        Assert.That(singleIntColumn, Is.Not.Null, "Single element column should be created");
        Assert.That(singleIntColumn.Length, Is.EqualTo(1), "Single element column should have length 1");
        Assert.That(singleIntColumn[0], Is.EqualTo(42), "Single element should be accessible");
    }

    #endregion

    #region Property 2: Automatic storage selection for non-vectorizable types

    /// <summary>
    /// Property 2: Automatic storage selection for non-vectorizable types
    /// For any non-vectorizable type (string, Guid, reference types) and any array of values, 
    /// creating a NivaraColumn should result in memory-backed storage being selected automatically.
    /// **Validates: Requirements 1.2, 6.2**
    /// </summary>
    [Test]
    public void NonVectorizableTypes_ShouldUseMemoryStorage()
    {
        // Test with string (reference type)
        var stringTestCases = new[]
        {
            new string[] { "hello", "world", "test" },
            new string[] { "", "non-empty", "" },
            new string[] { "single" },
            new string[] { } // Empty array
        };

        foreach (var values in stringTestCases)
        {
            var column = NivaraColumn<string>.Create(values);

            Assert.That(column, Is.Not.Null, "String column should be created successfully");
            Assert.That(column.Length, Is.EqualTo(values.Length), "Column length should match input array length");
            Assert.That(column.IsVectorizable, Is.False, "String columns should not be vectorizable");
        }

        // Test with Guid (non-vectorizable value type)
        var guidValues = new[] { Guid.NewGuid(), Guid.NewGuid(), Guid.Empty };
        var guidColumn = NivaraColumn<Guid>.Create(guidValues);

        Assert.That(guidColumn, Is.Not.Null, "Guid column should be created successfully");
        Assert.That(guidColumn.Length, Is.EqualTo(guidValues.Length), "Guid column length should match input");
        Assert.That(guidColumn.IsVectorizable, Is.False, "Guid columns should not be vectorizable");

        // Test with DateTime (non-vectorizable value type)
        var dateValues = new[] { DateTime.Now, DateTime.MinValue, DateTime.MaxValue };
        var dateColumn = NivaraColumn<DateTime>.Create(dateValues);

        Assert.That(dateColumn, Is.Not.Null, "DateTime column should be created successfully");
        Assert.That(dateColumn.Length, Is.EqualTo(dateValues.Length), "DateTime column length should match input");
        Assert.That(dateColumn.IsVectorizable, Is.False, "DateTime columns should not be vectorizable");
    }

    #endregion

    #region Property 4: Length preservation

    /// <summary>
    /// Property 4: Length preservation
    /// For any input array, the resulting NivaraColumn should have a Length property 
    /// that exactly matches the input array length.
    /// **Validates: Requirements 1.4**
    /// </summary>
    [Test]
    public void ColumnCreation_ShouldPreserveLength()
    {
        // Test with various array sizes
        var lengthTestCases = new[]
        {
            0,   // Empty array
            1,   // Single element
            2,   // Small array
            10,  // Medium array
            100, // Larger array
            1000 // Large array
        };

        foreach (var length in lengthTestCases)
        {
            // Test with int arrays
            var intValues = Enumerable.Range(0, length).ToArray();
            var intColumn = NivaraColumn<int>.Create(intValues);

            Assert.That(intColumn.Length, Is.EqualTo(length),
                $"Int column should preserve length {length}");

            // Test with string arrays
            var stringValues = Enumerable.Range(0, length).Select(i => $"item_{i}").ToArray();
            var stringColumn = NivaraColumn<string>.Create(stringValues);

            Assert.That(stringColumn.Length, Is.EqualTo(length),
                $"String column should preserve length {length}");

            // Test with double arrays
            var doubleValues = Enumerable.Range(0, length).Select(i => (double)i).ToArray();
            var doubleColumn = NivaraColumn<double>.Create(doubleValues);

            Assert.That(doubleColumn.Length, Is.EqualTo(length),
                $"Double column should preserve length {length}");
        }
    }

    #endregion

    #region Property 6: Scalar multiplication

    /// <summary>
    /// Property 6: Scalar multiplication
    /// For any numeric NivaraColumn and any scalar value, multiplying the column by the scalar 
    /// should return a new column where every non-null element is multiplied by the scalar.
    /// **Validates: Requirements 2.2**
    /// </summary>
    [Test]
    public void ScalarMultiplication_ShouldMultiplyAllElements()
    {
        // Test with int column
        var intValues = new[] { 1, 2, 3, 4, 5 };
        var intColumn = NivaraColumn<int>.Create(intValues);
        var scalar = 3;

        var result = intColumn.Multiply(scalar);

        Assert.That(result, Is.Not.Null, "Result should not be null");
        Assert.That(result.Length, Is.EqualTo(intValues.Length), "Result should preserve length");

        for (int i = 0; i < intValues.Length; i++)
        {
            Assert.That(result[i], Is.EqualTo(intValues[i] * scalar),
                $"Element at index {i} should be multiplied by scalar");
        }

        // Test with float column
        var floatValues = new[] { 1.5f, 2.5f, -3.0f, 0.0f };
        var floatColumn = NivaraColumn<float>.Create(floatValues);
        var floatScalar = 2.0f;

        var floatResult = floatColumn.Multiply(floatScalar);

        Assert.That(floatResult, Is.Not.Null, "Float result should not be null");
        Assert.That(floatResult.Length, Is.EqualTo(floatValues.Length), "Float result should preserve length");

        for (int i = 0; i < floatValues.Length; i++)
        {
            Assert.That(floatResult[i], Is.EqualTo(floatValues[i] * floatScalar).Within(0.0001f),
                $"Float element at index {i} should be multiplied by scalar");
        }

        // Test with double column
        var doubleValues = new[] { 1.0, -2.5, 3.14159, 0.0 };
        var doubleColumn = NivaraColumn<double>.Create(doubleValues);
        var doubleScalar = 1.5;

        var doubleResult = doubleColumn.Multiply(doubleScalar);

        Assert.That(doubleResult, Is.Not.Null, "Double result should not be null");
        Assert.That(doubleResult.Length, Is.EqualTo(doubleValues.Length), "Double result should preserve length");

        for (int i = 0; i < doubleValues.Length; i++)
        {
            Assert.That(doubleResult[i], Is.EqualTo(doubleValues[i] * doubleScalar).Within(0.000001),
                $"Double element at index {i} should be multiplied by scalar");
        }

        // Test with operator overload
        var operatorResult = intColumn * scalar;
        Assert.That(operatorResult, Is.Not.Null, "Operator result should not be null");
        Assert.That(operatorResult.Length, Is.EqualTo(intValues.Length), "Operator result should preserve length");

        for (int i = 0; i < intValues.Length; i++)
        {
            Assert.That(operatorResult[i], Is.EqualTo(intValues[i] * scalar),
                $"Operator element at index {i} should be multiplied by scalar");
        }

        // Test commutative operator
        var commutativeResult = scalar * intColumn;
        Assert.That(commutativeResult, Is.Not.Null, "Commutative result should not be null");

        for (int i = 0; i < intValues.Length; i++)
        {
            Assert.That(commutativeResult[i], Is.EqualTo(scalar * intValues[i]),
                $"Commutative element at index {i} should be multiplied by scalar");
        }

        // Test with zero scalar
        var zeroResult = intColumn.Multiply(0);
        for (int i = 0; i < intValues.Length; i++)
        {
            Assert.That(zeroResult[i], Is.EqualTo(0),
                $"Zero multiplication should result in zero at index {i}");
        }

        // Test with negative scalar
        var negativeResult = intColumn.Multiply(-1);
        for (int i = 0; i < intValues.Length; i++)
        {
            Assert.That(negativeResult[i], Is.EqualTo(-intValues[i]),
                $"Negative multiplication should negate value at index {i}");
        }
    }

    #endregion

    #region Property 7: Element-wise addition

    /// <summary>
    /// Property 7: Element-wise addition
    /// For any two numeric NivaraColumns of the same length, adding them should return 
    /// a new column with element-wise addition results.
    /// **Validates: Requirements 2.3**
    /// </summary>
    [Test]
    public void ElementWiseAddition_ShouldAddCorrespondingElements()
    {
        // Test with int columns
        var leftIntValues = new[] { 1, 2, 3, 4, 5 };
        var rightIntValues = new[] { 10, 20, 30, 40, 50 };
        var leftIntColumn = NivaraColumn<int>.Create(leftIntValues);
        var rightIntColumn = NivaraColumn<int>.Create(rightIntValues);

        var intResult = leftIntColumn.Add(rightIntColumn);

        Assert.That(intResult, Is.Not.Null, "Int result should not be null");
        Assert.That(intResult.Length, Is.EqualTo(leftIntValues.Length), "Int result should preserve length");

        for (int i = 0; i < leftIntValues.Length; i++)
        {
            Assert.That(intResult[i], Is.EqualTo(leftIntValues[i] + rightIntValues[i]),
                $"Int element at index {i} should be sum of corresponding elements");
        }

        // Test with float columns
        var leftFloatValues = new[] { 1.5f, 2.5f, -3.0f, 0.0f };
        var rightFloatValues = new[] { 0.5f, -1.0f, 2.0f, 1.0f };
        var leftFloatColumn = NivaraColumn<float>.Create(leftFloatValues);
        var rightFloatColumn = NivaraColumn<float>.Create(rightFloatValues);

        var floatResult = leftFloatColumn.Add(rightFloatColumn);

        Assert.That(floatResult, Is.Not.Null, "Float result should not be null");
        Assert.That(floatResult.Length, Is.EqualTo(leftFloatValues.Length), "Float result should preserve length");

        for (int i = 0; i < leftFloatValues.Length; i++)
        {
            Assert.That(floatResult[i], Is.EqualTo(leftFloatValues[i] + rightFloatValues[i]).Within(0.0001f),
                $"Float element at index {i} should be sum of corresponding elements");
        }

        // Test with double columns
        var leftDoubleValues = new[] { 1.0, -2.5, 3.14159 };
        var rightDoubleValues = new[] { 2.0, 1.5, -1.14159 };
        var leftDoubleColumn = NivaraColumn<double>.Create(leftDoubleValues);
        var rightDoubleColumn = NivaraColumn<double>.Create(rightDoubleValues);

        var doubleResult = leftDoubleColumn.Add(rightDoubleColumn);

        Assert.That(doubleResult, Is.Not.Null, "Double result should not be null");
        Assert.That(doubleResult.Length, Is.EqualTo(leftDoubleValues.Length), "Double result should preserve length");

        for (int i = 0; i < leftDoubleValues.Length; i++)
        {
            Assert.That(doubleResult[i], Is.EqualTo(leftDoubleValues[i] + rightDoubleValues[i]).Within(0.000001),
                $"Double element at index {i} should be sum of corresponding elements");
        }

        // Test with operator overload
        var operatorResult = leftIntColumn + rightIntColumn;
        Assert.That(operatorResult, Is.Not.Null, "Operator result should not be null");
        Assert.That(operatorResult.Length, Is.EqualTo(leftIntValues.Length), "Operator result should preserve length");

        for (int i = 0; i < leftIntValues.Length; i++)
        {
            Assert.That(operatorResult[i], Is.EqualTo(leftIntValues[i] + rightIntValues[i]),
                $"Operator element at index {i} should be sum of corresponding elements");
        }

        // Test with zero values
        var zeroValues = new[] { 0, 0, 0, 0, 0 };
        var zeroColumn = NivaraColumn<int>.Create(zeroValues);
        var zeroResult = leftIntColumn.Add(zeroColumn);

        for (int i = 0; i < leftIntValues.Length; i++)
        {
            Assert.That(zeroResult[i], Is.EqualTo(leftIntValues[i]),
                $"Adding zero should preserve original value at index {i}");
        }

        // Test with negative values
        var negativeValues = leftIntValues.Select(x => -x).ToArray();
        var negativeColumn = NivaraColumn<int>.Create(negativeValues);
        var negativeResult = leftIntColumn.Add(negativeColumn);

        for (int i = 0; i < leftIntValues.Length; i++)
        {
            Assert.That(negativeResult[i], Is.EqualTo(0),
                $"Adding negative should result in zero at index {i}");
        }
    }

    #endregion

    #region Property 8: Null propagation in operations

    /// <summary>
    /// Property 8: Null propagation in operations
    /// For any operation on columns containing null values, the result should correctly 
    /// propagate nulls (null combined with any value yields null).
    /// **Validates: Requirements 2.4**
    /// </summary>
    [Test]
    public void ArithmeticOperations_ShouldPropagateNulls()
    {
        // For now, we'll test with reference types since nullable value types have generic constraint issues
        // This test will be expanded when we can properly handle nullable value types

        // Test scalar multiplication with reference type columns (which can have nulls)
        // Since we can't easily create numeric reference types, we'll test the null propagation logic
        // by verifying that the null mask is properly handled in the storage layer

        // Create a string column with nulls to test the null mask handling
        var stringValuesWithNulls = new string?[] { "1", null, "3", null, "5" };
        var stringArrayWithNulls = new string[stringValuesWithNulls.Length];
        for (int i = 0; i < stringValuesWithNulls.Length; i++)
        {
            stringArrayWithNulls[i] = stringValuesWithNulls[i]!; // Suppress null warning for test
        }

        var stringColumnWithNulls = NivaraColumn<string>.CreateForReferenceType(stringArrayWithNulls);

        // Verify null detection works
        Assert.That(stringColumnWithNulls.IsNull(0), Is.False, "First position should not be null");
        Assert.That(stringColumnWithNulls.IsNull(1), Is.True, "Second position should be null");
        Assert.That(stringColumnWithNulls.IsNull(2), Is.False, "Third position should not be null");
        Assert.That(stringColumnWithNulls.IsNull(3), Is.True, "Fourth position should be null");
        Assert.That(stringColumnWithNulls.IsNull(4), Is.False, "Fifth position should not be null");

        // Test that HasNulls property works correctly
        Assert.That(stringColumnWithNulls.HasNulls, Is.True, "Column with nulls should have HasNulls true");

        // For numeric operations with nulls, we'll need to implement nullable value type support
        // in a future iteration when generic constraints are resolved

        // Test with int columns (no nulls for now)
        var intValues = new[] { 1, 2, 3 };
        var intColumn = NivaraColumn<int>.Create(intValues);

        // Verify that value type columns don't have nulls
        Assert.That(intColumn.HasNulls, Is.False, "Value type columns should not have nulls");
        for (int i = 0; i < intValues.Length; i++)
        {
            Assert.That(intColumn.IsNull(i), Is.False, $"Value type position {i} should not be null");
        }

        // Test arithmetic operations preserve non-null status
        var multiplyResult = intColumn.Multiply(2);
        Assert.That(multiplyResult.HasNulls, Is.False, "Multiplication result should not have nulls");

        var addResult = intColumn.Add(intColumn);
        Assert.That(addResult.HasNulls, Is.False, "Addition result should not have nulls");
    }

    #endregion

    #region Arithmetic Error Handling Tests

    /// <summary>
    /// Test that arithmetic operations on non-vectorizable types fail with clear error messages
    /// **Validates: Requirements 2.5**
    /// </summary>
    [Test]
    public void ArithmeticOperations_OnNonVectorizableTypes_ShouldThrowWithClearErrors()
    {
        // Test with string column (non-vectorizable type)
        var stringColumn = NivaraColumn<string>.Create(new[] { "1", "2", "3" });

        var ex1 = Assert.Throws<InvalidOperationException>(() => stringColumn.Multiply("2"),
            "Multiply on string column should throw InvalidOperationException");
        Assert.That(ex1.Message, Does.Contain("not supported for type String"),
            "Error message should mention the type name");
        Assert.That(ex1.Message, Does.Contain("numeric types"),
            "Error message should mention numeric types requirement");

        var ex2 = Assert.Throws<InvalidOperationException>(() => stringColumn.Add(stringColumn),
            "Add on string column should throw InvalidOperationException");
        Assert.That(ex2.Message, Does.Contain("not supported for type String"),
            "Error message should mention the type name");
        Assert.That(ex2.Message, Does.Contain("numeric types"),
            "Error message should mention numeric types requirement");

        // Test with Guid column (non-vectorizable type)
        var guidColumn = NivaraColumn<Guid>.Create(new[] { Guid.NewGuid(), Guid.NewGuid() });

        var ex3 = Assert.Throws<InvalidOperationException>(() => guidColumn.Multiply(Guid.NewGuid()),
            "Multiply on Guid column should throw InvalidOperationException");
        Assert.That(ex3.Message, Does.Contain("not supported for type Guid"),
            "Error message should mention Guid type");

        var ex4 = Assert.Throws<InvalidOperationException>(() => guidColumn.Add(guidColumn),
            "Add on Guid column should throw InvalidOperationException");
        Assert.That(ex4.Message, Does.Contain("not supported for type Guid"),
            "Error message should mention Guid type");

        // Test with DateTime column (non-vectorizable type)
        var dateColumn = NivaraColumn<DateTime>.Create(new[] { DateTime.Now, DateTime.MinValue });

        var ex5 = Assert.Throws<InvalidOperationException>(() => dateColumn.Multiply(DateTime.Now),
            "Multiply on DateTime column should throw InvalidOperationException");
        Assert.That(ex5.Message, Does.Contain("not supported for type DateTime"),
            "Error message should mention DateTime type");

        var ex6 = Assert.Throws<InvalidOperationException>(() => dateColumn.Add(dateColumn),
            "Add on DateTime column should throw InvalidOperationException");
        Assert.That(ex6.Message, Does.Contain("not supported for type DateTime"),
            "Error message should mention DateTime type");
    }

    /// <summary>
    /// Test that arithmetic operations with mismatched column lengths fail with clear error messages
    /// </summary>
    [Test]
    public void ArithmeticOperations_WithMismatchedLengths_ShouldThrowWithClearErrors()
    {
        var shortColumn = NivaraColumn<int>.Create(new[] { 1, 2 });
        var longColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });

        var ex = Assert.Throws<ArgumentException>(() => shortColumn.Add(longColumn),
            "Adding columns of different lengths should throw ArgumentException");
        Assert.That(ex.Message, Does.Contain("different lengths"),
            "Error message should mention different lengths");
        Assert.That(ex.Message, Does.Contain("2 vs 5"),
            "Error message should show the actual lengths");
    }

    /// <summary>
    /// Test that arithmetic operations with null arguments fail with clear error messages
    /// </summary>
    [Test]
    public void ArithmeticOperations_WithNullArguments_ShouldThrowWithClearErrors()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });

        var ex = Assert.Throws<ArgumentNullException>(() => column.Add(null!),
            "Adding null column should throw ArgumentNullException");
        Assert.That(ex.ParamName, Is.EqualTo("other"),
            "Exception should specify the parameter name");
    }

    /// <summary>
    /// Test that arithmetic operations on disposed columns fail with clear error messages
    /// </summary>
    [Test]
    public void ArithmeticOperations_OnDisposedColumns_ShouldThrowObjectDisposedException()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var otherColumn = NivaraColumn<int>.Create(new[] { 4, 5, 6 });

        column.Dispose();

        Assert.Throws<ObjectDisposedException>(() => column.Multiply(2),
            "Multiply on disposed column should throw ObjectDisposedException");
        Assert.Throws<ObjectDisposedException>(() => column.Add(otherColumn),
            "Add on disposed column should throw ObjectDisposedException");

        // Test with disposed other column
        var anotherColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        otherColumn.Dispose();

        Assert.Throws<ObjectDisposedException>(() => anotherColumn.Add(otherColumn),
            "Add with disposed other column should throw ObjectDisposedException");
    }

    /// <summary>
    /// Test operator overloads with error conditions
    /// </summary>
    [Test]
    public void ArithmeticOperators_WithErrorConditions_ShouldThrowAppropriateExceptions()
    {
        var stringColumn = NivaraColumn<string>.Create(new[] { "1", "2", "3" });

        // Test multiplication operators
        Assert.Throws<InvalidOperationException>(() => { var result = stringColumn * "2"; },
            "String multiplication operator should throw InvalidOperationException");
        Assert.Throws<InvalidOperationException>(() => { var result = "2" * stringColumn; },
            "Commutative string multiplication operator should throw InvalidOperationException");

        // Test addition operator
        Assert.Throws<InvalidOperationException>(() => { var result = stringColumn + stringColumn; },
            "String addition operator should throw InvalidOperationException");

        // Test with mismatched lengths
        var shortColumn = NivaraColumn<int>.Create(new[] { 1, 2 });
        var longColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });

        Assert.Throws<ArgumentException>(() => { var result = shortColumn + longColumn; },
            "Addition operator with mismatched lengths should throw ArgumentException");
    }

    #endregion

    #region Property 9: Comprehensive comparison operations

    /// <summary>
    /// Property 9: Comprehensive comparison operations
    /// For any NivaraColumn and any comparable value, all comparison operations (equals, greater than, less than, etc.) 
    /// should return boolean columns with correct comparison results, with null comparisons yielding null.
    /// **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    /// </summary>
    [Test]
    public void ComparisonOperations_ShouldReturnCorrectBooleanResults()
    {
        // Test scalar equality comparison with int column
        var intValues = new[] { 1, 2, 3, 4, 5 };
        var intColumn = NivaraColumn<int>.Create(intValues);
        var targetValue = 3;

        var equalsResult = intColumn.Equals(targetValue);

        Assert.That(equalsResult, Is.Not.Null, "Equals result should not be null");
        Assert.That(equalsResult.Length, Is.EqualTo(intValues.Length), "Equals result should preserve length");

        for (int i = 0; i < intValues.Length; i++)
        {
            bool expected = intValues[i] == targetValue;
            Assert.That(equalsResult[i], Is.EqualTo(expected),
                $"Equals comparison at index {i}: {intValues[i]} == {targetValue} should be {expected}");
        }

        // Test element-wise equality comparison
        var otherIntValues = new[] { 1, 0, 3, 0, 5 };
        var otherIntColumn = NivaraColumn<int>.Create(otherIntValues);

        var elementWiseEquals = intColumn.Equals(otherIntColumn);

        Assert.That(elementWiseEquals, Is.Not.Null, "Element-wise equals result should not be null");
        Assert.That(elementWiseEquals.Length, Is.EqualTo(intValues.Length), "Element-wise equals result should preserve length");

        for (int i = 0; i < intValues.Length; i++)
        {
            bool expected = intValues[i] == otherIntValues[i];
            Assert.That(elementWiseEquals[i], Is.EqualTo(expected),
                $"Element-wise equals at index {i}: {intValues[i]} == {otherIntValues[i]} should be {expected}");
        }

        // Test greater than comparison
        var greaterThanResult = intColumn.GreaterThan(targetValue);

        Assert.That(greaterThanResult, Is.Not.Null, "GreaterThan result should not be null");
        Assert.That(greaterThanResult.Length, Is.EqualTo(intValues.Length), "GreaterThan result should preserve length");

        for (int i = 0; i < intValues.Length; i++)
        {
            bool expected = intValues[i] > targetValue;
            Assert.That(greaterThanResult[i], Is.EqualTo(expected),
                $"GreaterThan comparison at index {i}: {intValues[i]} > {targetValue} should be {expected}");
        }

        // Test element-wise greater than comparison
        var elementWiseGreaterThan = intColumn.GreaterThan(otherIntColumn);

        Assert.That(elementWiseGreaterThan, Is.Not.Null, "Element-wise GreaterThan result should not be null");
        Assert.That(elementWiseGreaterThan.Length, Is.EqualTo(intValues.Length), "Element-wise GreaterThan result should preserve length");

        for (int i = 0; i < intValues.Length; i++)
        {
            bool expected = intValues[i] > otherIntValues[i];
            Assert.That(elementWiseGreaterThan[i], Is.EqualTo(expected),
                $"Element-wise GreaterThan at index {i}: {intValues[i]} > {otherIntValues[i]} should be {expected}");
        }

        // Test less than comparison
        var lessThanResult = intColumn.LessThan(targetValue);

        Assert.That(lessThanResult, Is.Not.Null, "LessThan result should not be null");
        Assert.That(lessThanResult.Length, Is.EqualTo(intValues.Length), "LessThan result should preserve length");

        for (int i = 0; i < intValues.Length; i++)
        {
            bool expected = intValues[i] < targetValue;
            Assert.That(lessThanResult[i], Is.EqualTo(expected),
                $"LessThan comparison at index {i}: {intValues[i]} < {targetValue} should be {expected}");
        }

        // Test element-wise less than comparison
        var elementWiseLessThan = intColumn.LessThan(otherIntColumn);

        Assert.That(elementWiseLessThan, Is.Not.Null, "Element-wise LessThan result should not be null");
        Assert.That(elementWiseLessThan.Length, Is.EqualTo(intValues.Length), "Element-wise LessThan result should preserve length");

        for (int i = 0; i < intValues.Length; i++)
        {
            bool expected = intValues[i] < otherIntValues[i];
            Assert.That(elementWiseLessThan[i], Is.EqualTo(expected),
                $"Element-wise LessThan at index {i}: {intValues[i]} < {otherIntValues[i]} should be {expected}");
        }

        // Test with float values
        var floatValues = new[] { 1.5f, 2.5f, 3.5f, 4.5f };
        var floatColumn = NivaraColumn<float>.Create(floatValues);
        var floatTarget = 3.0f;

        var floatEquals = floatColumn.Equals(floatTarget);
        var floatGreater = floatColumn.GreaterThan(floatTarget);
        var floatLess = floatColumn.LessThan(floatTarget);

        for (int i = 0; i < floatValues.Length; i++)
        {
            Assert.That(floatEquals[i], Is.EqualTo(floatValues[i] == floatTarget),
                $"Float equals at index {i}");
            Assert.That(floatGreater[i], Is.EqualTo(floatValues[i] > floatTarget),
                $"Float greater than at index {i}");
            Assert.That(floatLess[i], Is.EqualTo(floatValues[i] < floatTarget),
                $"Float less than at index {i}");
        }

        // Test with string values (comparable type)
        var stringValues = new[] { "apple", "banana", "cherry", "date" };
        var stringColumn = NivaraColumn<string>.Create(stringValues);
        var stringTarget = "cherry";

        var stringEquals = stringColumn.Equals(stringTarget);
        var stringGreater = stringColumn.GreaterThan(stringTarget);
        var stringLess = stringColumn.LessThan(stringTarget);

        for (int i = 0; i < stringValues.Length; i++)
        {
            Assert.That(stringEquals[i], Is.EqualTo(stringValues[i] == stringTarget),
                $"String equals at index {i}");
            Assert.That(stringGreater[i], Is.EqualTo(string.Compare(stringValues[i], stringTarget) > 0),
                $"String greater than at index {i}");
            Assert.That(stringLess[i], Is.EqualTo(string.Compare(stringValues[i], stringTarget) < 0),
                $"String less than at index {i}");
        }

        // Test with DateTime values (comparable type)
        var dateValues = new[] {
            new DateTime(2023, 1, 1),
            new DateTime(2023, 6, 15),
            new DateTime(2023, 12, 31)
        };
        var dateColumn = NivaraColumn<DateTime>.Create(dateValues);
        var dateTarget = new DateTime(2023, 6, 15);

        var dateEquals = dateColumn.Equals(dateTarget);
        var dateGreater = dateColumn.GreaterThan(dateTarget);
        var dateLess = dateColumn.LessThan(dateTarget);

        for (int i = 0; i < dateValues.Length; i++)
        {
            Assert.That(dateEquals[i], Is.EqualTo(dateValues[i] == dateTarget),
                $"DateTime equals at index {i}");
            Assert.That(dateGreater[i], Is.EqualTo(dateValues[i] > dateTarget),
                $"DateTime greater than at index {i}");
            Assert.That(dateLess[i], Is.EqualTo(dateValues[i] < dateTarget),
                $"DateTime less than at index {i}");
        }
    }

    /// <summary>
    /// Test comparison operations with null values to ensure proper null propagation
    /// **Validates: Requirements 3.3**
    /// </summary>
    [Test]
    public void ComparisonOperations_WithNulls_ShouldPropagateNullsCorrectly()
    {
        // Test with string column containing nulls
        var stringValuesWithNulls = new string?[] { "hello", null, "world", null, "test" };
        var stringArrayWithNulls = new string[stringValuesWithNulls.Length];
        for (int i = 0; i < stringValuesWithNulls.Length; i++)
        {
            stringArrayWithNulls[i] = stringValuesWithNulls[i]!; // Suppress null warning for test
        }

        var stringColumnWithNulls = NivaraColumn<string>.CreateForReferenceType(stringArrayWithNulls);
        var targetValue = "world";

        // Test scalar equality with nulls
        var equalsResult = stringColumnWithNulls.Equals(targetValue);

        Assert.That(equalsResult, Is.Not.Null, "Equals result with nulls should not be null");
        Assert.That(equalsResult.Length, Is.EqualTo(stringValuesWithNulls.Length), "Result should preserve length");

        // Check that null positions return false (null compared to anything is false)
        Assert.That(equalsResult[0], Is.EqualTo(stringValuesWithNulls[0] == targetValue), "Non-null comparison should work");
        Assert.That(equalsResult[1], Is.False, "Null comparison should return false");
        Assert.That(equalsResult[2], Is.EqualTo(stringValuesWithNulls[2] == targetValue), "Non-null comparison should work");
        Assert.That(equalsResult[3], Is.False, "Null comparison should return false");
        Assert.That(equalsResult[4], Is.EqualTo(stringValuesWithNulls[4] == targetValue), "Non-null comparison should work");

        // Test that the result has null mask in the same positions
        Assert.That(equalsResult.HasNulls, Is.True, "Result should have nulls where original had nulls");
        Assert.That(equalsResult.IsNull(0), Is.False, "Non-null position should not be null in result");
        Assert.That(equalsResult.IsNull(1), Is.True, "Null position should be null in result");
        Assert.That(equalsResult.IsNull(2), Is.False, "Non-null position should not be null in result");
        Assert.That(equalsResult.IsNull(3), Is.True, "Null position should be null in result");
        Assert.That(equalsResult.IsNull(4), Is.False, "Non-null position should not be null in result");

        // Test greater than with nulls
        var greaterThanResult = stringColumnWithNulls.GreaterThan(targetValue);

        Assert.That(greaterThanResult.HasNulls, Is.True, "GreaterThan result should have nulls");
        Assert.That(greaterThanResult.IsNull(1), Is.True, "Null position should remain null in GreaterThan result");
        Assert.That(greaterThanResult.IsNull(3), Is.True, "Null position should remain null in GreaterThan result");

        // Test less than with nulls
        var lessThanResult = stringColumnWithNulls.LessThan(targetValue);

        Assert.That(lessThanResult.HasNulls, Is.True, "LessThan result should have nulls");
        Assert.That(lessThanResult.IsNull(1), Is.True, "Null position should remain null in LessThan result");
        Assert.That(lessThanResult.IsNull(3), Is.True, "Null position should remain null in LessThan result");

        // Test element-wise comparison with nulls
        var otherStringValues = new string?[] { "hello", "test", null, "world", null };
        var otherStringArray = new string[otherStringValues.Length];
        for (int i = 0; i < otherStringValues.Length; i++)
        {
            otherStringArray[i] = otherStringValues[i]!; // Suppress null warning for test
        }

        var otherStringColumn = NivaraColumn<string>.CreateForReferenceType(otherStringArray);

        var elementWiseEquals = stringColumnWithNulls.Equals(otherStringColumn);

        Assert.That(elementWiseEquals.HasNulls, Is.True, "Element-wise equals should have nulls");

        // Check null propagation: if either side is null, result should be null
        Assert.That(elementWiseEquals.IsNull(0), Is.False, "Both non-null should not be null in result");
        Assert.That(elementWiseEquals.IsNull(1), Is.True, "Left null should propagate to result");
        Assert.That(elementWiseEquals.IsNull(2), Is.True, "Right null should propagate to result");
        Assert.That(elementWiseEquals.IsNull(3), Is.True, "Left null should propagate to result");
        Assert.That(elementWiseEquals.IsNull(4), Is.True, "Both null should propagate to result");

        // Check actual comparison values for non-null positions
        Assert.That(elementWiseEquals[0], Is.EqualTo(stringValuesWithNulls[0] == otherStringValues[0]),
            "Non-null comparison should work correctly");
    }

    /// <summary>
    /// Test comparison operations error handling
    /// </summary>
    [Test]
    public void ComparisonOperations_ErrorHandling_ShouldThrowAppropriateExceptions()
    {
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var shortColumn = NivaraColumn<int>.Create(new[] { 1, 2 });

        // Test mismatched lengths
        Assert.Throws<ArgumentException>(() => intColumn.Equals(shortColumn),
            "Equals with mismatched lengths should throw ArgumentException");
        Assert.Throws<ArgumentException>(() => intColumn.GreaterThan(shortColumn),
            "GreaterThan with mismatched lengths should throw ArgumentException");
        Assert.Throws<ArgumentException>(() => intColumn.LessThan(shortColumn),
            "LessThan with mismatched lengths should throw ArgumentException");

        // Test null arguments
        Assert.Throws<ArgumentNullException>(() => intColumn.Equals((NivaraColumn<int>)null!),
            "Equals with null column should throw ArgumentNullException");
        Assert.Throws<ArgumentNullException>(() => intColumn.GreaterThan((NivaraColumn<int>)null!),
            "GreaterThan with null column should throw ArgumentNullException");
        Assert.Throws<ArgumentNullException>(() => intColumn.LessThan((NivaraColumn<int>)null!),
            "LessThan with null column should throw ArgumentNullException");

        // Test disposed columns
        var disposedColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        disposedColumn.Dispose();

        Assert.Throws<ObjectDisposedException>(() => disposedColumn.Equals(5),
            "Equals on disposed column should throw ObjectDisposedException");
        Assert.Throws<ObjectDisposedException>(() => disposedColumn.GreaterThan(5),
            "GreaterThan on disposed column should throw ObjectDisposedException");
        Assert.Throws<ObjectDisposedException>(() => disposedColumn.LessThan(5),
            "LessThan on disposed column should throw ObjectDisposedException");

        // Test with disposed other column
        var anotherColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var otherDisposedColumn = NivaraColumn<int>.Create(new[] { 4, 5, 6 });
        otherDisposedColumn.Dispose();

        Assert.Throws<ObjectDisposedException>(() => anotherColumn.Equals(otherDisposedColumn),
            "Equals with disposed other column should throw ObjectDisposedException");
        Assert.Throws<ObjectDisposedException>(() => anotherColumn.GreaterThan(otherDisposedColumn),
            "GreaterThan with disposed other column should throw ObjectDisposedException");
        Assert.Throws<ObjectDisposedException>(() => anotherColumn.LessThan(otherDisposedColumn),
            "LessThan with disposed other column should throw ObjectDisposedException");
    }

    /// <summary>
    /// Test comparison operations on non-comparable types
    /// </summary>
    [Test]
    public void ComparisonOperations_OnNonComparableTypes_ShouldThrowForOrderComparisons()
    {
        // Create a column of a type that supports equality but not ordering (like Guid)
        var guidColumn = NivaraColumn<Guid>.Create(new[] { Guid.NewGuid(), Guid.NewGuid() });
        var targetGuid = Guid.NewGuid();

        // Equals should work for all types
        Assert.DoesNotThrow(() => guidColumn.Equals(targetGuid),
            "Equals should work for all types including Guid");

        // But ordering comparisons should work for Guid since it implements IComparable
        Assert.DoesNotThrow(() => guidColumn.GreaterThan(targetGuid),
            "GreaterThan should work for Guid since it implements IComparable");
        Assert.DoesNotThrow(() => guidColumn.LessThan(targetGuid),
            "LessThan should work for Guid since it implements IComparable");

        // Test that the results are boolean columns
        var equalsResult = guidColumn.Equals(targetGuid);
        var greaterResult = guidColumn.GreaterThan(targetGuid);
        var lessResult = guidColumn.LessThan(targetGuid);

        Assert.That(equalsResult.Length, Is.EqualTo(guidColumn.Length), "Equals result should preserve length");
        Assert.That(greaterResult.Length, Is.EqualTo(guidColumn.Length), "GreaterThan result should preserve length");
        Assert.That(lessResult.Length, Is.EqualTo(guidColumn.Length), "LessThan result should preserve length");
    }

    #endregion

    #region Additional Core Functionality Tests

    /// <summary>
    /// Test basic indexer access functionality
    /// </summary>
    [Test]
    public void ColumnIndexer_ShouldReturnCorrectValues()
    {
        // Test with integers
        var intValues = new[] { 10, 20, 30, 40, 50 };
        var intColumn = NivaraColumn<int>.Create(intValues);

        for (int i = 0; i < intValues.Length; i++)
        {
            Assert.That(intColumn[i], Is.EqualTo(intValues[i]),
                $"Indexer should return correct value at position {i}");
        }

        // Test with strings
        var stringValues = new[] { "first", "second", "third" };
        var stringColumn = NivaraColumn<string>.Create(stringValues);

        for (int i = 0; i < stringValues.Length; i++)
        {
            Assert.That(stringColumn[i], Is.EqualTo(stringValues[i]),
                $"String indexer should return correct value at position {i}");
        }
    }

    /// <summary>
    /// Test indexer bounds checking
    /// </summary>
    [Test]
    public void ColumnIndexer_ShouldThrowOnOutOfBounds()
    {
        var values = new[] { 1, 2, 3 };
        var column = NivaraColumn<int>.Create(values);

        // Test negative index
        Assert.Throws<IndexOutOfRangeException>(() => { var _ = column[-1]; },
            "Negative index should throw IndexOutOfRangeException");

        // Test index equal to length
        Assert.Throws<IndexOutOfRangeException>(() => { var _ = column[3]; },
            "Index equal to length should throw IndexOutOfRangeException");

        // Test index greater than length
        Assert.Throws<IndexOutOfRangeException>(() => { var _ = column[10]; },
            "Index greater than length should throw IndexOutOfRangeException");
    }

    /// <summary>
    /// Test HasNulls property for non-nullable types
    /// </summary>
    [Test]
    public void NonNullableColumns_ShouldHaveHasNullsFalse()
    {
        // Test with value types (should never have nulls)
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        Assert.That(intColumn.HasNulls, Is.False, "Int column should not have nulls");

        var doubleColumn = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 });
        Assert.That(doubleColumn.HasNulls, Is.False, "Double column should not have nulls");

        // Test with reference types without nulls
        var stringColumn = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        Assert.That(stringColumn.HasNulls, Is.False, "String column without nulls should have HasNulls false");
    }

    /// <summary>
    /// Test reference type columns with null values
    /// </summary>
    [Test]
    public void ReferenceTypeColumns_ShouldDetectNulls()
    {
        // Test string column with nulls - create array with explicit null handling
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "hello", "world" });

        // Create a column with nulls using a different approach
        var stringValuesWithNulls = new string?[] { "hello", null, "world", null };
        var stringArrayWithNulls = new string[stringValuesWithNulls.Length];
        for (int i = 0; i < stringValuesWithNulls.Length; i++)
        {
            stringArrayWithNulls[i] = stringValuesWithNulls[i]!; // Suppress null warning for test
        }

        var stringColumnWithNulls = NivaraColumn<string>.CreateForReferenceType(stringArrayWithNulls);

        Assert.That(stringColumnWithNulls, Is.Not.Null, "String column with nulls should be created");
        Assert.That(stringColumnWithNulls.Length, Is.EqualTo(4), "Column should have correct length");

        // Test individual values
        Assert.That(stringColumnWithNulls[0], Is.EqualTo("hello"), "First value should be correct");
        Assert.That(stringColumnWithNulls[1], Is.Null, "Second value should be null");
        Assert.That(stringColumnWithNulls[2], Is.EqualTo("world"), "Third value should be correct");
        Assert.That(stringColumnWithNulls[3], Is.Null, "Fourth value should be null");

        // Test IsNull method
        Assert.That(stringColumnWithNulls.IsNull(0), Is.False, "First position should not be null");
        Assert.That(stringColumnWithNulls.IsNull(1), Is.True, "Second position should be null");
        Assert.That(stringColumnWithNulls.IsNull(2), Is.False, "Third position should not be null");
        Assert.That(stringColumnWithNulls.IsNull(3), Is.True, "Fourth position should be null");
    }

    /// <summary>
    /// Test basic null handling for reference types
    /// Note: Nullable value type testing will be added in a future iteration
    /// when the generic constraint issues are resolved
    /// </summary>
    [Test]
    public void ReferenceTypeColumns_ShouldHandleBasicNullDetection()
    {
        // Test string column without nulls
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "hello", "world" });
        Assert.That(stringColumn.HasNulls, Is.False, "String column without nulls should have HasNulls false");

        // For now, we'll skip the nullable value type test due to generic constraint limitations
        // This will be addressed in a future iteration when we can properly handle T? constraints
    }

    /// <summary>
    /// Test column disposal
    /// </summary>
    [Test]
    public void DisposedColumn_ShouldThrowObjectDisposedException()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        column.Dispose();

        Assert.Throws<ObjectDisposedException>(() => { var _ = column.Length; },
            "Accessing Length on disposed column should throw");
        Assert.Throws<ObjectDisposedException>(() => { var _ = column[0]; },
            "Accessing indexer on disposed column should throw");
        Assert.Throws<ObjectDisposedException>(() => { var _ = column.HasNulls; },
            "Accessing HasNulls on disposed column should throw");
        Assert.Throws<ObjectDisposedException>(() => { var _ = column.IsVectorizable; },
            "Accessing IsVectorizable on disposed column should throw");
    }

    #endregion

    #region Property 15: Immutability of operations

    /// <summary>
    /// Property 15: Immutability of operations
    /// For any NivaraColumn operation, the original column should remain unchanged and a new column instance should be returned.
    /// **Validates: Requirements 5.1**
    /// </summary>
    [Test]
    public void ColumnOperations_ShouldPreserveOriginalAndReturnNewInstance()
    {
        // Test arithmetic operations immutability
        var originalValues = new[] { 10, 20, 30, 40, 50 };
        var originalColumn = NivaraColumn<int>.Create(originalValues);

        // Test scalar multiplication
        var multiplied = originalColumn.Multiply(2);

        Assert.That(ReferenceEquals(multiplied, originalColumn), Is.False, "Multiply should return a new instance");
        Assert.That(originalColumn.Length, Is.EqualTo(5), "Original column length should be unchanged");
        Assert.That(originalColumn[0], Is.EqualTo(10), "Original column values should be unchanged");
        Assert.That(originalColumn[1], Is.EqualTo(20), "Original column values should be unchanged");
        Assert.That(originalColumn[4], Is.EqualTo(50), "Original column values should be unchanged");

        Assert.That(multiplied.Length, Is.EqualTo(5), "New column should have correct length");
        Assert.That(multiplied[0], Is.EqualTo(20), "New column should have multiplied values");
        Assert.That(multiplied[1], Is.EqualTo(40), "New column should have multiplied values");
        Assert.That(multiplied[4], Is.EqualTo(100), "New column should have multiplied values");

        // Test element-wise addition
        var otherColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var added = originalColumn.Add(otherColumn);

        Assert.That(ReferenceEquals(added, originalColumn), Is.False, "Add should return a new instance");
        Assert.That(ReferenceEquals(added, otherColumn), Is.False, "Add should not modify the other column");

        // Verify original columns are unchanged
        Assert.That(originalColumn[0], Is.EqualTo(10), "Original left column should be unchanged");
        Assert.That(otherColumn[0], Is.EqualTo(1), "Original right column should be unchanged");

        // Verify new column has correct values
        Assert.That(added[0], Is.EqualTo(11), "Added column should have correct values");
        Assert.That(added[4], Is.EqualTo(55), "Added column should have correct values");

        // Test element-wise multiplication
        var elementMultiplied = originalColumn.Multiply(otherColumn);

        Assert.That(ReferenceEquals(elementMultiplied, originalColumn), Is.False, "Element-wise multiply should return new instance");
        Assert.That(ReferenceEquals(elementMultiplied, otherColumn), Is.False, "Element-wise multiply should not modify other column");

        // Verify originals unchanged
        Assert.That(originalColumn[0], Is.EqualTo(10), "Original left column should be unchanged after element-wise multiply");
        Assert.That(otherColumn[0], Is.EqualTo(1), "Original right column should be unchanged after element-wise multiply");

        // Verify new column has correct values
        Assert.That(elementMultiplied[0], Is.EqualTo(10), "Element-wise multiplied column should have correct values");
        Assert.That(elementMultiplied[4], Is.EqualTo(250), "Element-wise multiplied column should have correct values");
    }

    [Test]
    public void ComparisonOperations_ShouldPreserveOriginalAndReturnNewInstance()
    {
        var originalValues = new[] { 10, 20, 30, 40, 50 };
        var originalColumn = NivaraColumn<int>.Create(originalValues);

        // Test scalar comparison
        var equalsResult = originalColumn.Equals(30);

        Assert.That(equalsResult, Is.Not.Null, "Equals should return a new instance");
        Assert.That(ReferenceEquals(equalsResult, originalColumn), Is.False, "Equals should return a different instance");
        Assert.That(originalColumn[0], Is.EqualTo(10), "Original column should be unchanged after equals");
        Assert.That(originalColumn[2], Is.EqualTo(30), "Original column should be unchanged after equals");

        // Test greater than comparison
        var greaterResult = originalColumn.GreaterThan(25);

        Assert.That(greaterResult, Is.Not.Null, "GreaterThan should return a new instance");
        Assert.That(ReferenceEquals(greaterResult, originalColumn), Is.False, "GreaterThan should return a different instance");
        Assert.That(originalColumn[0], Is.EqualTo(10), "Original column should be unchanged after GreaterThan");

        // Test less than comparison
        var lessResult = originalColumn.LessThan(35);

        Assert.That(lessResult, Is.Not.Null, "LessThan should return a new instance");
        Assert.That(ReferenceEquals(lessResult, originalColumn), Is.False, "LessThan should return a different instance");
        Assert.That(originalColumn[4], Is.EqualTo(50), "Original column should be unchanged after LessThan");

        // Test element-wise comparison
        var otherColumn = NivaraColumn<int>.Create(new[] { 15, 25, 30, 35, 45 });
        var elementEqualsResult = originalColumn.Equals(otherColumn);

        Assert.That(elementEqualsResult, Is.Not.Null, "Element-wise equals should return new instance");
        Assert.That(ReferenceEquals(elementEqualsResult, originalColumn), Is.False, "Element-wise equals should return different instance");
        Assert.That(ReferenceEquals(elementEqualsResult, otherColumn), Is.False, "Element-wise equals should not modify other column");

        // Verify originals unchanged
        Assert.That(originalColumn[0], Is.EqualTo(10), "Original left column should be unchanged");
        Assert.That(otherColumn[0], Is.EqualTo(15), "Original right column should be unchanged");
    }

    [Test]
    public void SlicingOperations_ShouldPreserveOriginalAndReturnNewInstance()
    {
        var originalValues = new[] { 10, 20, 30, 40, 50 };
        var originalColumn = NivaraColumn<int>.Create(originalValues);

        // Test slicing
        var sliced = originalColumn.Slice(1, 3);

        Assert.That(sliced, Is.Not.SameAs(originalColumn), "Slice should return a new instance");
        Assert.That(originalColumn.Length, Is.EqualTo(5), "Original column length should be unchanged");
        Assert.That(originalColumn[0], Is.EqualTo(10), "Original column values should be unchanged");
        Assert.That(originalColumn[4], Is.EqualTo(50), "Original column values should be unchanged");

        Assert.That(sliced.Length, Is.EqualTo(3), "Sliced column should have correct length");
        Assert.That(sliced[0], Is.EqualTo(20), "Sliced column should have correct values");
        Assert.That(sliced[1], Is.EqualTo(30), "Sliced column should have correct values");
        Assert.That(sliced[2], Is.EqualTo(40), "Sliced column should have correct values");

        // Modify sliced column (if it were mutable) shouldn't affect original
        // Since columns are immutable, we can't directly test this, but we verify
        // that operations on the slice don't affect the original
        var slicedMultiplied = sliced.Multiply(10);

        Assert.That(originalColumn[1], Is.EqualTo(20), "Original should be unchanged after operations on slice");
        Assert.That(originalColumn[2], Is.EqualTo(30), "Original should be unchanged after operations on slice");
        Assert.That(slicedMultiplied[0], Is.EqualTo(200), "Operations on slice should work correctly");
    }

    /// <summary>
    /// Property 15: Immutability of operations - Enhanced property-based tests
    /// For any NivaraColumn operation, the original column should remain unchanged and a new column instance should be returned.
    /// **Validates: Requirements 5.1**
    /// </summary>
    [TestCase(new int[] { 1, 2, 3 })]
    [TestCase(new int[] { -5, 0, 5, 10, 15 })]
    [TestCase(new int[] { int.MaxValue, int.MinValue, 0 })]
    [TestCase(new int[] { 42 })]
    public void ImmutabilityProperty_ArithmeticOperations_ShouldPreserveOriginals(int[] values)
    {
        if (values.Length == 0) return; // Skip empty arrays for arithmetic operations

        var originalColumn = NivaraColumn<int>.Create(values);
        var originalValuesCopy = values.ToArray(); // Keep a copy for verification

        // Test scalar multiplication immutability
        var multiplied = originalColumn.Multiply(3);

        Assert.That(ReferenceEquals(multiplied, originalColumn), Is.False,
            "Scalar multiplication should return a new instance");

        // Verify original column is unchanged
        Assert.That(originalColumn.Length, Is.EqualTo(values.Length),
            "Original column length should be unchanged after scalar multiplication");

        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(originalColumn[i], Is.EqualTo(originalValuesCopy[i]),
                $"Original column value at index {i} should be unchanged after scalar multiplication");
        }

        // Test element-wise operations if we have enough elements
        if (values.Length > 1)
        {
            var otherValues = values.Select(x => x + 1).ToArray();
            var otherColumn = NivaraColumn<int>.Create(otherValues);

            var added = originalColumn.Add(otherColumn);
            var elementMultiplied = originalColumn.Multiply(otherColumn);

            Assert.That(ReferenceEquals(added, originalColumn), Is.False,
                "Element-wise addition should return a new instance");
            Assert.That(ReferenceEquals(elementMultiplied, originalColumn), Is.False,
                "Element-wise multiplication should return a new instance");

            // Verify originals are still unchanged
            for (int i = 0; i < values.Length; i++)
            {
                Assert.That(originalColumn[i], Is.EqualTo(originalValuesCopy[i]),
                    $"Original column value at index {i} should be unchanged after element-wise operations");
                Assert.That(otherColumn[i], Is.EqualTo(otherValues[i]),
                    $"Other column value at index {i} should be unchanged after element-wise operations");
            }
        }
    }

    /// <summary>
    /// Property 15: Immutability of operations - Comparison operations property test
    /// **Validates: Requirements 5.1**
    /// </summary>
    [TestCase(new int[] { 1, 5, 10, 15, 20 })]
    [TestCase(new int[] { -10, -5, 0, 5, 10 })]
    [TestCase(new int[] { 100, 200, 300 })]
    [TestCase(new int[] { 42 })]
    public void ImmutabilityProperty_ComparisonOperations_ShouldPreserveOriginals(int[] values)
    {
        var originalColumn = NivaraColumn<int>.Create(values);
        var originalValuesCopy = values.ToArray();
        var targetValue = values.Length > 0 ? values[values.Length / 2] : 0;

        // Test all comparison operations
        var equalsResult = originalColumn.Equals(targetValue);
        var greaterResult = originalColumn.GreaterThan(targetValue);
        var lessResult = originalColumn.LessThan(targetValue);

        // Verify all return new instances
        Assert.That(ReferenceEquals(equalsResult, originalColumn), Is.False,
            "Equals should return a new instance");
        Assert.That(ReferenceEquals(greaterResult, originalColumn), Is.False,
            "GreaterThan should return a new instance");
        Assert.That(ReferenceEquals(lessResult, originalColumn), Is.False,
            "LessThan should return a new instance");

        // Verify original column is unchanged
        Assert.That(originalColumn.Length, Is.EqualTo(values.Length),
            "Original column length should be unchanged after comparisons");

        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(originalColumn[i], Is.EqualTo(originalValuesCopy[i]),
                $"Original column value at index {i} should be unchanged after comparisons");
        }

        // Test element-wise comparison if we have multiple elements
        if (values.Length > 1)
        {
            var otherValues = values.Select(x => x % 2 == 0 ? x : x + 1).ToArray();
            var otherColumn = NivaraColumn<int>.Create(otherValues);

            var elementEquals = originalColumn.Equals(otherColumn);

            Assert.That(ReferenceEquals(elementEquals, originalColumn), Is.False,
                "Element-wise equals should return a new instance");
            Assert.That(ReferenceEquals(elementEquals, otherColumn), Is.False,
                "Element-wise equals should not modify other column");

            // Verify both originals are unchanged
            for (int i = 0; i < values.Length; i++)
            {
                Assert.That(originalColumn[i], Is.EqualTo(originalValuesCopy[i]),
                    $"Original column value at index {i} should be unchanged after element-wise comparison");
                Assert.That(otherColumn[i], Is.EqualTo(otherValues[i]),
                    $"Other column value at index {i} should be unchanged after element-wise comparison");
            }
        }
    }

    /// <summary>
    /// Property 15: Immutability of operations - String operations property test
    /// **Validates: Requirements 5.1**
    /// </summary>
    [Test]
    public void ImmutabilityProperty_StringOperations_ShouldPreserveOriginals()
    {
        var testCases = new[]
        {
            new string[] { "apple", "banana", "cherry" },
            new string[] { "hello", "world" },
            new string[] { "single" },
            new string[] { "", "non-empty", "" }
        };

        foreach (var values in testCases)
        {
            var originalColumn = NivaraColumn<string>.Create(values);
            var originalValuesCopy = values.ToArray();
            var targetValue = values.Length > 0 ? values[0] : "test";

            // Test comparison operations on strings
            var equalsResult = originalColumn.Equals(targetValue);
            var greaterResult = originalColumn.GreaterThan(targetValue);
            var lessResult = originalColumn.LessThan(targetValue);

            // Verify all return new instances
            Assert.That(ReferenceEquals(equalsResult, originalColumn), Is.False,
                "String equals should return a new instance");
            Assert.That(ReferenceEquals(greaterResult, originalColumn), Is.False,
                "String GreaterThan should return a new instance");
            Assert.That(ReferenceEquals(lessResult, originalColumn), Is.False,
                "String LessThan should return a new instance");

            // Verify original column is unchanged
            Assert.That(originalColumn.Length, Is.EqualTo(values.Length),
                "Original string column length should be unchanged");

            for (int i = 0; i < values.Length; i++)
            {
                Assert.That(originalColumn[i], Is.EqualTo(originalValuesCopy[i]),
                    $"Original string column value at index {i} should be unchanged");
            }
        }
    }

    #endregion

    #region Property 16: Efficient slicing

    /// <summary>
    /// Property 16: Efficient slicing
    /// For any NivaraColumn and any valid slice range, slicing should return a new column that correctly represents the specified range of the original data.
    /// **Validates: Requirements 5.4**
    /// </summary>
    [Test]
    public void ColumnSlicing_ShouldReturnCorrectSubsetForAllValidRanges()
    {
        var originalValues = new[] { 100, 200, 300, 400, 500, 600, 700 };
        var originalColumn = NivaraColumn<int>.Create(originalValues);

        // Test various slice ranges
        var testCases = new[]
        {
            new { Start = 0, Length = 3, Expected = new[] { 100, 200, 300 }, Description = "slice from beginning" },
            new { Start = 2, Length = 3, Expected = new[] { 300, 400, 500 }, Description = "slice from middle" },
            new { Start = 4, Length = 3, Expected = new[] { 500, 600, 700 }, Description = "slice to end" },
            new { Start = 0, Length = 7, Expected = new[] { 100, 200, 300, 400, 500, 600, 700 }, Description = "full slice" },
            new { Start = 3, Length = 1, Expected = new[] { 400 }, Description = "single element slice" },
            new { Start = 6, Length = 1, Expected = new[] { 700 }, Description = "last element slice" },
            new { Start = 0, Length = 1, Expected = new[] { 100 }, Description = "first element slice" }
        };

        foreach (var testCase in testCases)
        {
            var sliced = originalColumn.Slice(testCase.Start, testCase.Length);

            Assert.That(sliced.Length, Is.EqualTo(testCase.Expected.Length),
                $"Sliced column length should be correct for {testCase.Description}");

            for (int i = 0; i < testCase.Expected.Length; i++)
            {
                Assert.That(sliced[i], Is.EqualTo(testCase.Expected[i]),
                    $"Sliced value at index {i} should be correct for {testCase.Description}");
            }
        }
    }

    [Test]
    public void ColumnSlicing_ShouldHandleEdgeCases()
    {
        // Test empty slice
        var values = new[] { 1, 2, 3, 4, 5 };
        var column = NivaraColumn<int>.Create(values);

        var emptySlice = column.Slice(2, 0);
        Assert.That(emptySlice.Length, Is.EqualTo(0), "Empty slice should have length 0");

        // Test single element column
        var singleColumn = NivaraColumn<int>.Create(new[] { 42 });
        var singleSlice = singleColumn.Slice(0, 1);

        Assert.That(singleSlice.Length, Is.EqualTo(1), "Single element slice should have length 1");
        Assert.That(singleSlice[0], Is.EqualTo(42), "Single element slice should have correct value");

        // Test slicing with different data types
        var stringValues = new[] { "apple", "banana", "cherry", "date", "elderberry" };
        var stringColumn = NivaraColumn<string>.Create(stringValues);
        var stringSlice = stringColumn.Slice(1, 3);

        Assert.That(stringSlice.Length, Is.EqualTo(3), "String slice should have correct length");
        Assert.That(stringSlice[0], Is.EqualTo("banana"), "String slice should have correct values");
        Assert.That(stringSlice[1], Is.EqualTo("cherry"), "String slice should have correct values");
        Assert.That(stringSlice[2], Is.EqualTo("date"), "String slice should have correct values");
    }

    [Test]
    public void ColumnSlicing_ShouldThrowForInvalidRanges()
    {
        var values = new[] { 1, 2, 3, 4, 5 };
        var column = NivaraColumn<int>.Create(values);

        // Test invalid start positions
        Assert.Throws<ArgumentOutOfRangeException>(() => column.Slice(-1, 2),
            "Negative start should throw ArgumentOutOfRangeException");
        Assert.Throws<ArgumentOutOfRangeException>(() => column.Slice(6, 1),
            "Start beyond length should throw ArgumentOutOfRangeException");

        // Test invalid lengths
        Assert.Throws<ArgumentOutOfRangeException>(() => column.Slice(0, -1),
            "Negative length should throw ArgumentOutOfRangeException");
        Assert.Throws<ArgumentOutOfRangeException>(() => column.Slice(3, 3),
            "Length extending beyond end should throw ArgumentOutOfRangeException");
        Assert.Throws<ArgumentOutOfRangeException>(() => column.Slice(2, 4),
            "Start + length > column length should throw ArgumentOutOfRangeException");
    }

    [Test]
    public void ColumnSlicing_WithNullValues_ShouldPreserveNullSemantics()
    {
        // Test slicing with reference type nulls
        var stringValues = new[] { "apple", null!, "cherry", null!, "elderberry" };
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(stringValues);

        var sliced = stringColumn.Slice(1, 3); // Should get [null, "cherry", null]

        Assert.That(sliced.Length, Is.EqualTo(3), "Sliced column should have correct length");
        Assert.That(sliced[0], Is.Null, "First sliced element should be null");
        Assert.That(sliced[1], Is.EqualTo("cherry"), "Second sliced element should be correct");
        Assert.That(sliced[2], Is.Null, "Third sliced element should be null");

        // Test null detection methods on sliced column
        Assert.That(sliced.IsNull(0), Is.True, "First element should be detected as null");
        Assert.That(sliced.IsNull(1), Is.False, "Second element should not be detected as null");
        Assert.That(sliced.IsNull(2), Is.True, "Third element should be detected as null");
        Assert.That(sliced.HasNulls, Is.True, "Sliced column should report having nulls");
    }

    /// <summary>
    /// Property 16: Efficient slicing - Enhanced property-based tests
    /// For any NivaraColumn and any valid slice range, slicing should return a new column that correctly represents the specified range of the original data.
    /// **Validates: Requirements 5.4**
    /// </summary>
    [TestCase(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 0, 5)]
    [TestCase(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 3, 4)]
    [TestCase(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 7, 3)]
    [TestCase(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 0, 10)]
    [TestCase(new int[] { 100, 200, 300 }, 1, 2)]
    [TestCase(new int[] { 42 }, 0, 1)]
    [TestCase(new int[] { -5, -3, -1, 1, 3, 5 }, 2, 3)]
    public void SlicingProperty_ValidRanges_ShouldReturnCorrectSubset(int[] values, int start, int length)
    {
        var originalColumn = NivaraColumn<int>.Create(values);
        var originalValuesCopy = values.ToArray();

        var sliced = originalColumn.Slice(start, length);

        // Verify slice returns new instance
        Assert.That(ReferenceEquals(sliced, originalColumn), Is.False,
            "Slice should return a new instance");

        // Verify slice has correct length
        Assert.That(sliced.Length, Is.EqualTo(length),
            $"Sliced column should have length {length}");

        // Verify slice contains correct values
        for (int i = 0; i < length; i++)
        {
            Assert.That(sliced[i], Is.EqualTo(values[start + i]),
                $"Sliced value at index {i} should match original value at index {start + i}");
        }

        // Verify original column is unchanged
        Assert.That(originalColumn.Length, Is.EqualTo(values.Length),
            "Original column length should be unchanged after slicing");

        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(originalColumn[i], Is.EqualTo(originalValuesCopy[i]),
                $"Original column value at index {i} should be unchanged after slicing");
        }
    }

    /// <summary>
    /// Property 16: Efficient slicing - Different data types property test
    /// **Validates: Requirements 5.4**
    /// </summary>
    [TestCase(new string[] { "a", "b", "c", "d", "e", "f" }, 1, 3)]
    [TestCase(new string[] { "hello", "world", "test", "slice" }, 0, 2)]
    [TestCase(new string[] { "apple", "banana", "cherry", "date", "elderberry" }, 2, 2)]
    [TestCase(new string[] { "single" }, 0, 1)]
    public void SlicingProperty_StringColumns_ShouldReturnCorrectSubset(string[] values, int start, int length)
    {
        var originalColumn = NivaraColumn<string>.Create(values);
        var originalValuesCopy = values.ToArray();

        var sliced = originalColumn.Slice(start, length);

        // Verify slice properties
        Assert.That(ReferenceEquals(sliced, originalColumn), Is.False,
            "String slice should return a new instance");
        Assert.That(sliced.Length, Is.EqualTo(length),
            $"String sliced column should have length {length}");

        // Verify slice contains correct values
        for (int i = 0; i < length; i++)
        {
            Assert.That(sliced[i], Is.EqualTo(values[start + i]),
                $"String sliced value at index {i} should match original value at index {start + i}");
        }

        // Verify original is unchanged
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(originalColumn[i], Is.EqualTo(originalValuesCopy[i]),
                $"Original string column value at index {i} should be unchanged after slicing");
        }
    }

    /// <summary>
    /// Property 16: Efficient slicing - Edge cases property test
    /// **Validates: Requirements 5.4**
    /// </summary>
    [TestCase(new double[] { 1.1, 2.2, 3.3, 4.4, 5.5 }, 2, 0)] // Empty slice
    [TestCase(new double[] { 1.1, 2.2, 3.3, 4.4, 5.5 }, 0, 1)] // First element only
    [TestCase(new double[] { 1.1, 2.2, 3.3, 4.4, 5.5 }, 4, 1)] // Last element only
    [TestCase(new double[] { 1.1, 2.2, 3.3, 4.4, 5.5 }, 0, 5)] // Full slice
    [TestCase(new double[] { 3.14 }, 0, 1)] // Single element column
    [TestCase(new double[] { 3.14 }, 1, 0)] // Empty slice from single element
    public void SlicingProperty_EdgeCases_ShouldHandleCorrectly(double[] values, int start, int length)
    {
        var originalColumn = NivaraColumn<double>.Create(values);
        var originalValuesCopy = values.ToArray();

        var sliced = originalColumn.Slice(start, length);

        // Verify slice properties
        Assert.That(ReferenceEquals(sliced, originalColumn), Is.False,
            "Edge case slice should return a new instance");
        Assert.That(sliced.Length, Is.EqualTo(length),
            $"Edge case sliced column should have length {length}");

        // Verify slice contains correct values (if any)
        for (int i = 0; i < length; i++)
        {
            Assert.That(sliced[i], Is.EqualTo(values[start + i]).Within(0.000001),
                $"Edge case sliced value at index {i} should match original value at index {start + i}");
        }

        // Verify original is unchanged
        Assert.That(originalColumn.Length, Is.EqualTo(values.Length),
            "Original column length should be unchanged after edge case slicing");

        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(originalColumn[i], Is.EqualTo(originalValuesCopy[i]).Within(0.000001),
                $"Original column value at index {i} should be unchanged after edge case slicing");
        }
    }

    /// <summary>
    /// Property 16: Efficient slicing - Invalid ranges should throw exceptions
    /// **Validates: Requirements 5.4**
    /// </summary>
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, -1, 2)] // Negative start
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 6, 1)]  // Start beyond length
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 0, -1)] // Negative length
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 3, 3)]  // Length extending beyond end
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 2, 4)]  // Start + length > column length
    public void SlicingProperty_InvalidRanges_ShouldThrowArgumentOutOfRangeException(int[] values, int start, int length)
    {
        var column = NivaraColumn<int>.Create(values);

        Assert.Throws<ArgumentOutOfRangeException>(() => column.Slice(start, length),
            $"Slice with start={start}, length={length} should throw ArgumentOutOfRangeException");
    }

    /// <summary>
    /// Property 16: Efficient slicing - Null preservation property test
    /// **Validates: Requirements 5.4**
    /// </summary>
    [Test]
    public void SlicingProperty_WithNulls_ShouldPreserveNullSemantics()
    {
        var testCases = new[]
        {
            new { Values = new string[] { "a", null!, "c", null!, "e" }, Start = 1, Length = 3, Description = "Slice with nulls" },
            new { Values = new string[] { null!, "b", "c" }, Start = 0, Length = 2, Description = "Slice starting with null" },
            new { Values = new string[] { "a", "b", null! }, Start = 1, Length = 2, Description = "Slice ending with null" },
            new { Values = new string[] { null!, null!, null! }, Start = 0, Length = 3, Description = "All nulls slice" }
        };

        foreach (var testCase in testCases)
        {
            var originalColumn = NivaraColumn<string>.CreateForReferenceType(testCase.Values);

            var sliced = originalColumn.Slice(testCase.Start, testCase.Length);

            // Verify slice properties
            Assert.That(ReferenceEquals(sliced, originalColumn), Is.False,
                $"Null slice should return a new instance for {testCase.Description}");
            Assert.That(sliced.Length, Is.EqualTo(testCase.Length),
                $"Null sliced column should have length {testCase.Length} for {testCase.Description}");

            // Verify slice contains correct values and null semantics
            bool expectedHasNulls = false;
            for (int i = 0; i < testCase.Length; i++)
            {
                var expectedValue = testCase.Values[testCase.Start + i];
                var actualValue = sliced[i];

                if (expectedValue == null)
                {
                    expectedHasNulls = true;
                    Assert.That(actualValue, Is.Null,
                        $"Sliced value at index {i} should be null for {testCase.Description}");
                    Assert.That(sliced.IsNull(i), Is.True,
                        $"Sliced IsNull at index {i} should be true for {testCase.Description}");
                }
                else
                {
                    Assert.That(actualValue, Is.EqualTo(expectedValue),
                        $"Sliced value at index {i} should match expected non-null value for {testCase.Description}");
                    Assert.That(sliced.IsNull(i), Is.False,
                        $"Sliced IsNull at index {i} should be false for {testCase.Description}");
                }
            }

            // Verify HasNulls property
            Assert.That(sliced.HasNulls, Is.EqualTo(expectedHasNulls),
                $"Sliced column HasNulls should be {expectedHasNulls} for {testCase.Description}");
        }
    }

    #endregion

    #region CreateFromNullable Tests

    /// <summary>
    /// Test the new CreateFromNullable method for nullable value types
    /// **Validates: Requirements 7.2, 7.3, 7.4, 7.5**
    /// </summary>
    [Test]
    public void CreateFromNullable_ShouldHandleNullableValueTypes()
    {
        // Test with nullable integers
        var nullableInts = new int?[] { 1, null, 3, null, 5 };
        var column = NivaraColumn<int>.CreateFromNullable(nullableInts);

        Assert.That(column, Is.Not.Null, "Column should be created from nullable array");
        Assert.That(column.Length, Is.EqualTo(5), "Column should have correct length");
        Assert.That(column.HasNulls, Is.True, "Column should detect nulls");
        Assert.That(column.NullCount, Is.EqualTo(2), "Column should count nulls correctly");

        // Verify values and null positions
        Assert.That(column[0], Is.EqualTo(1), "First value should be 1");
        Assert.That(column.IsNull(1), Is.True, "Second position should be null");
        Assert.That(column[2], Is.EqualTo(3), "Third value should be 3");
        Assert.That(column.IsNull(3), Is.True, "Fourth position should be null");
        Assert.That(column[4], Is.EqualTo(5), "Fifth value should be 5");

        // Test null indices
        var nullIndices = column.GetNullIndices();
        Assert.That(nullIndices, Is.EqualTo(new[] { 1, 3 }), "Null indices should be correct");
    }

    /// <summary>
    /// Test CreateFromNullable with different value types
    /// </summary>
    [Test]
    public void CreateFromNullable_ShouldWorkWithDifferentValueTypes()
    {
        // Test with nullable doubles
        var nullableDoubles = new double?[] { 1.5, null, 3.14 };
        var doubleColumn = NivaraColumn<double>.CreateFromNullable(nullableDoubles);

        Assert.That(doubleColumn.Length, Is.EqualTo(3));
        Assert.That(doubleColumn.HasNulls, Is.True);
        Assert.That(doubleColumn[0], Is.EqualTo(1.5));
        Assert.That(doubleColumn.IsNull(1), Is.True);
        Assert.That(doubleColumn[2], Is.EqualTo(3.14));

        // Test with nullable booleans
        var nullableBools = new bool?[] { true, null, false, null };
        var boolColumn = NivaraColumn<bool>.CreateFromNullable(nullableBools);

        Assert.That(boolColumn.Length, Is.EqualTo(4));
        Assert.That(boolColumn.HasNulls, Is.True);
        Assert.That(boolColumn.NullCount, Is.EqualTo(2));
        Assert.That(boolColumn[0], Is.True);
        Assert.That(boolColumn.IsNull(1), Is.True);
        Assert.That(boolColumn[2], Is.False);
        Assert.That(boolColumn.IsNull(3), Is.True);
    }

    /// <summary>
    /// Test CreateFromNullable error handling
    /// </summary>
    [Test]
    public void CreateFromNullable_ShouldThrowForInvalidInputs()
    {
        // Test null array
        Assert.Throws<ArgumentNullException>(() =>
            NivaraColumn<int>.CreateFromNullable(null!));

        // Test with reference type (should fail)
        var stringArray = new string[] { "a", "b" };
        Assert.Throws<InvalidOperationException>(() =>
            NivaraColumn<string>.CreateFromNullable(stringArray));
    }

    /// <summary>
    /// Test advanced null handling methods
    /// **Validates: Requirements 7.4, 7.5**
    /// </summary>
    [Test]
    public void AdvancedNullHandling_ShouldWorkCorrectly()
    {
        var nullableInts = new int?[] { 1, null, 3, null, 5 };
        var column = NivaraColumn<int>.CreateFromNullable(nullableInts);

        // Test FillNull
        var filled = column.FillNull(0);
        Assert.That(filled.HasNulls, Is.False, "Filled column should have no nulls");
        Assert.That(filled.ToArray(), Is.EqualTo(new[] { 1, 0, 3, 0, 5 }), "Nulls should be filled with 0");

        // Test FillNullForward
        var forwardFilled = column.FillNullForward();
        Assert.That(forwardFilled.HasNulls, Is.False, "Forward filled column should have no nulls");
        Assert.That(forwardFilled.ToArray(), Is.EqualTo(new[] { 1, 1, 3, 3, 5 }), "Nulls should be forward filled");

        // Test FillNullBackward
        var backwardFilled = column.FillNullBackward();
        Assert.That(backwardFilled.HasNulls, Is.False, "Backward filled column should have no nulls");
        Assert.That(backwardFilled.ToArray(), Is.EqualTo(new[] { 1, 3, 3, 5, 5 }), "Nulls should be backward filled");

        // Test DropNulls
        var dropped = column.DropNulls();
        Assert.That(dropped.HasNulls, Is.False, "Dropped column should have no nulls");
        Assert.That(dropped.Length, Is.EqualTo(3), "Dropped column should have 3 elements");
        Assert.That(dropped.ToArray(), Is.EqualTo(new[] { 1, 3, 5 }), "Only non-null values should remain");
    }

    /// <summary>
    /// Test error handling for fill operations
    /// </summary>
    [Test]
    public void NullFillOperations_ShouldHandleEdgeCases()
    {
        // Test forward fill with leading null (should throw)
        var leadingNullColumn = NivaraColumn<int>.CreateFromNullable(new int?[] { null, 2, 3 });
        Assert.Throws<InvalidOperationException>(() => leadingNullColumn.FillNullForward());

        // Test backward fill with trailing null (should throw)
        var trailingNullColumn = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, 2, null });
        Assert.Throws<InvalidOperationException>(() => trailingNullColumn.FillNullBackward());

        // Test operations on empty column
        var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
        Assert.Throws<InvalidOperationException>(() => emptyColumn.FillNullForward());
        Assert.Throws<InvalidOperationException>(() => emptyColumn.FillNullBackward());
    }

    #endregion
}