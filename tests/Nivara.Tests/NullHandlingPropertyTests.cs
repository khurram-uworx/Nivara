using NUnit.Framework;
using Nivara;
using Nivara.Storage;

namespace Nivara.Tests;

/// <summary>
/// Property-based tests for null handling functionality in NivaraColumn.
/// Tests Properties 17, 18, and 19 from the core column types design.
/// </summary>
[TestFixture]
public class NullHandlingPropertyTests
{
    #region Property 17: Null mask maintenance

    /// <summary>
    /// Property 17: Null mask maintenance
    /// For any column operation involving nulls, the null mask should be correctly updated 
    /// to reflect the null status of all elements in the result.
    /// **Validates: Requirements 7.2, 7.3**
    /// </summary>
    [Test]
    public void NullMaskMaintenance_ArithmeticOperations_PreservesNullPositions()
    {
        // Feature: core-column-types, Property 17: Null mask maintenance
        var testCases = new[]
        {
            new int?[] { 1, null, 3, null, 5 },
            new int?[] { null, null, null },
            new int?[] { 1, 2, 3 },
            new int?[] { null }
        };
        
        foreach (var values in testCases)
        {
            if (values.Length == 0) continue; // Skip empty arrays
            
            var column = NivaraColumn<int>.CreateFromNullable(values);
            var scalar = 5;
            
            // Test scalar multiplication
            var multiplied = column.Multiply(scalar);
            
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] == null)
                {
                    Assert.That(multiplied.IsNull(i), Is.True, 
                        $"Position {i} should be null after scalar multiplication");
                }
                else
                {
                    Assert.That(multiplied.IsNull(i), Is.False, 
                        $"Position {i} should not be null after scalar multiplication");
                    Assert.That(multiplied[i], Is.EqualTo(values[i]!.Value * scalar),
                        $"Position {i} should have correct multiplied value");
                }
            }
        }
    }

    [Test]
    public void NullMaskMaintenance_ElementWiseOperations_PropagatesNulls()
    {
        // Feature: core-column-types, Property 17: Null mask maintenance
        var testCases = new[]
        {
            (new int?[] { 1, null, 3 }, new int?[] { 2, 4, null }),
            (new int?[] { null, null }, new int?[] { 1, 2 }),
            (new int?[] { 1, 2 }, new int?[] { null, null }),
            (new int?[] { null }, new int?[] { null })
        };
        
        foreach (var (leftValues, rightValues) in testCases)
        {
            var leftColumn = NivaraColumn<int>.CreateFromNullable(leftValues);
            var rightColumn = NivaraColumn<int>.CreateFromNullable(rightValues);
            
            // Test element-wise addition
            var result = leftColumn.Add(rightColumn);
            
            for (int i = 0; i < leftValues.Length; i++)
            {
                bool shouldBeNull = leftValues[i] == null || rightValues[i] == null;
                
                if (shouldBeNull)
                {
                    Assert.That(result.IsNull(i), Is.True, 
                        $"Position {i} should be null when either operand is null");
                }
                else
                {
                    Assert.That(result.IsNull(i), Is.False, 
                        $"Position {i} should not be null when both operands are non-null");
                    Assert.That(result[i], Is.EqualTo(leftValues[i]!.Value + rightValues[i]!.Value),
                        $"Position {i} should have correct sum value");
                }
            }
        }
    }

    [Test]
    public void NullMaskMaintenance_ComparisonOperations_PropagatesNulls()
    {
        // Feature: core-column-types, Property 17: Null mask maintenance
        var testCases = new[]
        {
            new int?[] { 1, null, 3, null, 5 },
            new int?[] { null, null, null },
            new int?[] { 1, 2, 3 }
        };
        
        foreach (var values in testCases)
        {
            if (values.Length == 0) continue; // Skip empty arrays
            
            var column = NivaraColumn<int>.CreateFromNullable(values);
            var compareValue = 2;
            
            // Test scalar comparison
            var comparison = column.GreaterThan(compareValue);
            
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] == null)
                {
                    Assert.That(comparison.IsNull(i), Is.True, 
                        $"Position {i} should be null in comparison result when input is null");
                }
                else
                {
                    Assert.That(comparison.IsNull(i), Is.False, 
                        $"Position {i} should not be null in comparison result when input is non-null");
                    Assert.That(comparison[i], Is.EqualTo(values[i]!.Value > compareValue),
                        $"Position {i} should have correct comparison result");
                }
            }
        }
    }

    #endregion

    #region Property 18: Null checking methods

    /// <summary>
    /// Property 18: Null checking methods
    /// For any NivaraColumn, null checking methods should correctly identify null positions 
    /// and provide accurate null counts.
    /// **Validates: Requirements 7.4**
    /// </summary>
    [Test]
    public void NullCheckingMethods_CorrectlyIdentifyNulls()
    {
        // Feature: core-column-types, Property 18: Null checking methods
        var testCases = new[]
        {
            (new int?[] { 1, null, 3, null, 5 }, 2),
            (new int?[] { null, null, null }, 3),
            (new int?[] { 1, 2, 3 }, 0),
            (new int?[] { null }, 1),
            (new int?[] { }, 0)
        };
        
        foreach (var (values, expectedNullCount) in testCases)
        {
            var column = NivaraColumn<int>.CreateFromNullable(values);
            
            // Test HasNulls property
            bool expectedHasNulls = expectedNullCount > 0;
            Assert.That(column.HasNulls, Is.EqualTo(expectedHasNulls),
                $"HasNulls should be {expectedHasNulls} for column with {expectedNullCount} nulls");
            
            // Test individual null checks
            for (int i = 0; i < values.Length; i++)
            {
                bool expectedIsNull = values[i] == null;
                Assert.That(column.IsNull(i), Is.EqualTo(expectedIsNull),
                    $"IsNull({i}) should return {expectedIsNull}");
            }
        }
    }

    [Test]
    public void NullCheckingMethods_WorksWithReferenceTypes()
    {
        // Feature: core-column-types, Property 18: Null checking methods
        var testCases = new[]
        {
            new string?[] { "a", null, "c", null },
            new string?[] { null, null },
            new string?[] { "a", "b", "c" },
            new string?[] { null }
        };
        
        foreach (var values in testCases)
        {
            var column = NivaraColumn<string>.Create(values!); // Use null-forgiving operator
            
            int expectedNullCount = values.Count(v => v == null);
            bool expectedHasNulls = expectedNullCount > 0;
            
            Assert.That(column.HasNulls, Is.EqualTo(expectedHasNulls),
                $"HasNulls should be {expectedHasNulls} for string column");
            
            for (int i = 0; i < values.Length; i++)
            {
                bool expectedIsNull = values[i] == null;
                Assert.That(column.IsNull(i), Is.EqualTo(expectedIsNull),
                    $"IsNull({i}) should return {expectedIsNull} for string column");
            }
        }
    }

    #endregion

    #region Property 19: Null value filling

    /// <summary>
    /// Property 19: Null value filling
    /// For any NivaraColumn with nulls and any default value, filling nulls should replace 
    /// all null positions with the specified default while preserving non-null values.
    /// **Validates: Requirements 7.5**
    /// </summary>
    [Test]
    public void NullValueFilling_ReplacesNullsWithDefault()
    {
        // Feature: core-column-types, Property 19: Null value filling
        var testCases = new[]
        {
            (new int?[] { 1, null, 3, null, 5 }, 99),
            (new int?[] { null, null, null }, 42),
            (new int?[] { 1, 2, 3 }, 0),
            (new int?[] { null }, -1)
        };
        
        foreach (var (values, fillValue) in testCases)
        {
            var column = NivaraColumn<int>.CreateFromNullable(values);
            
            // Test FillNull method
            var filled = column.FillNull(fillValue);
            
            // Verify no nulls remain
            Assert.That(filled.HasNulls, Is.False, "Filled column should have no nulls");
            
            // Verify values are correct
            for (int i = 0; i < values.Length; i++)
            {
                int expectedValue = values[i] ?? fillValue;
                Assert.That(filled[i], Is.EqualTo(expectedValue),
                    $"Position {i} should have value {expectedValue} after filling nulls");
                Assert.That(filled.IsNull(i), Is.False,
                    $"Position {i} should not be null after filling");
            }
            
            // Verify original column is unchanged (immutability)
            for (int i = 0; i < values.Length; i++)
            {
                bool originalIsNull = values[i] == null;
                Assert.That(column.IsNull(i), Is.EqualTo(originalIsNull),
                    $"Original column position {i} should remain unchanged");
            }
        }
    }

    [Test]
    public void NullValueFilling_WorksWithReferenceTypes()
    {
        // Feature: core-column-types, Property 19: Null value filling
        var testCases = new[]
        {
            (new string?[] { "a", null, "c", null }, "DEFAULT"),
            (new string?[] { null, null }, "FILL"),
            (new string?[] { "x", "y", "z" }, "UNUSED")
        };
        
        foreach (var (values, fillValue) in testCases)
        {
            var column = NivaraColumn<string>.Create(values!); // Use null-forgiving operator
            
            var filled = column.FillNull(fillValue);
            
            // Verify no nulls remain
            Assert.That(filled.HasNulls, Is.False, "Filled string column should have no nulls");
            
            // Verify values are correct
            for (int i = 0; i < values.Length; i++)
            {
                string expectedValue = values[i] ?? fillValue;
                Assert.That(filled[i], Is.EqualTo(expectedValue),
                    $"Position {i} should have value '{expectedValue}' after filling nulls");
                Assert.That(filled.IsNull(i), Is.False,
                    $"Position {i} should not be null after filling");
            }
        }
    }

    [Test]
    public void NullValueFilling_EmptyColumn_ReturnsEmptyColumn()
    {
        // Feature: core-column-types, Property 19: Null value filling
        var emptyColumn = NivaraColumn<int>.Create(new int[0]);
        var filled = emptyColumn.FillNull(42);
        
        Assert.That(filled.Length, Is.EqualTo(0), "Filled empty column should remain empty");
        Assert.That(filled.HasNulls, Is.False, "Filled empty column should have no nulls");
    }

    #endregion
}