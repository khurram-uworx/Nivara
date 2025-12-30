using Nivara.Exceptions;
using NUnit.Framework;

namespace Nivara.Tests;

/// <summary>
/// Tests for TypeCompatibilityValidator functionality.
/// Validates comprehensive type checking and compatibility validation.
/// </summary>
[TestFixture]
public class TypeCompatibilityValidatorTests
{
    /// <summary>
    /// Property 13: Type compatibility validation
    /// For any two types, the validator should correctly determine arithmetic compatibility
    /// and provide clear error messages for incompatible operations.
    /// Validates: Requirements 7.1
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 13: Type compatibility validation")]
    public void Property_ArithmeticCompatibility_ValidatesCorrectly()
    {
        // Test cases for arithmetic compatibility
        var compatiblePairs = new[]
        {
            (typeof(int), typeof(int)),
            (typeof(int), typeof(long)),
            (typeof(float), typeof(double)),
            (typeof(byte), typeof(int)),
            (typeof(short), typeof(long)),
            (typeof(decimal), typeof(decimal))
        };

        var incompatiblePairs = new[]
        {
            (typeof(int), typeof(string)),
            (typeof(bool), typeof(int)),
            (typeof(DateTime), typeof(int)),
            (typeof(string), typeof(double)),
            (typeof(object), typeof(int))
        };

        // Property: Compatible numeric types should validate successfully
        foreach (var (leftType, rightType) in compatiblePairs)
        {
            Assert.DoesNotThrow(() =>
                TypeCompatibilityValidator.ValidateArithmeticCompatibility(leftType, rightType, "Test Operation"),
                $"Types {leftType.Name} and {rightType.Name} should be arithmetic compatible");

            Assert.That(TypeCompatibilityValidator.AreArithmeticCompatible(leftType, rightType), Is.True,
                $"Types {leftType.Name} and {rightType.Name} should be reported as arithmetic compatible");
        }

        // Property: Incompatible types should throw ColumnTypeMismatchException
        foreach (var (leftType, rightType) in incompatiblePairs)
        {
            Assert.Throws<ColumnTypeMismatchException>(() =>
                TypeCompatibilityValidator.ValidateArithmeticCompatibility(leftType, rightType, "Test Operation"),
                $"Types {leftType.Name} and {rightType.Name} should be arithmetic incompatible");

            Assert.That(TypeCompatibilityValidator.AreArithmeticCompatible(leftType, rightType), Is.False,
                $"Types {leftType.Name} and {rightType.Name} should be reported as arithmetic incompatible");
        }
    }

    /// <summary>
    /// Property 13: Type compatibility validation
    /// For any type, the validator should correctly determine comparison support
    /// and provide appropriate error messages for unsupported types.
    /// Validates: Requirements 7.1
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 13: Type compatibility validation")]
    public void Property_ComparisonSupport_ValidatesCorrectly()
    {
        var supportedTypes = new[]
        {
            typeof(int), typeof(long), typeof(float), typeof(double), typeof(decimal),
            typeof(string), typeof(DateTime), typeof(DateTimeOffset), typeof(bool),
            typeof(byte), typeof(short), typeof(uint), typeof(ulong), typeof(ushort), typeof(sbyte)
        };

        var unsupportedTypes = new[]
        {
            typeof(object)
        };

        // Property: Supported types should validate successfully
        foreach (var type in supportedTypes)
        {
            Assert.DoesNotThrow(() =>
                TypeCompatibilityValidator.ValidateComparisonSupport(type, "Test Comparison"),
                $"Type {type.Name} should support comparison operations");

            Assert.That(TypeCompatibilityValidator.SupportsComparison(type), Is.True,
                $"Type {type.Name} should be reported as supporting comparison");
        }

        // Property: Unsupported types should throw ColumnTypeMismatchException
        foreach (var type in unsupportedTypes)
        {
            Assert.Throws<ColumnTypeMismatchException>(() =>
                TypeCompatibilityValidator.ValidateComparisonSupport(type, "Test Comparison"),
                $"Type {type.Name} should not support comparison operations");

            Assert.That(TypeCompatibilityValidator.SupportsComparison(type), Is.False,
                $"Type {type.Name} should be reported as not supporting comparison");
        }
    }

    /// <summary>
    /// Property 13: Type compatibility validation
    /// For any two types, comparison compatibility should be symmetric and transitive
    /// where applicable, and provide consistent validation results.
    /// Validates: Requirements 7.1
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 13: Type compatibility validation")]
    public void Property_ComparisonCompatibility_IsSymmetricAndConsistent()
    {
        var testTypes = new[]
        {
            typeof(int), typeof(long), typeof(float), typeof(double),
            typeof(string), typeof(DateTime), typeof(bool)
        };

        // Property: Comparison compatibility should be symmetric
        foreach (var type1 in testTypes)
        {
            foreach (var type2 in testTypes)
            {
                var compatible1to2 = TypeCompatibilityValidator.AreComparisonCompatible(type1, type2);
                var compatible2to1 = TypeCompatibilityValidator.AreComparisonCompatible(type2, type1);

                Assert.That(compatible1to2, Is.EqualTo(compatible2to1),
                    $"Comparison compatibility between {type1.Name} and {type2.Name} should be symmetric");

                // If types are compatible, validation should not throw
                if (compatible1to2)
                {
                    Assert.DoesNotThrow(() =>
                        TypeCompatibilityValidator.ValidateComparisonCompatibility(type1, type2, "Test Comparison"),
                        $"Compatible types {type1.Name} and {type2.Name} should validate successfully");
                }
                else
                {
                    Assert.Throws<ColumnTypeMismatchException>(() =>
                        TypeCompatibilityValidator.ValidateComparisonCompatibility(type1, type2, "Test Comparison"),
                        $"Incompatible types {type1.Name} and {type2.Name} should throw exception");
                }
            }
        }
    }

    /// <summary>
    /// Property 13: Type compatibility validation
    /// For any frame with columns, type compatibility validation should correctly
    /// identify compatible and incompatible scenarios based on the requirement type.
    /// Validates: Requirements 7.1
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 13: Type compatibility validation")]
    public void Property_FrameTypeCompatibility_ValidatesRequirements()
    {
        // Create test frames with different type combinations
        var allNumericFrame = NivaraFrame.Create(
            ("IntCol", NivaraColumn<int>.Create(new[] { 1, 2, 3 })),
            ("DoubleCol", NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 })),
            ("FloatCol", NivaraColumn<float>.Create(new[] { 1.0f, 2.0f, 3.0f }))
        );

        var mixedTypeFrame = NivaraFrame.Create(
            ("IntCol", NivaraColumn<int>.Create(new[] { 1, 2, 3 })),
            ("StringCol", NivaraColumn<string>.Create(new[] { "a", "b", "c" })),
            ("BoolCol", NivaraColumn<bool>.Create(new[] { true, false, true }))
        );

        var sameTypeFrame = NivaraFrame.Create(
            ("IntCol1", NivaraColumn<int>.Create(new[] { 1, 2, 3 })),
            ("IntCol2", NivaraColumn<int>.Create(new[] { 4, 5, 6 })),
            ("IntCol3", NivaraColumn<int>.Create(new[] { 7, 8, 9 }))
        );

        try
        {
            // Property: AllNumeric requirement should pass for numeric frames
            Assert.DoesNotThrow(() =>
                TypeCompatibilityValidator.ValidateFrameTypeCompatibility(
                    allNumericFrame, "Numeric Operation", TypeCompatibilityRequirement.AllNumeric),
                "Frame with all numeric columns should pass AllNumeric validation");

            // Property: AllNumeric requirement should fail for mixed type frames
            Assert.Throws<SchemaValidationException>(() =>
                TypeCompatibilityValidator.ValidateFrameTypeCompatibility(
                    mixedTypeFrame, "Numeric Operation", TypeCompatibilityRequirement.AllNumeric),
                "Frame with mixed types should fail AllNumeric validation");

            // Property: AllSameType requirement should pass for same type frames
            Assert.DoesNotThrow(() =>
                TypeCompatibilityValidator.ValidateFrameTypeCompatibility(
                    sameTypeFrame, "Same Type Operation", TypeCompatibilityRequirement.AllSameType),
                "Frame with all same type columns should pass AllSameType validation");

            // Property: AllSameType requirement should fail for mixed type frames
            Assert.Throws<SchemaValidationException>(() =>
                TypeCompatibilityValidator.ValidateFrameTypeCompatibility(
                    mixedTypeFrame, "Same Type Operation", TypeCompatibilityRequirement.AllSameType),
                "Frame with mixed types should fail AllSameType validation");

            // Property: AllArithmeticCompatible should pass for numeric frames
            Assert.DoesNotThrow(() =>
                TypeCompatibilityValidator.ValidateFrameTypeCompatibility(
                    allNumericFrame, "Arithmetic Operation", TypeCompatibilityRequirement.AllArithmeticCompatible),
                "Frame with arithmetic compatible columns should pass AllArithmeticCompatible validation");
        }
        finally
        {
            // Clean up resources
            allNumericFrame.Dispose();
            mixedTypeFrame.Dispose();
            sameTypeFrame.Dispose();
        }
    }

    /// <summary>
    /// Property 13: Type compatibility validation
    /// For any operation with supported types, validation should succeed,
    /// and for unsupported types, should provide clear error messages.
    /// Validates: Requirements 7.1
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 13: Type compatibility validation")]
    public void Property_OperationSupport_ValidatesCorrectly()
    {
        var numericTypes = TypeCompatibilityValidator.GetNumericTypes();
        var supportedTypes = new[] { typeof(int), typeof(double), typeof(string) };
        var unsupportedTypes = new[] { typeof(bool), typeof(DateTime), typeof(object) };

        // Property: Supported types should validate successfully
        foreach (var supportedType in supportedTypes)
        {
            Assert.DoesNotThrow(() =>
                TypeCompatibilityValidator.ValidateOperationSupport(
                    supportedType, "Test Operation", supportedTypes),
                $"Supported type {supportedType.Name} should validate successfully");
        }

        // Property: Unsupported types should throw ColumnTypeMismatchException
        foreach (var unsupportedType in unsupportedTypes)
        {
            Assert.Throws<ColumnTypeMismatchException>(() =>
                TypeCompatibilityValidator.ValidateOperationSupport(
                    unsupportedType, "Test Operation", supportedTypes),
                $"Unsupported type {unsupportedType.Name} should throw exception");
        }

        // Property: Numeric types should be correctly identified
        var expectedNumericTypes = new[]
        {
            typeof(byte), typeof(sbyte), typeof(short), typeof(ushort),
            typeof(int), typeof(uint), typeof(long), typeof(ulong),
            typeof(float), typeof(double), typeof(decimal)
        };

        Assert.That(numericTypes.Count, Is.EqualTo(expectedNumericTypes.Length),
            "GetNumericTypes should return the expected number of numeric types");

        foreach (var expectedType in expectedNumericTypes)
        {
            Assert.That(numericTypes.Contains(expectedType), Is.True,
                $"GetNumericTypes should include {expectedType.Name}");
        }
    }

    /// <summary>
    /// Property 13: Type compatibility validation
    /// For any null inputs, validation methods should throw ArgumentNullException
    /// with appropriate parameter names.
    /// Validates: Requirements 7.1
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 13: Type compatibility validation")]
    public void Property_NullInputValidation_ThrowsAppropriateExceptions()
    {
        // Property: Null type parameters should throw ArgumentNullException
        Assert.Throws<ArgumentNullException>(() =>
            TypeCompatibilityValidator.ValidateArithmeticCompatibility(null!, typeof(int), "Test"),
            "Null leftType should throw ArgumentNullException");

        Assert.Throws<ArgumentNullException>(() =>
            TypeCompatibilityValidator.ValidateArithmeticCompatibility(typeof(int), null!, "Test"),
            "Null rightType should throw ArgumentNullException");

        Assert.Throws<ArgumentNullException>(() =>
            TypeCompatibilityValidator.ValidateComparisonSupport(null!, "Test"),
            "Null type should throw ArgumentNullException");

        Assert.Throws<ArgumentNullException>(() =>
            TypeCompatibilityValidator.ValidateFrameTypeCompatibility(null!, "Test", TypeCompatibilityRequirement.AllNumeric),
            "Null frame should throw ArgumentNullException");

        Assert.Throws<ArgumentNullException>(() =>
            TypeCompatibilityValidator.ValidateOperationSupport(typeof(int), "Test", null!),
            "Null supportedTypes should throw ArgumentNullException");
    }

    /// <summary>
    /// Property 13: Type compatibility validation
    /// For any compatible types, GetCompatibleTypes should return a collection
    /// that includes the original type and its compatible types.
    /// Validates: Requirements 7.1
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 13: Type compatibility validation")]
    public void Property_CompatibleTypes_IncludesOriginalType()
    {
        var testTypes = new[]
        {
            typeof(int), typeof(double), typeof(string), typeof(bool), typeof(DateTime)
        };

        // Property: GetCompatibleTypes should always include the original type
        foreach (var type in testTypes)
        {
            var compatibleTypes = TypeCompatibilityValidator.GetCompatibleTypes(type);

            Assert.That(compatibleTypes, Is.Not.Null,
                $"GetCompatibleTypes should not return null for {type.Name}");

            Assert.That(compatibleTypes.Contains(type), Is.True,
                $"GetCompatibleTypes should include the original type {type.Name}");

            Assert.That(compatibleTypes.Count, Is.GreaterThan(0),
                $"GetCompatibleTypes should return at least one type for {type.Name}");
        }

        // Property: Numeric types should have multiple compatible types
        var numericTypes = new[] { typeof(int), typeof(double), typeof(float), typeof(long) };

        foreach (var numericType in numericTypes)
        {
            var compatibleTypes = TypeCompatibilityValidator.GetCompatibleTypes(numericType);

            Assert.That(compatibleTypes.Count, Is.GreaterThan(1),
                $"Numeric type {numericType.Name} should have multiple compatible types");
        }

        // Property: Non-numeric types should typically only be compatible with themselves
        var nonNumericTypes = new[] { typeof(string), typeof(bool), typeof(DateTime) };

        foreach (var nonNumericType in nonNumericTypes)
        {
            var compatibleTypes = TypeCompatibilityValidator.GetCompatibleTypes(nonNumericType);

            Assert.That(compatibleTypes.Count, Is.EqualTo(1),
                $"Non-numeric type {nonNumericType.Name} should typically only be compatible with itself");

            Assert.That(compatibleTypes[0], Is.EqualTo(nonNumericType),
                $"Non-numeric type {nonNumericType.Name} should be compatible with itself");
        }
    }
}