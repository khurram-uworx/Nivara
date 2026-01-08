using Apache.Arrow.Types;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class TypeMapperTests
{
    [Test]
    public void MapClrToArrow_SupportedPrimitiveTypes_ReturnsCorrectArrowTypes()
    {
        // Test basic primitive types
        var testCases = new[]
        {
            (typeof(bool), typeof(BooleanType)),
            (typeof(int), typeof(Int32Type)),
            (typeof(long), typeof(Int64Type)),
            (typeof(float), typeof(FloatType)),
            (typeof(double), typeof(DoubleType)),
            (typeof(string), typeof(StringType)),
            (typeof(byte), typeof(UInt8Type)),
            (typeof(short), typeof(Int16Type)),
            (typeof(uint), typeof(UInt32Type)),
            (typeof(ulong), typeof(UInt64Type)),
            (typeof(ushort), typeof(UInt16Type)),
            (typeof(sbyte), typeof(Int8Type))
        };

        foreach (var (clrType, expectedArrowType) in testCases)
        {
            var result = TypeMapper.MapClrToArrow(clrType);
            Assert.That(result.GetType(), Is.EqualTo(expectedArrowType),
                $"CLR type {clrType.Name} should map to Arrow type {expectedArrowType.Name}");
        }
    }

    [Test]
    public void MapClrToArrow_DateTime_ReturnsTimestampType()
    {
        var result = TypeMapper.MapClrToArrow(typeof(DateTime));

        Assert.That(result, Is.InstanceOf<TimestampType>());
        var timestampType = (TimestampType)result;
        Assert.That(timestampType.Unit, Is.EqualTo(TimeUnit.Microsecond));
        // The timezone format may vary, but should represent UTC
        Assert.That(timestampType.Timezone, Is.EqualTo("+00:00").Or.EqualTo("UTC"));
    }

    [Test]
    public void MapClrToArrow_NullableTypes_HandlesCorrectly()
    {
        // Test nullable value types
        var testCases = new[]
        {
            (typeof(int?), typeof(Int32Type)),
            (typeof(bool?), typeof(BooleanType)),
            (typeof(DateTime?), typeof(TimestampType)),
            (typeof(double?), typeof(DoubleType))
        };

        foreach (var (nullableType, expectedArrowType) in testCases)
        {
            var result = TypeMapper.MapClrToArrow(nullableType);
            Assert.That(result.GetType(), Is.EqualTo(expectedArrowType),
                $"Nullable type {nullableType.Name} should map to Arrow type {expectedArrowType.Name}");
        }
    }

    [Test]
    public void MapClrToArrow_UnsupportedType_ThrowsUnsupportedTypeException()
    {
        var unsupportedTypes = new[]
        {
            typeof(Guid),
            typeof(TimeSpan),
            typeof(DateOnly),
            typeof(char),
            typeof(object)
        };

        foreach (var unsupportedType in unsupportedTypes)
        {
            var ex = Assert.Throws<UnsupportedTypeException>(() => TypeMapper.MapClrToArrow(unsupportedType));
            Assert.That(ex!.UnsupportedType, Is.EqualTo(unsupportedType));
            Assert.That(ex.SuggestedAlternatives, Is.Not.Empty,
                $"Should provide suggestions for unsupported type {unsupportedType.Name}");
        }
    }

    [Test]
    public void MapArrowToClr_SupportedArrowTypes_ReturnsCorrectClrTypes()
    {
        var testCases = new (IArrowType ArrowType, Type ExpectedClrType)[]
        {
            (BooleanType.Default, typeof(bool)),
            (Int32Type.Default, typeof(int)),
            (Int64Type.Default, typeof(long)),
            (FloatType.Default, typeof(float)),
            (DoubleType.Default, typeof(double)),
            (StringType.Default, typeof(string)),
            (UInt8Type.Default, typeof(byte)),
            (Int16Type.Default, typeof(short))
        };

        foreach (var (arrowType, expectedClrType) in testCases)
        {
            var result = TypeMapper.MapArrowToClr(arrowType);
            Assert.That(result, Is.EqualTo(expectedClrType),
                $"Arrow type {arrowType.GetType().Name} should map to CLR type {expectedClrType.Name}");
        }
    }

    [Test]
    public void MapArrowToClr_TimestampType_ReturnsDateTime()
    {
        var timestampType = new TimestampType(TimeUnit.Microsecond, TimeZoneInfo.Utc);
        var result = TypeMapper.MapArrowToClr(timestampType);

        Assert.That(result, Is.EqualTo(typeof(DateTime)));
    }

    [Test]
    public void CreateParquetField_SupportedTypes_CreatesCorrectFields()
    {
        var testCases = new[]
        {
            (typeof(bool), "BoolField"),
            (typeof(int), "IntField"),
            (typeof(long), "LongField"),
            (typeof(float), "FloatField"),
            (typeof(double), "DoubleField"),
            (typeof(string), "StringField"),
            (typeof(DateTime), "DateTimeField")
        };

        foreach (var (clrType, fieldName) in testCases)
        {
            var field = TypeMapper.CreateParquetField(fieldName, clrType);

            Assert.That(field.Name, Is.EqualTo(fieldName));
            Assert.That(field.ClrType, Is.EqualTo(clrType));
        }
    }

    [Test]
    public void CreateParquetField_NullableTypes_SetsNullabilityCorrectly()
    {
        // Value types should be non-nullable by default
        var nonNullableField = TypeMapper.CreateParquetField("IntField", typeof(int));
        Assert.That(nonNullableField.IsNullable, Is.False);

        // Nullable value types should be nullable
        var nullableField = TypeMapper.CreateParquetField("NullableIntField", typeof(int?));
        Assert.That(nullableField.IsNullable, Is.True);

        // Reference types should be nullable by default
        var stringField = TypeMapper.CreateParquetField("StringField", typeof(string));
        Assert.That(stringField.IsNullable, Is.True);
    }

    [Test]
    public void CreateParquetField_UnsupportedType_ThrowsUnsupportedTypeException()
    {
        var ex = Assert.Throws<UnsupportedTypeException>(() =>
            TypeMapper.CreateParquetField("UnsupportedField", typeof(Guid)));

        Assert.That(ex!.UnsupportedType, Is.EqualTo(typeof(Guid)));
        Assert.That(ex.SuggestedAlternatives, Contains.Item("string"));
        Assert.That(ex.SuggestedAlternatives, Contains.Item("byte[]"));
    }

    [Test]
    public void IsArrowSupported_SupportedTypes_ReturnsTrue()
    {
        var supportedTypes = new[]
        {
            typeof(bool), typeof(int), typeof(long), typeof(float), typeof(double),
            typeof(string), typeof(DateTime), typeof(byte), typeof(short),
            typeof(bool?), typeof(int?), typeof(DateTime?)
        };

        foreach (var type in supportedTypes)
        {
            Assert.That(TypeMapper.IsArrowSupported(type), Is.True,
                $"Type {type.Name} should be supported for Arrow conversion");
        }
    }

    [Test]
    public void IsArrowSupported_UnsupportedTypes_ReturnsFalse()
    {
        var unsupportedTypes = new[]
        {
            typeof(Guid), typeof(TimeSpan), typeof(DateOnly), typeof(char), typeof(object)
        };

        foreach (var type in unsupportedTypes)
        {
            Assert.That(TypeMapper.IsArrowSupported(type), Is.False,
                $"Type {type.Name} should not be supported for Arrow conversion");
        }
    }

    [Test]
    public void IsParquetSupported_SupportedTypes_ReturnsTrue()
    {
        var supportedTypes = new[]
        {
            typeof(bool), typeof(int), typeof(long), typeof(float), typeof(double),
            typeof(string), typeof(DateTime), typeof(byte), typeof(short), typeof(decimal),
            typeof(bool?), typeof(int?), typeof(decimal?)
        };

        foreach (var type in supportedTypes)
        {
            Assert.That(TypeMapper.IsParquetSupported(type), Is.True,
                $"Type {type.Name} should be supported for Parquet conversion");
        }
    }

    [Test]
    public void IsParquetSupported_UnsupportedTypes_ReturnsFalse()
    {
        var unsupportedTypes = new[]
        {
            typeof(Guid), typeof(TimeSpan), typeof(DateOnly), typeof(char), typeof(object)
        };

        foreach (var type in unsupportedTypes)
        {
            Assert.That(TypeMapper.IsParquetSupported(type), Is.False,
                $"Type {type.Name} should not be supported for Parquet conversion");
        }
    }

    [Test]
    public void TypeMapping_SpecialCases_HandlesCorrectly()
    {
        // Test Guid suggestions
        var guidEx = Assert.Throws<UnsupportedTypeException>(() => TypeMapper.MapClrToArrow(typeof(Guid)));
        Assert.That(guidEx!.SuggestedAlternatives, Contains.Item("string"));
        Assert.That(guidEx.SuggestedAlternatives, Contains.Item("byte[]"));

        // Test TimeSpan suggestions
        var timeSpanEx = Assert.Throws<UnsupportedTypeException>(() => TypeMapper.MapClrToArrow(typeof(TimeSpan)));
        Assert.That(timeSpanEx!.SuggestedAlternatives, Contains.Item("long (ticks)"));
        Assert.That(timeSpanEx.SuggestedAlternatives, Contains.Item("double (seconds)"));

        // Test enum suggestions
        var enumEx = Assert.Throws<UnsupportedTypeException>(() => TypeMapper.MapClrToArrow(typeof(DayOfWeek)));
        Assert.That(enumEx!.SuggestedAlternatives, Contains.Item("int"));
        Assert.That(enumEx.SuggestedAlternatives, Contains.Item("string"));
    }

    [Test]
    public void TypeMapping_RoundTrip_PreservesTypeInformation()
    {
        // Test that mapping CLR -> Arrow -> CLR preserves the original type for supported types
        var supportedTypes = new[]
        {
            typeof(bool), typeof(int), typeof(long), typeof(float), typeof(double), typeof(string)
        };

        foreach (var originalType in supportedTypes)
        {
            var arrowType = TypeMapper.MapClrToArrow(originalType);
            var roundTripType = TypeMapper.MapArrowToClr(arrowType);

            Assert.That(roundTripType, Is.EqualTo(originalType),
                $"Round-trip mapping should preserve type {originalType.Name}");
        }
    }

    #region Property-Based Tests

    /// <summary>
    /// Property 5: Type Mapping Consistency
    /// For any supported CLR type, the type mapping should be bidirectional and consistent across Arrow and Parquet formats
    /// **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
    /// </summary>
    [Test]
    public void TypeMappingConsistency_AllSupportedTypes_MaintainsBidirectionalMapping()
    {
        // Feature: arrow-parquet-io, Property 5: Type Mapping Consistency
        
        // Test all supported types for Arrow mapping consistency
        var supportedTypes = TypeMapper.GetSupportedTypes().ToArray();
        
        foreach (var clrType in supportedTypes)
        {
            // Test CLR -> Arrow -> CLR round-trip consistency
            var arrowType = TypeMapper.MapClrToArrow(clrType);
            var roundTripClrType = TypeMapper.MapArrowToClr(arrowType);
            
            Assert.That(roundTripClrType, Is.EqualTo(clrType),
                $"CLR type {clrType.Name} should round-trip through Arrow mapping consistently");
            
            // Test that Arrow support detection is consistent
            Assert.That(TypeMapper.IsArrowSupported(clrType), Is.True,
                $"Type {clrType.Name} should be detected as Arrow-supported");
        }
    }

    [Test]
    public void TypeMappingConsistency_NullableTypes_PreservesUnderlyingTypeMapping()
    {
        // Feature: arrow-parquet-io, Property 5: Type Mapping Consistency
        
        var valueTypes = new[]
        {
            typeof(bool), typeof(int), typeof(long), typeof(float), typeof(double), typeof(DateTime),
            typeof(byte), typeof(short), typeof(uint), typeof(ulong), typeof(ushort), typeof(sbyte)
        };
        
        foreach (var valueType in valueTypes)
        {
            if (!TypeMapper.IsArrowSupported(valueType)) continue;
            
            // Create nullable version of the type
            var nullableType = typeof(Nullable<>).MakeGenericType(valueType);
            
            // Both nullable and non-nullable should map to the same Arrow type
            var nonNullableArrowType = TypeMapper.MapClrToArrow(valueType);
            var nullableArrowType = TypeMapper.MapClrToArrow(nullableType);
            
            Assert.That(nullableArrowType.GetType(), Is.EqualTo(nonNullableArrowType.GetType()),
                $"Nullable type {nullableType.Name} should map to same Arrow type as {valueType.Name}");
            
            // Both should be detected as Arrow-supported
            Assert.That(TypeMapper.IsArrowSupported(nullableType), Is.True,
                $"Nullable type {nullableType.Name} should be detected as Arrow-supported");
        }
    }

    [Test]
    public void TypeMappingConsistency_ParquetSupport_AlignedWithArrowSupport()
    {
        // Feature: arrow-parquet-io, Property 5: Type Mapping Consistency
        
        var testTypes = new[]
        {
            // Supported types
            typeof(bool), typeof(int), typeof(long), typeof(float), typeof(double), 
            typeof(string), typeof(DateTime), typeof(byte), typeof(short),
            typeof(bool?), typeof(int?), typeof(DateTime?),
            
            // Parquet-only types
            typeof(decimal), typeof(decimal?),
            
            // Unsupported types
            typeof(Guid), typeof(TimeSpan), typeof(char), typeof(object)
        };
        
        foreach (var type in testTypes)
        {
            bool isArrowSupported = TypeMapper.IsArrowSupported(type);
            bool isParquetSupported = TypeMapper.IsParquetSupported(type);
            
            // For types that are Arrow-supported, verify they can create Parquet fields (except for some edge cases)
            if (isArrowSupported && type != typeof(uint) && type != typeof(ulong) && type != typeof(ushort) && type != typeof(sbyte))
            {
                Assert.DoesNotThrow(() => TypeMapper.CreateParquetField("TestField", type),
                    $"Arrow-supported type {type.Name} should be able to create Parquet fields");
            }
            
            // Verify consistency in support detection
            if (type == typeof(decimal) || type == typeof(decimal?))
            {
                // Decimal is Parquet-supported but not Arrow-supported
                Assert.That(isParquetSupported, Is.True, $"Decimal type {type.Name} should be Parquet-supported");
                Assert.That(isArrowSupported, Is.False, $"Decimal type {type.Name} should not be Arrow-supported");
            }
            else if (isArrowSupported)
            {
                // Most Arrow-supported types should also be Parquet-supported
                Assert.That(isParquetSupported, Is.True,
                    $"Arrow-supported type {type.Name} should generally be Parquet-supported");
            }
        }
    }

    [Test]
    public void TypeMappingConsistency_UnsupportedTypes_ProvideConsistentErrorMessages()
    {
        // Feature: arrow-parquet-io, Property 5: Type Mapping Consistency
        
        var unsupportedTypes = new[]
        {
            typeof(Guid), typeof(TimeSpan), typeof(DateOnly), typeof(TimeOnly),
            typeof(char), typeof(object), typeof(DayOfWeek), typeof(int[])
        };
        
        foreach (var unsupportedType in unsupportedTypes)
        {
            // Verify Arrow mapping throws UnsupportedTypeException
            var arrowEx = Assert.Throws<UnsupportedTypeException>(() => TypeMapper.MapClrToArrow(unsupportedType));
            Assert.That(arrowEx!.UnsupportedType, Is.EqualTo(unsupportedType),
                $"Arrow exception should reference the correct unsupported type {unsupportedType.Name}");
            Assert.That(arrowEx.SuggestedAlternatives, Is.Not.Empty,
                $"Arrow exception should provide suggestions for {unsupportedType.Name}");
            
            // Verify Parquet field creation throws UnsupportedTypeException (for most types)
            if (unsupportedType != typeof(int[])) // Arrays might have different handling
            {
                var parquetEx = Assert.Throws<UnsupportedTypeException>(() => 
                    TypeMapper.CreateParquetField("TestField", unsupportedType));
                Assert.That(parquetEx!.UnsupportedType, Is.EqualTo(unsupportedType),
                    $"Parquet exception should reference the correct unsupported type {unsupportedType.Name}");
            }
            
            // Verify support detection is consistent
            Assert.That(TypeMapper.IsArrowSupported(unsupportedType), Is.False,
                $"Type {unsupportedType.Name} should be detected as Arrow-unsupported");
            Assert.That(TypeMapper.IsParquetSupported(unsupportedType), Is.False,
                $"Type {unsupportedType.Name} should be detected as Parquet-unsupported");
        }
    }

    [Test]
    public void TypeMappingConsistency_SpecialCases_HandleTimestampTypesCorrectly()
    {
        // Feature: arrow-parquet-io, Property 5: Type Mapping Consistency
        
        // Test DateTime mapping produces consistent TimestampType
        var dateTimeArrowType = TypeMapper.MapClrToArrow(typeof(DateTime));
        Assert.That(dateTimeArrowType, Is.InstanceOf<TimestampType>(),
            "DateTime should map to TimestampType");
        
        var timestampType = (TimestampType)dateTimeArrowType;
        Assert.That(timestampType.Unit, Is.EqualTo(TimeUnit.Microsecond),
            "DateTime should map to microsecond precision TimestampType");
        
        // Test that different TimestampType instances map back to DateTime
        var timestampTypes = new[]
        {
            new TimestampType(TimeUnit.Microsecond, TimeZoneInfo.Utc),
            new TimestampType(TimeUnit.Millisecond, TimeZoneInfo.Local),
            new TimestampType(TimeUnit.Second, (TimeZoneInfo?)null)
        };
        
        foreach (var timestampType2 in timestampTypes)
        {
            var mappedClrType = TypeMapper.MapArrowToClr(timestampType2);
            Assert.That(mappedClrType, Is.EqualTo(typeof(DateTime)),
                $"TimestampType with {timestampType2.Unit} should map to DateTime");
        }
    }

    [Test]
    public void TypeMappingConsistency_ParquetFieldCreation_HandlesNullabilityCorrectly()
    {
        // Feature: arrow-parquet-io, Property 5: Type Mapping Consistency
        
        var testCases = new[]
        {
            // (Type, ExpectedNullability)
            (typeof(int), false),      // Value type - non-nullable
            (typeof(int?), true),      // Nullable value type - nullable
            (typeof(string), true),    // Reference type - nullable
            (typeof(bool), false),     // Value type - non-nullable
            (typeof(bool?), true),     // Nullable value type - nullable
            (typeof(DateTime), false), // Value type - non-nullable
            (typeof(DateTime?), true)  // Nullable value type - nullable
        };
        
        foreach (var (type, expectedNullable) in testCases)
        {
            if (!TypeMapper.IsParquetSupported(type)) continue;
            
            var field = TypeMapper.CreateParquetField("TestField", type);
            
            Assert.That(field.IsNullable, Is.EqualTo(expectedNullable),
                $"Type {type.Name} should have nullability {expectedNullable} in Parquet field");
            
            // Verify the CLR type is preserved correctly
            var expectedClrType = Nullable.GetUnderlyingType(type) ?? type;
            Assert.That(field.ClrType, Is.EqualTo(expectedClrType),
                $"Parquet field should preserve underlying CLR type for {type.Name}");
        }
    }

    #endregion
}
