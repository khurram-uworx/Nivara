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
}
