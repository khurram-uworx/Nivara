using Apache.Arrow;
using Apache.Arrow.Types;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class ArrowExtendedDomainRoundTripTests
{
    [Test]
    public void RoundTrip_DateOnlyColumn_PreservesValuesAndNulls()
    {
        var values = new DateOnly?[] { new(2023, 1, 15), null, new(2024, 12, 31) };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("DateColumn", column));

        var result = ArrowInterop.FromArrowTable(ArrowInterop.ToArrowTable(frame));

        var resultColumn = result.GetColumn<DateOnly>("DateColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
                Assert.That(resultColumn[i], Is.EqualTo(values[i]));
        }
    }

    [Test]
    public void RoundTrip_GuidColumn_PreservesValuesAndNulls()
    {
        var values = new Guid?[] { Guid.NewGuid(), null, Guid.NewGuid() };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("GuidColumn", column));

        var result = ArrowInterop.FromArrowTable(ArrowInterop.ToArrowTable(frame));

        var resultColumn = result.GetColumn<Guid>("GuidColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
                Assert.That(resultColumn[i], Is.EqualTo(values[i]));
        }
    }

    [Test]
    public void RoundTrip_HalfColumn_PreservesValuesAndNulls()
    {
        var values = new Half?[] { (Half)1.5f, null, (Half)2.25f, (Half)(-3.5f) };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("HalfColumn", column));

        var result = ArrowInterop.FromArrowTable(ArrowInterop.ToArrowTable(frame));

        var resultColumn = result.GetColumn<Half>("HalfColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
                Assert.That(resultColumn[i], Is.EqualTo(values[i]));
        }
    }

    [Test]
    public void RoundTrip_TimeOnlyColumn_PreservesValuesAndNulls()
    {
        var values = new TimeOnly?[] { new(14, 30, 45, 123), null, new(0, 0, 0) };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("TimeColumn", column));

        var result = ArrowInterop.FromArrowTable(ArrowInterop.ToArrowTable(frame));

        var resultColumn = result.GetColumn<TimeOnly>("TimeColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
                Assert.That(resultColumn[i], Is.EqualTo(values[i]));
        }
    }

    [Test]
    public void RoundTrip_TimeSpanColumn_PreservesValuesAndNulls()
    {
        var values = new TimeSpan?[] { TimeSpan.FromTicks(123456789L), null, TimeSpan.Zero, TimeSpan.FromDays(2) };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("SpanColumn", column));

        var result = ArrowInterop.FromArrowTable(ArrowInterop.ToArrowTable(frame));

        var resultColumn = result.GetColumn<TimeSpan>("SpanColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
                Assert.That(resultColumn[i], Is.EqualTo(values[i]));
        }
    }

    [Test]
    public void RoundTrip_NIntColumn_RestoresOriginalTypeViaMetadata()
    {
        var values = new nint?[] { (nint)42, null, (nint)(-17), nint.MinValue };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("NIntColumn", column));

        var result = ArrowInterop.FromArrowTable(ArrowInterop.ToArrowTable(frame));

        var resultColumn = result.GetColumn<nint>("NIntColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
                Assert.That(resultColumn[i], Is.EqualTo(values[i]));
        }
    }

    [Test]
    public void RoundTrip_NUIntColumn_RestoresOriginalTypeViaMetadata()
    {
        var values = new nuint?[] { (nuint)99, null, (nuint)0 };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("NUIntColumn", column));

        var result = ArrowInterop.FromArrowTable(ArrowInterop.ToArrowTable(frame));

        var resultColumn = result.GetColumn<nuint>("NUIntColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
                Assert.That(resultColumn[i], Is.EqualTo(values[i]));
        }
    }

    [Test]
    public void RoundTrip_CharColumn_RestoresOriginalTypeViaMetadata()
    {
        var values = new char?[] { 'A', null, 'Ω', '€' };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("CharColumn", column));

        var result = ArrowInterop.FromArrowTable(ArrowInterop.ToArrowTable(frame));

        var resultColumn = result.GetColumn<char>("CharColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
                Assert.That(resultColumn[i], Is.EqualTo(values[i]));
        }
    }

    [Test]
    public void RoundTrip_DateTimeOffsetColumn_PreservesUtcInstant()
    {
        var values = new DateTimeOffset?[]
        {
            new DateTimeOffset(2023, 6, 15, 10, 30, 0, TimeSpan.FromHours(5)),
            null,
            new DateTimeOffset(2024, 1, 1, 0, 0, 0, TimeSpan.Zero)
        };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("OffsetColumn", column));

        var result = ArrowInterop.FromArrowTable(ArrowInterop.ToArrowTable(frame));

        var resultColumn = result.GetColumn<DateTimeOffset>("OffsetColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
            {
                // The UTC instant is preserved; the offset is normalized to UTC
                Assert.That(resultColumn[i].UtcTicks, Is.EqualTo(values[i]!.Value.UtcTicks));
                Assert.That(resultColumn[i].Offset, Is.EqualTo(TimeSpan.Zero));
            }
        }
    }

    [Test]
    public void RoundTrip_MixedExtendedDomainFrame_RestoresAllTypes()
    {
        var frame = NivaraFrame.Create(
            ("HalfColumn", NivaraColumn<Half>.Create(new[] { (Half)1.5f, (Half)2.5f })),
            ("GuidColumn", NivaraColumn<Guid>.Create(new[] { Guid.NewGuid(), Guid.NewGuid() })),
            ("DateColumn", NivaraColumn<DateOnly>.Create(new[] { new DateOnly(2023, 1, 1), new DateOnly(2023, 1, 2) })),
            ("TimeColumn", NivaraColumn<TimeOnly>.Create(new[] { new TimeOnly(1, 2, 3), new TimeOnly(4, 5, 6) })),
            ("SpanColumn", NivaraColumn<TimeSpan>.Create(new[] { TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(2) })),
            ("NIntColumn", NivaraColumn<nint>.Create(new[] { (nint)1, (nint)2 })),
            ("NUIntColumn", NivaraColumn<nuint>.Create(new[] { (nuint)1, (nuint)2 })),
            ("CharColumn", NivaraColumn<char>.Create(new[] { 'a', 'b' })),
            ("OffsetColumn", NivaraColumn<DateTimeOffset>.Create(new[] { new DateTimeOffset(2024, 1, 1, 0, 0, 0, TimeSpan.Zero), new DateTimeOffset(2024, 6, 15, 10, 30, 0, 500, TimeSpan.Zero) }))
        );

        var result = ArrowInterop.FromArrowTable(ArrowInterop.ToArrowTable(frame));

        Assert.That(result.GetColumn<Half>("HalfColumn")[0], Is.EqualTo((Half)1.5f));
        Assert.That(result.GetColumn<Guid>("GuidColumn")[0], Is.EqualTo(frame.GetColumn<Guid>("GuidColumn")[0]));
        Assert.That(result.GetColumn<DateOnly>("DateColumn")[0], Is.EqualTo(new DateOnly(2023, 1, 1)));
        Assert.That(result.GetColumn<TimeOnly>("TimeColumn")[0], Is.EqualTo(new TimeOnly(1, 2, 3)));
        Assert.That(result.GetColumn<TimeSpan>("SpanColumn")[0], Is.EqualTo(TimeSpan.FromSeconds(1)));
        Assert.That(result.GetColumn<nint>("NIntColumn")[0], Is.EqualTo((nint)1));
        Assert.That(result.GetColumn<nuint>("NUIntColumn")[0], Is.EqualTo((nuint)1));
        Assert.That(result.GetColumn<char>("CharColumn")[0], Is.EqualTo('a'));
        Assert.That(result.GetColumn<DateTimeOffset>("OffsetColumn")[0].UtcTicks,
            Is.EqualTo(frame.GetColumn<DateTimeOffset>("OffsetColumn")[0].UtcTicks));
    }

    [Test]
    public void RoundTrip_SeriesLevel_ExtendedTypes_RoundTripCorrectly()
    {
        var halfSeries = NivaraSeries<Half>.Create(new[] { (Half)1.5f, (Half)(-2.25f) });
        var halfRoundTrip = ArrowInterop.FromArrowArray<Half>(ArrowInterop.ToArrowArray(halfSeries));
        Assert.That(halfRoundTrip[0], Is.EqualTo((Half)1.5f));
        Assert.That(halfRoundTrip[1], Is.EqualTo((Half)(-2.25f)));

        var dateSeries = NivaraSeries<DateOnly>.Create(new[] { new DateOnly(2024, 5, 6), new DateOnly(2024, 5, 7) });
        var dateRoundTrip = ArrowInterop.FromArrowArray<DateOnly>(ArrowInterop.ToArrowArray(dateSeries));
        Assert.That(dateRoundTrip[0], Is.EqualTo(new DateOnly(2024, 5, 6)));

        var timeSeries = NivaraSeries<TimeOnly>.Create(new[] { new TimeOnly(23, 59, 59), new TimeOnly(0, 0, 1) });
        var timeRoundTrip = ArrowInterop.FromArrowArray<TimeOnly>(ArrowInterop.ToArrowArray(timeSeries));
        Assert.That(timeRoundTrip[1], Is.EqualTo(new TimeOnly(0, 0, 1)));

        var guidSeries = NivaraSeries<Guid>.Create(new[] { Guid.NewGuid(), Guid.NewGuid() });
        var guidRoundTrip = ArrowInterop.FromArrowArray<Guid>(ArrowInterop.ToArrowArray(guidSeries));
        Assert.That(guidRoundTrip[0], Is.EqualTo(guidSeries[0]));

        var spanSeries = NivaraSeries<TimeSpan>.Create(new[] { TimeSpan.FromMinutes(90), TimeSpan.Zero });
        var spanRoundTrip = ArrowInterop.FromArrowArray<TimeSpan>(ArrowInterop.ToArrowArray(spanSeries));
        Assert.That(spanRoundTrip[0], Is.EqualTo(TimeSpan.FromMinutes(90)));

        var nintSeries = NivaraSeries<nint>.Create(new[] { (nint)7, (nint)(-7) });
        var nintRoundTrip = ArrowInterop.FromArrowArray<nint>(ArrowInterop.ToArrowArray(nintSeries));
        Assert.That(nintRoundTrip[1], Is.EqualTo((nint)(-7)));

        var nuintSeries = NivaraSeries<nuint>.Create(new[] { (nuint)7, (nuint)0 });
        var nuintRoundTrip = ArrowInterop.FromArrowArray<nuint>(ArrowInterop.ToArrowArray(nuintSeries));
        Assert.That(nuintRoundTrip[0], Is.EqualTo((nuint)7));

        var charSeries = NivaraSeries<char>.Create(new[] { 'z', 'Z' });
        var charRoundTrip = ArrowInterop.FromArrowArray<char>(ArrowInterop.ToArrowArray(charSeries));
        Assert.That(charRoundTrip[1], Is.EqualTo('Z'));

        var offsetSeries = NivaraSeries<DateTimeOffset>.Create(new[] { new DateTimeOffset(2024, 2, 29, 12, 0, 0, TimeSpan.FromHours(3)) });
        var offsetRoundTrip = ArrowInterop.FromArrowArray<DateTimeOffset>(ArrowInterop.ToArrowArray(offsetSeries));
        Assert.That(offsetRoundTrip[0].UtcTicks, Is.EqualTo(offsetSeries[0].UtcTicks));
    }

    [Test]
    public void ToArrowTable_ExtendedTypes_MapToExpectedArrowTypes()
    {
        var frame = NivaraFrame.Create(
            ("HalfColumn", NivaraColumn<Half>.Create(new[] { (Half)1.5f })),
            ("GuidColumn", NivaraColumn<Guid>.Create(new[] { Guid.NewGuid() })),
            ("DateColumn", NivaraColumn<DateOnly>.Create(new[] { new DateOnly(2024, 1, 1) })),
            ("TimeColumn", NivaraColumn<TimeOnly>.Create(new[] { new TimeOnly(1, 2, 3) })),
            ("SpanColumn", NivaraColumn<TimeSpan>.Create(new[] { TimeSpan.FromSeconds(1) })),
            ("NIntColumn", NivaraColumn<nint>.Create(new[] { (nint)1 })),
            ("NUIntColumn", NivaraColumn<nuint>.Create(new[] { (nuint)1 })),
            ("CharColumn", NivaraColumn<char>.Create(new[] { 'a' })),
            ("OffsetColumn", NivaraColumn<DateTimeOffset>.Create(new[] { DateTimeOffset.UtcNow }))
        );

        var table = ArrowInterop.ToArrowTable(frame);
        var schema = table.Schema;

        Assert.That(schema.GetFieldByIndex(0).DataType, Is.InstanceOf<HalfFloatType>());
        Assert.That(schema.GetFieldByIndex(1).DataType, Is.InstanceOf<FixedSizeBinaryType>());
        Assert.That(schema.GetFieldByIndex(2).DataType, Is.InstanceOf<Date32Type>());
        Assert.That(schema.GetFieldByIndex(3).DataType, Is.InstanceOf<Time64Type>());
        Assert.That(schema.GetFieldByIndex(4).DataType, Is.InstanceOf<DurationType>());
        Assert.That(schema.GetFieldByIndex(5).DataType, Is.InstanceOf<Int64Type>());
        Assert.That(schema.GetFieldByIndex(6).DataType, Is.InstanceOf<UInt64Type>());
        Assert.That(schema.GetFieldByIndex(7).DataType, Is.InstanceOf<StringType>());
        Assert.That(schema.GetFieldByIndex(8).DataType, Is.InstanceOf<TimestampType>());
    }

    [Test]
    public void ToArrowTable_WritesClrTypeMetadata_ForExtendedDomainColumns()
    {
        var frame = NivaraFrame.Create(
            ("HalfColumn", NivaraColumn<Half>.Create(new[] { (Half)1.5f })),
            ("NIntColumn", NivaraColumn<nint>.Create(new[] { (nint)1 })),
            ("OffsetColumn", NivaraColumn<DateTimeOffset>.Create(new[] { DateTimeOffset.UtcNow })),
            ("IntColumn", NivaraColumn<int>.Create(new[] { 1 }))
        );

        var table = ArrowInterop.ToArrowTable(frame);
        var metadata = table.Schema.Metadata;

        Assert.That(metadata, Is.Not.Null);
        Assert.That(metadata!.TryGetValue(TypeMapper.GetClrTypeMetadataKey("HalfColumn"), out var halfType), Is.True);
        Assert.That(halfType, Is.EqualTo("System.Half"));
        Assert.That(metadata.TryGetValue(TypeMapper.GetClrTypeMetadataKey("NIntColumn"), out var nintType), Is.True);
        Assert.That(nintType, Is.EqualTo("System.IntPtr"));
        Assert.That(metadata.TryGetValue(TypeMapper.GetClrTypeMetadataKey("OffsetColumn"), out _), Is.True);
        Assert.That(metadata.ContainsKey(TypeMapper.GetClrTypeMetadataKey("IntColumn")), Is.False,
            "Base-domain columns should not carry clrType metadata");
    }

    [Test]
    public void FromArrowTable_NoMetadata_ReadsBaseArrowType()
    {
        // A foreign Arrow file has no Nivara clrType metadata: Int64 reads back as long,
        // UInt64 as ulong, String as string, Timestamp as DateTime.
        var longBuilder = new Int64Array.Builder();
        longBuilder.Append(42);
        var longArray = longBuilder.Build();

        var timestampBuilder = new TimestampArray.Builder(new TimestampType(TimeUnit.Microsecond, TimeZoneInfo.Utc));
        timestampBuilder.Append(new DateTimeOffset(2024, 3, 1, 12, 0, 0, TimeSpan.Zero));
        var timestampArray = timestampBuilder.Build();

        var schema = new Apache.Arrow.Schema(
            new[] { new Field("NIntColumn", Int64Type.Default, true), new Field("OffsetColumn", new TimestampType(TimeUnit.Microsecond, TimeZoneInfo.Utc), true) },
            null);
        var recordBatch = new RecordBatch(schema, new IArrowArray[] { longArray, timestampArray }, 1);
        var table = Table.TableFromRecordBatches(schema, new[] { recordBatch });

        var result = ArrowInterop.FromArrowTable(table);

        Assert.That(result.GetColumn("NIntColumn").ElementType, Is.EqualTo(typeof(long)));
        Assert.That(result.GetColumn<long>("NIntColumn")[0], Is.EqualTo(42L));
        Assert.That(result.GetColumn("OffsetColumn").ElementType, Is.EqualTo(typeof(DateTime)));
    }
}
