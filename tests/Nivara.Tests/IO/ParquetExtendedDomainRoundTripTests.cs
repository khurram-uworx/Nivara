using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class ParquetExtendedDomainRoundTripTests
{
    [Test]
    public void RoundTrip_DateOnlyColumn_PreservesValuesAndNulls()
    {
        var values = new DateOnly?[] { new(2023, 1, 15), null, new(2024, 12, 31) };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("DateColumn", column));

        var result = WriteThenRead(frame);

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

        var result = WriteThenRead(frame);

        var resultColumn = result.GetColumn<Guid>("GuidColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
                Assert.That(resultColumn[i], Is.EqualTo(values[i]));
        }
    }

    [Test]
    public void RoundTrip_HalfColumn_RestoresOriginalTypeViaMetadata()
    {
        var values = new Half?[] { (Half)1.5f, null, (Half)2.25f, (Half)(-3.5f) };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("HalfColumn", column));

        var result = WriteThenRead(frame);

        var resultColumn = result.GetColumn<Half>("HalfColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
                Assert.That(resultColumn[i], Is.EqualTo(values[i]));
        }
    }

    [Test]
    public void RoundTrip_TimeOnlyColumn_RestoresOriginalTypeViaMetadata()
    {
        var values = new TimeOnly?[] { new(14, 30, 45, 123), null, new(0, 0, 0) };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("TimeColumn", column));

        var result = WriteThenRead(frame);

        var resultColumn = result.GetColumn<TimeOnly>("TimeColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
                Assert.That(resultColumn[i], Is.EqualTo(values[i]));
        }
    }

    [Test]
    public void RoundTrip_TimeSpanColumn_RestoresOriginalTypeViaMetadata()
    {
        var values = new TimeSpan?[] { TimeSpan.FromTicks(123456789L), null, TimeSpan.Zero, TimeSpan.FromDays(2) };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("SpanColumn", column));

        var result = WriteThenRead(frame);

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

        var result = WriteThenRead(frame);

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

        var result = WriteThenRead(frame);

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

        var result = WriteThenRead(frame);

        var resultColumn = result.GetColumn<char>("CharColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
                Assert.That(resultColumn[i], Is.EqualTo(values[i]));
        }
    }

    [Test]
    public void RoundTrip_DateTimeOffsetColumn_RestoresOriginalTypeViaMetadata()
    {
        var values = new DateTimeOffset?[]
        {
            new DateTimeOffset(2023, 6, 15, 10, 30, 0, TimeSpan.FromHours(5)),
            null,
            new DateTimeOffset(2024, 1, 1, 0, 0, 0, TimeSpan.Zero)
        };
        var column = NivaraColumn.CreateFromNullable(values);
        var frame = NivaraFrame.Create(("OffsetColumn", column));

        var result = WriteThenRead(frame);

        var resultColumn = result.GetColumn<DateTimeOffset>("OffsetColumn");
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(resultColumn.IsNull(i), Is.EqualTo(values[i] == null));
            if (values[i] is not null)
                Assert.That(resultColumn[i], Is.EqualTo(values[i]));
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
            ("OffsetColumn", NivaraColumn<DateTimeOffset>.Create(new[] { DateTimeOffset.UtcNow, DateTimeOffset.UtcNow }))
        );

        var result = WriteThenRead(frame);

        Assert.That(result.GetColumn<Half>("HalfColumn")[0], Is.EqualTo((Half)1.5f));
        Assert.That(result.GetColumn<Guid>("GuidColumn")[0], Is.EqualTo(frame.GetColumn<Guid>("GuidColumn")[0]));
        Assert.That(result.GetColumn<DateOnly>("DateColumn")[0], Is.EqualTo(new DateOnly(2023, 1, 1)));
        Assert.That(result.GetColumn<TimeOnly>("TimeColumn")[0], Is.EqualTo(new TimeOnly(1, 2, 3)));
        Assert.That(result.GetColumn<TimeSpan>("SpanColumn")[0], Is.EqualTo(TimeSpan.FromSeconds(1)));
        Assert.That(result.GetColumn<nint>("NIntColumn")[0], Is.EqualTo((nint)1));
        Assert.That(result.GetColumn<nuint>("NUIntColumn")[0], Is.EqualTo((nuint)1));
        Assert.That(result.GetColumn<char>("CharColumn")[0], Is.EqualTo('a'));
        Assert.That(result.GetColumn<DateTimeOffset>("OffsetColumn")[0].UtcDateTime,
            Is.EqualTo(frame.GetColumn<DateTimeOffset>("OffsetColumn")[0].UtcDateTime));
    }

    [Test]
    public async Task RoundTrip_AsyncWriteRead_ExtendedDomain_PreservesTypes()
    {
        var frame = NivaraFrame.Create(
            ("HalfColumn", NivaraColumn<Half>.Create(new[] { (Half)1.5f, (Half)2.5f })),
            ("GuidColumn", NivaraColumn<Guid>.Create(new[] { Guid.NewGuid(), Guid.NewGuid() })),
            ("TimeColumn", NivaraColumn<TimeOnly>.Create(new[] { new TimeOnly(1, 2, 3), new TimeOnly(4, 5, 6) })),
            ("SpanColumn", NivaraColumn<TimeSpan>.Create(new[] { TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(2) }))
        );
        var tempFile = Path.GetTempFileName();

        try
        {
            await NivaraParquetWriter.WriteParquetAsync(frame, tempFile);
            var result = await NivaraParquetReader.ReadParquetAsync(tempFile);

            Assert.That(result.GetColumn<Half>("HalfColumn")[0], Is.EqualTo((Half)1.5f));
            Assert.That(result.GetColumn<Guid>("GuidColumn")[0], Is.EqualTo(frame.GetColumn<Guid>("GuidColumn")[0]));
            Assert.That(result.GetColumn<TimeOnly>("TimeColumn")[1], Is.EqualTo(new TimeOnly(4, 5, 6)));
            Assert.That(result.GetColumn<TimeSpan>("SpanColumn")[1], Is.EqualTo(TimeSpan.FromSeconds(2)));
        }
        finally
        {
            frame.Dispose();
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void RoundTrip_BatchWrite_ExtendedDomain_ConcatenatesAndRestores()
    {
        var frame1 = NivaraFrame.Create(
            ("HalfColumn", NivaraColumn<Half>.Create(new[] { (Half)1.5f, (Half)2.5f })),
            ("GuidColumn", NivaraColumn<Guid>.Create(new[] { Guid.NewGuid(), Guid.NewGuid() }))
        );
        var frame2 = NivaraFrame.Create(
            ("HalfColumn", NivaraColumn<Half>.Create(new[] { (Half)3.5f })),
            ("GuidColumn", NivaraColumn<Guid>.Create(new[] { Guid.NewGuid() }))
        );
        var tempFile = Path.GetTempFileName();

        try
        {
            NivaraParquetWriter.WriteParquetBatch(new[] { frame1, frame2 }, tempFile);
            var result = NivaraParquetReader.ReadParquet(tempFile);

            Assert.That(result.RowCount, Is.EqualTo(3));
            Assert.That(result.GetColumn<Half>("HalfColumn")[2], Is.EqualTo((Half)3.5f));
            Assert.That(result.GetColumn<Guid>("GuidColumn")[0], Is.EqualTo(frame1.GetColumn<Guid>("GuidColumn")[0]));
        }
        finally
        {
            frame1.Dispose();
            frame2.Dispose();
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    private static NivaraFrame WriteThenRead(NivaraFrame frame)
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            NivaraParquetWriter.WriteParquet(frame, tempFile);
            return NivaraParquetReader.ReadParquet(tempFile);
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }
}
