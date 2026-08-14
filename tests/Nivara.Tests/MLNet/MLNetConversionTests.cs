using Microsoft.ML;
using Nivara.MLNet;
using NUnit.Framework;

namespace Nivara.Tests.MLNet;

[TestFixture]
public class MLNetConversionTests
{
    class PrimitiveRow
    {
        public bool BoolValue { get; set; }
        public byte ByteValue { get; set; }
        public sbyte SByteValue { get; set; }
        public short ShortValue { get; set; }
        public ushort UShortValue { get; set; }
        public int IntValue { get; set; }
        public uint UIntValue { get; set; }
        public long LongValue { get; set; }
        public ulong ULongValue { get; set; }
        public float FloatValue { get; set; }
        public double DoubleValue { get; set; }
        public string TextValue { get; set; } = "";
        public DateTime DateValue { get; set; }
        public DateTimeOffset OffsetValue { get; set; }
    }

    class SimpleIntRow
    {
        public int A { get; set; }
        public int B { get; set; }
    }

    class StringRow
    {
        public int Id { get; set; }
        public string Text { get; set; } = "";
    }

    class VectorRow
    {
        public float[] Features { get; set; } = [];
    }

    class DoubleVectorRow
    {
        public double[] Features { get; set; } = [];
    }

    MLContext mlContext;

    [SetUp]
    public void Setup()
    {
        mlContext = new MLContext(seed: 42);
    }

    [Test]
    public void ToNivaraFrame_ReadsAllPrimitiveDataViewTypes_Faithfully()
    {
        var data = Enumerable.Range(0, 3).Select(i => new PrimitiveRow
        {
            BoolValue = i % 2 == 0,
            ByteValue = (byte)(i + 1),
            SByteValue = (sbyte)(i - 1),
            ShortValue = (short)(i + 10),
            UShortValue = (ushort)(i + 20),
            IntValue = i + 100,
            UIntValue = (uint)(i + 200),
            LongValue = i + 1000L,
            ULongValue = (ulong)i + 2000UL,
            FloatValue = i + 1.5f,
            DoubleValue = i + 2.5,
            TextValue = $"row{i}",
            DateValue = new DateTime(2024, 1, 1).AddDays(i),
            OffsetValue = new DateTimeOffset(2024, 1, 1, 12, 0, 0, TimeSpan.Zero).AddDays(i)
        }).ToList();

        var dataView = mlContext.Data.LoadFromEnumerable(data);
        var frame = MLNetInterop.ToNivaraFrame(dataView, mlContext);

        Assert.That(frame.ColumnCount, Is.EqualTo(14));
        Assert.That(frame.GetColumn("BoolValue").ElementType, Is.EqualTo(typeof(bool)));
        Assert.That(frame.GetColumn("ByteValue").ElementType, Is.EqualTo(typeof(byte)));
        Assert.That(frame.GetColumn("SByteValue").ElementType, Is.EqualTo(typeof(sbyte)));
        Assert.That(frame.GetColumn("ShortValue").ElementType, Is.EqualTo(typeof(short)));
        Assert.That(frame.GetColumn("UShortValue").ElementType, Is.EqualTo(typeof(ushort)));
        Assert.That(frame.GetColumn("IntValue").ElementType, Is.EqualTo(typeof(int)));
        Assert.That(frame.GetColumn("UIntValue").ElementType, Is.EqualTo(typeof(uint)));
        Assert.That(frame.GetColumn("LongValue").ElementType, Is.EqualTo(typeof(long)));
        Assert.That(frame.GetColumn("ULongValue").ElementType, Is.EqualTo(typeof(ulong)));
        Assert.That(frame.GetColumn("FloatValue").ElementType, Is.EqualTo(typeof(float)));
        Assert.That(frame.GetColumn("DoubleValue").ElementType, Is.EqualTo(typeof(double)));
        Assert.That(frame.GetColumn("TextValue").ElementType, Is.EqualTo(typeof(string)));
        Assert.That(frame.GetColumn("DateValue").ElementType, Is.EqualTo(typeof(DateTime)));
        Assert.That(frame.GetColumn("OffsetValue").ElementType, Is.EqualTo(typeof(DateTimeOffset)));

        Assert.That(frame.GetColumn<int>("IntValue")[2], Is.EqualTo(102));
        Assert.That(frame.GetColumn<uint>("UIntValue")[2], Is.EqualTo(202u));
        Assert.That(frame.GetColumn<long>("LongValue")[1], Is.EqualTo(1001L));
        Assert.That(frame.GetColumn<ulong>("ULongValue")[1], Is.EqualTo(2001UL));
        Assert.That(frame.GetColumn<sbyte>("SByteValue")[0], Is.EqualTo((sbyte)-1));
        Assert.That(frame.GetColumn<ushort>("UShortValue")[0], Is.EqualTo((ushort)20));
        Assert.That(frame.GetColumn<bool>("BoolValue")[0], Is.True);
        Assert.That(frame.GetColumn<string>("TextValue")[1], Is.EqualTo("row1"));
        Assert.That(frame.GetColumn<DateTime>("DateValue")[1], Is.EqualTo(new DateTime(2024, 1, 2)));
        Assert.That(frame.GetColumn<DateTimeOffset>("OffsetValue")[1].UtcTicks,
            Is.EqualTo(new DateTimeOffset(2024, 1, 2, 12, 0, 0, TimeSpan.Zero).UtcTicks));
    }

    [Test]
    public void ToNivaraFrame_Int32Column_IsPreservedNotDropped()
    {
        var data = Enumerable.Range(0, 5).Select(i => new SimpleIntRow { A = i, B = i * 10 }).ToList();
        var dataView = mlContext.Data.LoadFromEnumerable(data);

        var frame = MLNetInterop.ToNivaraFrame(dataView, mlContext);

        Assert.That(frame.ColumnNames, Contains.Item("A"));
        Assert.That(frame.ColumnNames, Contains.Item("B"));
        Assert.That(frame.GetColumn("A").ElementType, Is.EqualTo(typeof(int)));
        Assert.That(frame.GetColumn<int>("A")[4], Is.EqualTo(4));
        Assert.That(frame.GetColumn<int>("B")[3], Is.EqualTo(30));
    }

    [Test]
    public void ToNivaraFrame_KeyColumn_ReadsAsUInt()
    {
        var data = new[] { "cat", "dog", "cat", "bird", "dog" }
            .Select((s, i) => new StringRow { Text = s, Id = i })
            .ToList();
        var dataView = mlContext.Data.LoadFromEnumerable(data);
        var keyed = mlContext.Transforms.Conversion.MapValueToKey("Key", "Text").Fit(dataView).Transform(dataView);

        var frame = MLNetInterop.ToNivaraFrame(keyed, mlContext);

        Assert.That(frame.ColumnNames, Contains.Item("Key"));
        Assert.That(frame.GetColumn("Key").ElementType, Is.EqualTo(typeof(uint)));
        var keyColumn = frame.GetColumn<uint>("Key");
        Assert.That(keyColumn[0], Is.EqualTo(keyColumn[2]), "identical strings map to the same key");
        Assert.That(keyColumn[0], Is.Not.EqualTo(keyColumn[1]));
    }

    [Test]
    public void ToNivaraFrame_VectorSingleColumn_ExtractsFirstElement()
    {
        var data = new[]
        {
            new VectorRow { Features = new float[] { 1f, 2f, 3f } },
            new VectorRow { Features = new float[] { 4f, 5f, 6f } }
        };
        var dataView = mlContext.Data.LoadFromEnumerable(data);

        var frame = MLNetInterop.ToNivaraFrame(dataView, mlContext);

        Assert.That(frame.GetColumn("Features").ElementType, Is.EqualTo(typeof(float)));
        Assert.That(frame.GetColumn<float>("Features")[0], Is.EqualTo(1f));
        Assert.That(frame.GetColumn<float>("Features")[1], Is.EqualTo(4f));
    }

    [Test]
    public void ToNivaraFrame_VectorDoubleColumn_ThrowsNotSupported()
    {
        var data = new[] { new DoubleVectorRow { Features = new double[] { 1.0, 2.0 } } };
        var dataView = mlContext.Data.LoadFromEnumerable(data);

        var ex = Assert.Throws<NotSupportedException>(() => MLNetInterop.ToNivaraFrame(dataView, mlContext));
        Assert.That(ex!.Message, Does.Contain("Features"));
    }

    [Test]
    public void ToDataView_UnsupportedColumnType_Throws()
    {
        var frame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.Create(new[] { "a", "b" })),
            ("Value", NivaraColumn<float>.Create(new[] { 1f, 2f }))
        );

        // LoadFromEnumerable is lazy; the ConvertToFloat error surfaces only on materialization
        var dataView = frame.ToDataView(mlContext);
        Assert.Throws<InvalidOperationException>(
            () => mlContext.Data.CreateEnumerable<TwoColumnData>(dataView, reuseRowObject: false).ToList());
    }

    [Test]
    public void ToFeatureVectors_ExtendedNumericTypes_ConvertToFloat()
    {
        var frame = NivaraFrame.Create(
            ("UIntValue", NivaraColumn<uint>.Create(new[] { 1u, 2u })),
            ("UShortValue", NivaraColumn<ushort>.Create(new[] { (ushort)3, (ushort)4 })),
            ("SByteValue", NivaraColumn<sbyte>.Create(new[] { (sbyte)(-1), (sbyte)5 })),
            ("HalfValue", NivaraColumn<Half>.Create(new[] { (Half)1.5f, (Half)2.5f }))
        );

        var vectors = frame.ToFeatureVectors("UIntValue", "UShortValue", "SByteValue", "HalfValue");

        Assert.That(vectors[0].Length, Is.EqualTo(4));
        Assert.That(vectors[0].GetItemOrDefault(0), Is.EqualTo(1f));
        Assert.That(vectors[0].GetItemOrDefault(1), Is.EqualTo(3f));
        Assert.That(vectors[0].GetItemOrDefault(2), Is.EqualTo(-1f));
        Assert.That(vectors[0].GetItemOrDefault(3), Is.EqualTo(1.5f));
        Assert.That(vectors[1].GetItemOrDefault(3), Is.EqualTo(2.5f));
    }

    [Test]
    public void ToDataView_NullNumericValue_BecomesZero()
    {
        var column = NivaraColumn.CreateFromNullable(new float?[] { 1.5f, null, 3.5f });
        var frame = NivaraFrame.Create(("Value", column));

        var dataView = frame.ToDataView(mlContext);
        var roundTrip = MLNetInterop.ToNivaraFrame(dataView, mlContext);

        // A 1-column frame maps to ML.NET's GenericData.Features vector contract
        Assert.That(roundTrip.ColumnNames, Is.EqualTo(new[] { "Features" }));
        Assert.That(roundTrip.GetColumn<float>("Features")[0], Is.EqualTo(1.5f));
        Assert.That(roundTrip.GetColumn<float>("Features")[1], Is.EqualTo(0f));
        Assert.That(roundTrip.GetColumn<float>("Features")[2], Is.EqualTo(3.5f));
    }
}
