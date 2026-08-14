using Nivara.Exceptions;
using Nivara.IO;
using Nivara.Linq;
using Nivara.Operations;
using NUnit.Framework;

namespace Nivara.Tests.Query;

[TestFixture]
public class NivaraQueryFeatureTests
{
    sealed class Person
    {
        public string Name { get; set; } = string.Empty;
        public string City { get; set; } = string.Empty;
        public int Age { get; set; }
        public double Salary { get; set; }
    }

    sealed class NullablePerson
    {
        public string Name { get; set; } = string.Empty;
        public int? Age { get; set; }
    }

    sealed class NameCityRow
    {
        public string Name { get; set; } = string.Empty;
        public string City { get; set; } = string.Empty;
    }

    sealed class NameAgeRow
    {
        public string Name { get; set; } = string.Empty;
        public int Age { get; set; }
    }

    static NivaraFrame CreatePeopleFrame()
    {
        var names = NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Carol", "Dan", "Eve" });
        var cities = NivaraColumn<string>.CreateForReferenceType(new[] { "NYC", "LA", "NYC", "LA", "NYC" });
        var ages = NivaraColumn<int>.Create(new[] { 25, 40, 35, 20, 50 });
        var salaries = NivaraColumn<double>.Create(new[] { 80000.0, 120000.0, 95000.0, 50000.0, 110000.0 });

        return NivaraFrame.Create(
            ("Name", names),
            ("City", cities),
            ("Age", ages),
            ("Salary", salaries)
        );
    }

    static NivaraFrame CreateNullableFrame()
    {
        var names = NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c", "d" });
        var ages = NivaraColumn.CreateFromNullable(new int?[] { 30, null, 20, null });
        return NivaraFrame.Create(("Name", names), ("Age", ages));
    }

    // ── Distinct ──

    [Test]
    public void Distinct_DedupsAllColumns_KeepsFirstOccurrence()
    {
        var names = NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "a", "c", "b" });
        var cities = NivaraColumn<string>.CreateForReferenceType(new[] { "NYC", "LA", "NYC", "LA", "LA" });
        using var frame = NivaraFrame.Create(("Name", names), ("City", cities));

        var rows = frame.Query<NameCityRow>().Distinct().ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "a", "b", "c" }));
    }

    [Test]
    public void DistinctBy_KeyColumn_DedupsOnKey()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>().DistinctBy(p => p.City).ToObjects();

        Assert.That(rows.Select(p => p.City), Is.EqualTo(new[] { "NYC", "LA" }));
        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "Alice", "Bob" }));
    }

    [Test]
    public void DistinctBy_ComputedKey_ThrowsUnsupported()
    {
        using var frame = CreatePeopleFrame();

        Assert.Throws<UnsupportedQueryExpressionException>(() => frame.Query<Person>().DistinctBy(p => p.Age * 2));
    }

    // ── SelectRows ──

    [Test]
    public void SelectRows_PicksAndReordersRows()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>().SelectRows(4, 0, 2).ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "Eve", "Alice", "Carol" }));
    }

    [Test]
    public void SelectRows_EmptyIndices_ThrowsArgumentException()
    {
        using var frame = CreatePeopleFrame();

        Assert.Throws<ArgumentException>(() => frame.Query<Person>().SelectRows());
    }

    // ── Typed multi-key sort ──

    [Test]
    public void OrderBy_DirectionDescending_SortsDescending()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>()
            .OrderBy(p => p.Age, direction: SortDirection.Descending)
            .ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "Eve", "Bob", "Carol", "Alice", "Dan" }));
    }

    [Test]
    public void OrderBy_NullsFirst_PlacesNullKeysFirst()
    {
        using var frame = CreateNullableFrame();

        var rows = frame.Query<NullablePerson>()
            .OrderBy(p => p.Age, nullOrdering: NullOrdering.NullsFirst)
            .ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "b", "d", "c", "a" }));
    }

    [Test]
    public void OrderBy_NullsLast_PlacesNullKeysLast()
    {
        using var frame = CreateNullableFrame();

        var rows = frame.Query<NullablePerson>()
            .OrderBy(p => p.Age, nullOrdering: NullOrdering.NullsLast)
            .ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "c", "a", "b", "d" }));
    }

    [Test]
    public void OrderBy_ThenBy_ComposeLexicographically()
    {
        var names = NivaraColumn<string>.CreateForReferenceType(new[] { "b", "a", "b", "a" });
        var ages = NivaraColumn<int>.Create(new[] { 30, 20, 10, 40 });
        using var frame = NivaraFrame.Create(("Name", names), ("Age", ages));

        var rows = frame.Query<NameAgeRow>()
            .OrderBy(p => p.Name)
            .ThenBy(p => p.Age)
            .ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "a", "a", "b", "b" }));
        Assert.That(rows.Select(p => p.Age), Is.EqualTo(new[] { 20, 40, 10, 30 }));
    }

    [Test]
    public void OrderBy_ThenByDescending_SortsWithinPrimaryGroups()
    {
        var names = NivaraColumn<string>.CreateForReferenceType(new[] { "b", "a", "b", "a" });
        var ages = NivaraColumn<int>.Create(new[] { 30, 20, 10, 40 });
        using var frame = NivaraFrame.Create(("Name", names), ("Age", ages));

        var rows = frame.Query<NameAgeRow>()
            .OrderBy(p => p.Name)
            .ThenBy(p => p.Age, direction: SortDirection.Descending)
            .ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "a", "a", "b", "b" }));
        Assert.That(rows.Select(p => p.Age), Is.EqualTo(new[] { 40, 20, 30, 10 }));
    }

    [Test]
    public void OrderBy_ComputedKey_MaterializesSortColumn()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>().OrderBy(p => p.Age * 2).ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "Dan", "Alice", "Carol", "Bob", "Eve" }));
    }

    // ── Lazy file-source queries ──

    [Test]
    public void CsvScanAsQuery_CollectAndToObjects_ProduceSameRows()
    {
        using var testFiles = new TestFileScope();

        var query = testFiles.CsvQuery;
        Assert.That(query.IsLazy, Is.True);

        var frame = query.Collect();
        Assert.That(frame.RowCount, Is.EqualTo(3));
        Assert.That(frame.GetColumn<int>("Salary")[0], Is.EqualTo(75000));

        var rows = testFiles.CsvQuery.ToObjects();
        Assert.That(rows.Select(r => r.Name), Is.EqualTo(new[] { "Alice", "Bob", "Charlie" }));
    }

    [Test]
    public void CsvScanCsvAsQuery_WhereFilters_MatchesCollect()
    {
        using var testFiles = new TestFileScope();

        var rows = testFiles.CsvQuery.Where(p => p.Age > 25).ToObjects();

        Assert.That(rows.Select(r => r.Name), Is.EqualTo(new[] { "Alice", "Charlie" }));
    }

    [Test]
    public void JsonScanAsQuery_CollectAndToObjects_ProduceSameRows()
    {
        using var testFiles = new TestFileScope();

        var query = testFiles.JsonQuery;
        Assert.That(query.IsLazy, Is.True);

        var frame = query.Collect();
        Assert.That(frame.RowCount, Is.EqualTo(3));
        Assert.That(frame.GetColumn<double>("Salary")[1], Is.EqualTo(65000));

        var rows = testFiles.JsonQuery.ToObjects();
        Assert.That(rows.Select(r => r.Name), Is.EqualTo(new[] { "Alice", "Bob", "Charlie" }));
    }

    [Test]
    public void JsonScanJsonAsQuery_WhereFilters_MatchesCollect()
    {
        using var testFiles = new TestFileScope();

        var rows = testFiles.JsonQuery.Where(p => p.Salary > 70000).ToObjects();

        Assert.That(rows.Select(r => r.Name), Is.EqualTo(new[] { "Alice", "Charlie" }));
    }

    [Test]
    public void ScanAsQuery_MissingFile_ThrowsFileNotFound()
    {
        var missingPath = Path.Combine(Path.GetTempPath(), "nivara_missing_" + Guid.NewGuid() + ".csv");

        Assert.Throws<FileNotFoundException>(() => Csv.ScanQuery<CsvPerson>(missingPath));
    }

    sealed class CsvPerson
    {
        public string Name { get; set; } = string.Empty;
        public int Age { get; set; }
        public int Salary { get; set; }
    }

    sealed class JsonPerson
    {
        public string Name { get; set; } = string.Empty;
        public double Age { get; set; }
        public double Salary { get; set; }
    }

    sealed class TestFileScope : IDisposable
    {
        readonly string directory;

        public TestFileScope()
        {
            directory = Path.Combine(Path.GetTempPath(), "NivaraTypedQueryTests", Guid.NewGuid().ToString());
            Directory.CreateDirectory(directory);

            var csvPath = Path.Combine(directory, "people.csv");
            File.WriteAllText(csvPath, """
                Name,Age,Salary
                Alice,30,75000
                Bob,25,65000
                Charlie,35,85000
                """);

            var jsonPath = Path.Combine(directory, "people.json");
            File.WriteAllText(jsonPath, """
                [
                  {"Name": "Alice", "Age": 30, "Salary": 75000},
                  {"Name": "Bob", "Age": 25, "Salary": 65000},
                  {"Name": "Charlie", "Age": 35, "Salary": 85000}
                ]
                """);

            CsvQuery = Csv.ScanQuery<CsvPerson>(csvPath);
            JsonQuery = Json.ScanQuery<JsonPerson>(jsonPath);
        }

        public NivaraQuery<CsvPerson> CsvQuery { get; }
        public NivaraQuery<JsonPerson> JsonQuery { get; }

        public void Dispose()
        {
            if (Directory.Exists(directory))
                Directory.Delete(directory, true);
        }
    }
}
