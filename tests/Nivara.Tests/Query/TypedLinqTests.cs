using Nivara.Exceptions;
using Nivara.Linq;
using NUnit.Framework;

namespace Nivara.Tests.Query;

[TestFixture]
public class TypedLinqTests
{
    sealed class Person
    {
        public string Name { get; set; } = string.Empty;
        public string City { get; set; } = string.Empty;
        public int Age { get; set; }
        public double Salary { get; set; }
        public bool IsActive { get; set; }
    }

    static NivaraFrame CreatePeopleFrame()
    {
        var names = NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Carol", "Dan", "Eve" });
        var cities = NivaraColumn<string>.CreateForReferenceType(new[] { "NYC", "LA", "NYC", "LA", "NYC" });
        var ages = NivaraColumn<int>.Create(new[] { 25, 40, 35, 20, 50 });
        var salaries = NivaraColumn<double>.Create(new[] { 80000.0, 120000.0, 95000.0, 50000.0, 110000.0 });
        var active = NivaraColumn<bool>.Create(new[] { true, true, false, false, true });

        return NivaraFrame.Create(
            ("Name", names),
            ("City", cities),
            ("Age", ages),
            ("Salary", salaries),
            ("IsActive", active)
        );
    }

    static NivaraFrame CreateNullableFrame()
    {
        var values = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3 });
        var labels = NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" });
        return NivaraFrame.Create(("Value", values), ("Label", labels));
    }

    // ── Query<T>() entry ──

    [Test]
    public void Query_ToObjects_MaterializesAllRows()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>().ToObjects();

        Assert.That(rows.Count, Is.EqualTo(5));
        Assert.That(rows[0].Name, Is.EqualTo("Alice"));
        Assert.That(rows[0].City, Is.EqualTo("NYC"));
        Assert.That(rows[0].Age, Is.EqualTo(25));
        Assert.That(rows[2].Salary, Is.EqualTo(95000.0));
        Assert.That(rows[4].IsActive, Is.True);
    }

    [Test]
    public void Query_RowTypeNotMapping_SchemaValidationException()
    {
        using var frame = CreatePeopleFrame();

        Assert.Throws<SchemaValidationException>(() => frame.Query<UnmappedRow>());
    }

    sealed class UnmappedRow
    {
        public string Name { get; set; } = string.Empty;
        public string Missing { get; set; } = string.Empty;
    }

    [Test]
    public void Query_RowTypeWithNoProperties_SchemaValidationException()
    {
        using var frame = CreatePeopleFrame();

        Assert.Throws<SchemaValidationException>(() => frame.Query<EmptyRow>());
    }

    sealed class EmptyRow
    { }

    // ── Where ──

    [Test]
    public void Where_Comparison_FiltersRows()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>().Where(p => p.Age > 30).ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "Bob", "Carol", "Eve" }));
    }

    [Test]
    public void Where_StringEquality_FiltersRows()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>().Where(p => p.City == "LA").ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "Bob", "Dan" }));
    }

    [Test]
    public void Where_AndOrNot_ComposeCorrectly()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>()
            .Where(p => p.City == "NYC" && p.Age >= 30)
            .Where(p => !p.IsActive)
            .ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "Carol" }));
    }

    [Test]
    public void Where_Arithmetic_ComparesComputedValues()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>().Where(p => p.Age * 2 > 70).ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "Bob", "Eve" }));
    }

    [Test]
    public void Where_NumericLiteral_ImplicitCoercionWorks()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>().Where(p => p.Salary >= 100000).ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "Bob", "Eve" }));
    }

    [Test]
    public void Where_MethodCall_FailsFast()
    {
        using var frame = CreatePeopleFrame();

        Assert.Throws<UnsupportedQueryExpressionException>(() =>
            frame.Query<Person>().Where(p => p.Name.ToUpper() == "ALICE"));
    }

    [Test]
    public void Where_ClosureCapture_FailsFast()
    {
        using var frame = CreatePeopleFrame();
        var threshold = 30;

        Assert.Throws<UnsupportedQueryExpressionException>(() =>
            frame.Query<Person>().Where(p => p.Age > threshold));
    }

    [Test]
    public void Where_Coalesce_FailsFast()
    {
        using var frame = CreateNullableFrame();

        Assert.Throws<UnsupportedQueryExpressionException>(() =>
            frame.Query<NullableRow>().Where(p => (p.Value ?? 0) > 10));
    }

    // ── Select ──

    [Test]
    public void Select_AnonymousProjection_ProducesColumnsAndRows()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>()
            .Select(p => new { p.Name, p.City })
            .ToObjects();

        Assert.That(rows.Count, Is.EqualTo(5));
        Assert.That(rows[0].Name, Is.EqualTo("Alice"));
        Assert.That(rows[0].City, Is.EqualTo("NYC"));
    }

    [Test]
    public void Select_ComputedMember_ComputesCorrectly()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>()
            .Select(p => new { p.Name, DoubleAge = p.Age * 2 })
            .ToObjects();

        Assert.That(rows[0].Name, Is.EqualTo("Alice"));
        Assert.That(rows[0].DoubleAge, Is.EqualTo(50));
        Assert.That(rows[4].DoubleAge, Is.EqualTo(100));
    }

    [Test]
    public void Select_MemberInitProjection_ProducesRows()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>()
            .Select(p => new PersonRow { Name = p.Name, Age = p.Age })
            .ToObjects();

        Assert.That(rows[0], Is.EqualTo(new PersonRow { Name = "Alice", Age = 25 }));
    }

    sealed class PersonRow
    {
        public string Name { get; set; } = string.Empty;
        public int Age { get; set; }

        public override bool Equals(object? obj)
        {
            return obj is PersonRow other && other.Name == Name && other.Age == Age;
        }

        public override int GetHashCode() => HashCode.Combine(Name, Age);
    }

    [Test]
    public void Select_ScalarProjection_FailsFast()
    {
        using var frame = CreatePeopleFrame();

        Assert.Throws<UnsupportedQueryExpressionException>(() =>
            frame.Query<Person>().Select(p => p.Name));
    }

    [Test]
    public void Select_WhereChain_MaterializesFilteredProjection()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>()
            .Where(p => p.City == "NYC")
            .Select(p => new { p.Name, p.Age })
            .ToObjects();

        Assert.That(rows.Select(r => r.Name), Is.EqualTo(new[] { "Alice", "Carol", "Eve" }));
    }

    [Test]
    public void Select_Collect_ReturnsFrameWithProjectedColumns()
    {
        using var frame = CreatePeopleFrame();

        using var result = frame.Query<Person>()
            .Select(p => new { p.Name, p.Age })
            .Collect();

        Assert.That(result.ColumnNames, Is.EqualTo(new[] { "Name", "Age" }));
        Assert.That(result.GetColumn<string>("Name")[1], Is.EqualTo("Bob"));
        Assert.That(result.GetColumn<int>("Age")[1], Is.EqualTo(40));
    }

    // ── OrderBy / Skip / Take ──

    [Test]
    public void OrderBy_AndThenBy_SortsCorrectly()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>()
            .OrderByDescending(p => p.City)
            .ThenBy(p => p.Age)
            .ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "Alice", "Carol", "Eve", "Dan", "Bob" }));
    }

    [Test]
    public void SkipTake_SlicesRows()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>()
            .OrderBy(p => p.Age)
            .Skip(1)
            .Take(2)
            .ToObjects();

        Assert.That(rows.Select(p => p.Name), Is.EqualTo(new[] { "Alice", "Carol" }));
    }

    // ── GroupBy ──

    [Test]
    public void GroupBy_SelectAggregates_ComputesPerGroup()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>()
            .GroupBy(p => p.City)
            .Select(g => new { g.Key, Count = g.Count(), AvgAge = g.Average(p => p.Age) })
            .ToObjects();

        Assert.That(rows.Count, Is.EqualTo(2));
        Assert.That(rows.Single(r => r.Key == "NYC").Count, Is.EqualTo(3));
        Assert.That(rows.Single(r => r.Key == "LA").Count, Is.EqualTo(2));
        Assert.That(rows.Single(r => r.Key == "NYC").AvgAge, Is.EqualTo((25 + 35 + 50) / 3.0).Within(1e-9));
    }

    [Test]
    public void GroupBy_SelectSumAndMax_ComputesCorrectly()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>()
            .GroupBy(p => p.City)
            .Select(g => new { g.Key, Total = g.Sum(p => p.Age), Oldest = g.Max(p => p.Age) })
            .ToObjects();

        var nyc = rows.Single(r => r.Key == "NYC");
        Assert.That(nyc.Total, Is.EqualTo(25L + 35L + 50L));
        Assert.That(nyc.Oldest, Is.EqualTo(50));
    }

    [Test]
    public void GroupBy_Collect_ReturnsDistinctKeys()
    {
        using var frame = CreatePeopleFrame();

        using var result = frame.Query<Person>().GroupBy(p => p.City).Collect();

        Assert.That(result.RowCount, Is.EqualTo(2));
        var cities = result.GetColumn<string>("City").ToArray();
        Assert.That(cities, Is.EquivalentTo(new[] { "NYC", "LA" }));
    }

    [Test]
    public void GroupBy_RenamedKey_ProjectsKeyToMemberName()
    {
        using var frame = CreatePeopleFrame();

        var rows = frame.Query<Person>()
            .GroupBy(p => p.City)
            .Select(g => new { City = g.Key, Count = g.Count() })
            .ToObjects();

        Assert.That(rows.Single(r => r.City == "LA").Count, Is.EqualTo(2));
    }

    [Test]
    public void GroupBy_NonAggregateMember_FailsFast()
    {
        using var frame = CreatePeopleFrame();

        Assert.Throws<UnsupportedQueryExpressionException>(() =>
            frame.Query<Person>().GroupBy(p => p.City).Select(g => new CityGroup { Key = g.Key, Extra = 5 }));
    }

    sealed class CityGroup
    {
        public string Key { get; set; } = string.Empty;
        public int Extra { get; set; }
    }

    // ── Nullability ──

    [Test]
    public void ToObjects_NullableColumnToNullableProperty_MaterializesNull()
    {
        using var frame = CreateNullableFrame();

        var rows = frame.Query<NullableRow>().ToObjects();

        Assert.That(rows[0].Value, Is.EqualTo(1));
        Assert.That(rows[1].Value, Is.Null);
        Assert.That(rows[2].Value, Is.EqualTo(3));
    }

    sealed class NullableRow
    {
        public int? Value { get; set; }
        public string Label { get; set; } = string.Empty;
    }

    [Test]
    public void ToObjects_NullIntoNonNullableValueType_Throws()
    {
        using var frame = CreateNullableFrame();

        Assert.Throws<SchemaValidationException>(() => frame.Query<NonNullableRow>().ToObjects());
    }

    sealed class NonNullableRow
    {
        public int Value { get; set; }
    }

    // ── Schema exposure ──

    [Test]
    public void Schema_ReflectsProjectionColumns()
    {
        using var frame = CreatePeopleFrame();

        var query = frame.Query<Person>().Select(p => new { p.Name, p.Age });

        Assert.That(query.Schema.ColumnNames, Is.EqualTo(new[] { "Name", "Age" }));
        Assert.That(query.Schema.GetColumnType("Name"), Is.EqualTo(typeof(string)));
        Assert.That(query.Schema.GetColumnType("Age"), Is.EqualTo(typeof(int)));
    }

    [Test]
    public void ExplainPlan_ContainsOperations()
    {
        using var frame = CreatePeopleFrame();

        var plan = frame.Query<Person>().Where(p => p.Age > 30).ExplainPlan();

        Assert.That(plan, Does.Contain("Filter"));
    }
}
