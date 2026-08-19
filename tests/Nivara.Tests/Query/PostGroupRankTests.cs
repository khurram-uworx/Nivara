using Nivara.Linq;
using Nivara.Operations;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Tests for post-aggregation ranking via window functions on NivaraQuery{T}.
/// Validates that DenseRank/Rank/PercentRank/RowNumber compose correctly
/// after GroupBy+Select in the LINQ pipeline.
/// </summary>
[TestFixture]
public class PostGroupRankTests
{
    // ── DenseRank over raw frame ──

    [Test]
    public void DenseRank_OverFrame_RanksCorrectly()
    {
        using var frame = CreateRegionFrame();

        using var result = frame.Query<Region>()
            .DenseRank("ErrorRank", [new SortKey("ErrorRate", SortDirection.Descending)])
            .Collect();

        var errorRank = result.GetColumn<long>("ErrorRank");
        var names = result.GetColumn<string>("Name").ToArray();
        var byName = names.Zip(errorRank.ToArray(), (n, r) => (Name: n, Rank: r))
            .ToDictionary(x => x.Name, x => x.Rank);

        Assert.That(byName["ap-south-1"], Is.EqualTo(1));
        Assert.That(byName["us-east-1"], Is.EqualTo(2));
        Assert.That(byName["us-west-2"], Is.EqualTo(3));
        Assert.That(byName["eu-west-1"], Is.EqualTo(4));
    }

    [Test]
    public void DenseRank_WithTies_ProducesNoGaps()
    {
        var keys = NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "C", "D" });
        var scores = NivaraColumn<double>.Create(new[] { 10.0, 20.0, 20.0, 30.0 });
        using var frame = NivaraFrame.Create(("Key", keys), ("Score", scores));

        using var result = frame.Query<TwoCol>()
            .DenseRank("Rank", [new SortKey("Score", SortDirection.Descending)])
            .Collect();

        var rank = result.GetColumn<long>("Rank");
        var keyArr = result.GetColumn<string>("Key").ToArray();
        var byKey = keyArr.Zip(rank.ToArray(), (k, r) => (Key: k, Rank: r))
            .ToDictionary(x => x.Key, x => x.Rank);

        Assert.That(byKey["D"], Is.EqualTo(1));
        Assert.That(byKey["B"], Is.EqualTo(2));
        Assert.That(byKey["C"], Is.EqualTo(2)); // tie — no gap
        Assert.That(byKey["A"], Is.EqualTo(3));
    }

    // ── DenseRank after GroupBy+Select ──

    [Test]
    public void DenseRank_AfterGroupBySelect_RanksAggregatedResults()
    {
        using var frame = CreatePeopleFrame();

        using var result = frame.Query<Person>()
            .GroupBy(p => p.City)
            .Select(g => new { City = g.Key, AvgAge = g.Average(p => p.Age) })
            .DenseRank("AgeRank", [new SortKey("AvgAge", SortDirection.Descending)])
            .Collect();

        var ageRank = result.GetColumn<long>("AgeRank");
        var cities = result.GetColumn<string>("City").ToArray();
        var avgAges = result.GetColumn<double>("AvgAge").ToArray();
        var byCity = cities
            .Zip(ageRank.ToArray().Zip(avgAges, (r, a) => (Rank: r, Avg: a)),
                (c, x) => (City: c, x.Rank, x.Avg))
            .ToDictionary(x => x.City, x => (x.Rank, x.Avg));

        // NYC avg: (25+35+50)/3 = 36.67, LA avg: (40+20)/2 = 30
        Assert.That(byCity["NYC"].Rank, Is.EqualTo(1));
        Assert.That(byCity["LA"].Rank, Is.EqualTo(2));
    }

    // ── Rank (gaps on ties) ──

    [Test]
    public void Rank_AfterGroupBySelect_RanksWithGapsOnTies()
    {
        using var frame = CreatePeopleFrame();

        using var result = frame.Query<Person>()
            .GroupBy(p => p.City)
            .Select(g => new { City = g.Key, Total = g.Count() })
            .Rank("CountRank", [new SortKey("Total", SortDirection.Descending)])
            .Collect();

        var countRank = result.GetColumn<long>("CountRank");
        var cities = result.GetColumn<string>("City").ToArray();
        var byCity = cities.Zip(countRank.ToArray(), (c, r) => (City: c, Rank: r))
            .ToDictionary(x => x.City, x => x.Rank);

        // NYC: 3 people, LA: 2 people
        Assert.That(byCity["NYC"], Is.EqualTo(1));
        Assert.That(byCity["LA"], Is.EqualTo(2));
    }

    // ── PercentRank ──

    [Test]
    public void PercentRank_OverFrame_ComputesCorrectly()
    {
        using var frame = CreateRegionFrame();

        using var result = frame.Query<Region>()
            .PercentRank("PctRank", [new SortKey("ErrorRate", SortDirection.Descending)])
            .Collect();

        var pctRank = result.GetColumn<double>("PctRank");
        var names = result.GetColumn<string>("Name").ToArray();
        var byName = names.Zip(pctRank.ToArray(), (n, r) => (Name: n, Pct: r))
            .ToDictionary(x => x.Name, x => x.Pct);

        // 4 rows: (rank-1) / (4-1)
        Assert.That(byName["ap-south-1"], Is.EqualTo(0.0).Within(1e-10));
        Assert.That(byName["eu-west-1"], Is.EqualTo(1.0).Within(1e-10));
    }

    // ── RowNumber ──

    [Test]
    public void RowNumber_OverFrame_AssignsSequentialNumbers()
    {
        using var frame = CreateRegionFrame();

        using var result = frame.Query<Region>()
            .RowNumber("RowNum", orderBy: [new SortKey("Name")])
            .Collect();

        var rowNum = result.GetColumn<long>("RowNum");
        var names = result.GetColumn<string>("Name").ToArray();
        var byName = names.Zip(rowNum.ToArray(), (n, r) => (Name: n, Row: r))
            .ToDictionary(x => x.Name, x => x.Row);

        Assert.That(byName["ap-south-1"], Is.EqualTo(1));
        Assert.That(byName["eu-west-1"], Is.EqualTo(2));
        Assert.That(byName["us-east-1"], Is.EqualTo(3));
        Assert.That(byName["us-west-2"], Is.EqualTo(4));
    }

    // ── WindowSpec overload ──

    [Test]
    public void DenseRank_WithWindowSpec_ComposesCorrectly()
    {
        using var frame = CreateRegionFrame();
        var spec = new WindowSpec().OrderBy("ErrorRate", SortDirection.Descending);

        using var result = frame.Query<Region>()
            .DenseRank("ErrorRank", spec)
            .Collect();

        var errorRank = result.GetColumn<long>("ErrorRank");
        var names = result.GetColumn<string>("Name").ToArray();
        var byName = names.Zip(errorRank.ToArray(), (n, r) => (Name: n, Rank: r))
            .ToDictionary(x => x.Name, x => x.Rank);

        Assert.That(byName["ap-south-1"], Is.EqualTo(1));
        Assert.That(byName["eu-west-1"], Is.EqualTo(4));
    }

    // ── Regression: pre-group window functions still work ──

    [Test]
    public void DenseRank_PreGroup_WindowFunctionStillWorks()
    {
        using var frame = CreatePeopleFrame();

        using var result = frame.Query<Person>()
            .DenseRank("AgeRank", [new SortKey("Age", SortDirection.Ascending)])
            .Collect();

        var ageRank = result.GetColumn<long>("AgeRank");
        var names = result.GetColumn<string>("Name").ToArray();
        var byName = names.Zip(ageRank.ToArray(), (n, r) => (Name: n, Rank: r))
            .ToDictionary(x => x.Name, x => x.Rank);

        Assert.That(byName["Dan"], Is.EqualTo(1));    // age 20
        Assert.That(byName["Alice"], Is.EqualTo(2));   // age 25
        Assert.That(byName["Carol"], Is.EqualTo(3));   // age 35
        Assert.That(byName["Bob"], Is.EqualTo(4));     // age 40
        Assert.That(byName["Eve"], Is.EqualTo(5));     // age 50
    }

    // ── Helpers ──

    sealed class Person
    {
        public string Name { get; set; } = string.Empty;
        public string City { get; set; } = string.Empty;
        public int Age { get; set; }
        public double Salary { get; set; }
    }

    sealed class Region
    {
        public string Name { get; set; } = string.Empty;
        public double ErrorRate { get; set; }
        public int RequestCount { get; set; }
    }

    sealed class TwoCol
    {
        public string Key { get; set; } = string.Empty;
        public double Score { get; set; }
    }

    static NivaraFrame CreateRegionFrame()
    {
        var names = NivaraColumn<string>.CreateForReferenceType(new[] { "us-east-1", "eu-west-1", "ap-south-1", "us-west-2" });
        var errorRates = NivaraColumn<double>.Create(new[] { 0.12, 0.05, 0.25, 0.08 });
        var counts = NivaraColumn<int>.Create(new[] { 100, 200, 50, 150 });

        return NivaraFrame.Create(
            ("Name", names),
            ("ErrorRate", errorRates),
            ("RequestCount", counts));
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
            ("Salary", salaries));
    }
}
