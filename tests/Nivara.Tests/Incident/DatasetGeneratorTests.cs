using Nivara.Samples.Incident;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

/// <summary>
/// Pins the two properties of <see cref="DatasetGenerator.GenerateFromRecordCount"/> that #563
/// found broken: the written file carries exactly the requested number of rows, and every one of
/// those rows is a populated generated row rather than a default-filled tail.
/// <para>
/// Before the fix the request arrays were sized to <c>totalRecords</c> but only
/// <c>floor(totalRecords / 30) * 30</c> slots were filled, so the trailing 10 rows of a
/// 10,000-record file were defaults (<c>StatusCode == 0</c>, null <c>Service</c>/<c>Region</c>).
/// Both writers used the full arrays, so the defaults reached disk.
/// </para>
/// </summary>
[TestFixture]
public class DatasetGeneratorTests
{
    [Test]
    public void Generate_RecordCountThatIsAMultipleOf30_SpreadsEvenlyWithNoRemainder()
    {
        const int totalRecords = 9_000;

        WithGeneratedFrame(totalRecords, frame =>
        {
            Assert.That(frame.RowCount, Is.EqualTo(totalRecords));

            var perMinute = RowsPerMinute(frame);
            Assert.That(perMinute.Count, Is.EqualTo(30), "all 30 minutes should be represented");
            Assert.That(perMinute.Values, Is.All.EqualTo(totalRecords / 30),
                "a multiple of 30 leaves no remainder to distribute");
        });
    }

    [Test]
    public void Generate_RecordCountWithARemainder_DistributesItAcrossTheLeadingMinutes()
    {
        const int totalRecords = 10_003;
        const int perMinute = totalRecords / 30;
        const int remainder = totalRecords % 30;

        WithGeneratedFrame(totalRecords, frame =>
        {
            Assert.That(frame.RowCount, Is.EqualTo(totalRecords));

            var rowsPerMinute = RowsPerMinute(frame);
            Assert.Multiple(() =>
            {
                Assert.That(rowsPerMinute.Count, Is.EqualTo(30));
                for (int minute = 0; minute < 30; minute++)
                {
                    int expected = perMinute + (minute < remainder ? 1 : 0);
                    Assert.That(rowsPerMinute[minute], Is.EqualTo(expected),
                        $"minute {minute}");
                }
            });
        });
    }

    [Test]
    public void Generate_RecordCountBelowTheMinuteCount_WritesOneRowPerLeadingMinute()
    {
        const int totalRecords = 20;

        WithGeneratedFrame(totalRecords, frame =>
        {
            Assert.That(frame.RowCount, Is.EqualTo(totalRecords));

            var rowsPerMinute = RowsPerMinute(frame);
            Assert.That(rowsPerMinute.Count, Is.EqualTo(totalRecords),
                "when totalRecords < 30, requestsPerMinute is 0 and only the first minutes carry a row");
            Assert.That(rowsPerMinute.Values, Is.All.EqualTo(1));
        });
    }

    [Test]
    public void Generate_RequestedRows_AreAllPopulatedWithNoDefaults()
    {
        const int totalRecords = 10_003;

        WithGeneratedFrame(totalRecords, frame =>
        {
            var timestamp = frame.GetColumn<long>("Timestamp");
            var status = frame.GetColumn<int>("StatusCode");
            var duration = frame.GetColumn<double>("DurationMs");
            var service = frame.GetColumn<string>("Service");
            var endpoint = frame.GetColumn<string>("Endpoint");
            var region = frame.GetColumn<string>("Region");
            var traceId = frame.GetColumn<string>("TraceId");

            var allowedStatuses = new HashSet<int> { 200, 429, 500, 502, 503 };
            int zeroTimestamps = 0, badStatuses = 0, nonPositiveDurations = 0, missingText = 0;

            for (int i = 0; i < frame.RowCount; i++)
            {
                if ((long)timestamp.GetValue(i)! == 0) zeroTimestamps++;
                if (!allowedStatuses.Contains((int)status.GetValue(i)!)) badStatuses++;
                if ((double)duration.GetValue(i)! <= 0) nonPositiveDurations++;

                if (string.IsNullOrEmpty((string?)service.GetValue(i))
                    || string.IsNullOrEmpty((string?)endpoint.GetValue(i))
                    || string.IsNullOrEmpty((string?)region.GetValue(i))
                    || string.IsNullOrEmpty((string?)traceId.GetValue(i)))
                    missingText++;
            }

            Assert.Multiple(() =>
            {
                Assert.That(zeroTimestamps, Is.Zero, "no row should carry the default Timestamp 0");
                Assert.That(badStatuses, Is.Zero, "every StatusCode should be a generated value");
                Assert.That(nonPositiveDurations, Is.Zero, "every DurationMs should be > 0");
                Assert.That(missingText, Is.Zero, "no row should have an empty Service/Endpoint/Region/TraceId");
            });
        });
    }

    static void WithGeneratedFrame(long totalRecords, Action<NivaraFrame> assert)
    {
        var dir = Path.Combine(Path.GetTempPath(), $"inc-gen-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            DatasetGenerator.GenerateFromRecordCount(dir, "A", totalRecords);

            using var query = Ingestion.LoadParquet(Path.Combine(dir, "requests.parquet"));
            using var frame = query.Collect();
            assert(frame);
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, true);
        }
    }

    /// <summary>
    /// Buckets request rows by whole minute offset from the earliest <em>minute boundary</em>.
    /// Each row sits within the minute it was generated for (the sub-minute jitter is under
    /// 60,000 ms), so flooring the earliest timestamp to its minute boundary and dividing recovers
    /// the generator's minute index exactly. Dividing from the earliest actual timestamp instead
    /// would misassign rows whose jitter straddles the earliest row's.
    /// </summary>
    static Dictionary<int, int> RowsPerMinute(NivaraFrame frame)
    {
        var timestamp = frame.GetColumn<long>("Timestamp");

        long earliest = long.MaxValue;
        for (int i = 0; i < frame.RowCount; i++)
            earliest = Math.Min(earliest, (long)timestamp.GetValue(i)!);
        long firstMinute = earliest - earliest % TimeSpan.TicksPerMinute;

        var counts = new Dictionary<int, int>();
        for (int i = 0; i < frame.RowCount; i++)
        {
            int minute = (int)(((long)timestamp.GetValue(i)! - firstMinute) / TimeSpan.TicksPerMinute);
            counts[minute] = counts.GetValueOrDefault(minute) + 1;
        }

        return counts;
    }
}
