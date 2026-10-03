using NUnit.Framework;

namespace Nivara.Tests.IO;

/// <summary>
/// Deterministic probe for "is any handle still open on this file?".
/// <para>
/// The lazy sources (<c>CsvLazySource</c>, <c>JsonLazySource</c>, <c>ParquetLazySource</c>) open
/// their backing file with <c>FileShare.Read</c> and no <c>FileShare.Delete</c>, so a handle that
/// outlives its query blocks deletion on Windows with <see cref="IOException"/> — the misleading
/// "being used by another process" teardown failure behind issues #496 and #498.
/// </para>
/// <para>
/// Probing with <c>FileShare.None</c> fails while any other handle is open. That detects the leak
/// directly, instead of inferring it from whether a delete happened to fail — which depends on
/// finalizer timing and so reports intermittently.
/// </para>
/// <para>
/// <b>Platform coverage.</b> On Windows the probe fails via the mandatory share-mode check. On
/// Unix, .NET maps share modes onto <c>flock(2)</c> advisory locks and unlinking an open file
/// always succeeds, so a delete-based assertion can never fail there. Whether this probe has
/// teeth on the Unix CI runner is asserted by
/// <c>ScanAsQueryFrameHandleTests.Probe_DetectsDeliberatelyLeakedHandle</c>, which fails loudly
/// rather than skipping — a gate that cannot see a leak on the platform it runs on is worse than
/// no gate, because it reads as coverage.
/// </para>
/// </summary>
static class FileHandleProbe
{
    /// <summary>
    /// Asserts no handle other than the probe's own is open on <paramref name="path"/>.
    /// </summary>
    /// <param name="path">The file to probe.</param>
    /// <param name="because">
    /// Why this probe matters, phrased to complete "… because <paramref name="because"/>".
    /// </param>
    internal static void AssertUnlocked(string path, string because)
    {
        try
        {
            using var probe = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None);
        }
        catch (IOException ex)
        {
            Assert.Fail($"'{Path.GetFileName(path)}' is still locked: {because}. {ex.Message}");
        }
    }

    /// <summary>
    /// Asserts a handle <em>is</em> currently open on <paramref name="path"/>, i.e. that the
    /// probe would detect a leak here.
    /// <para>
    /// This is what separates a meaningful gate from a vacuous one. A case that reads to EOF and
    /// then asserts the file is unlocked passes whether or not the consumer ever disposed
    /// anything — <c>CsvLazySource</c> and <c>JsonLazySource</c> both close their chunk reader at
    /// EOF on their own. Asserting the handle is still open at the midpoint proves the case is
    /// exercising the release path rather than the source's self-release.
    /// </para>
    /// </summary>
    /// <param name="path">The file to probe.</param>
    /// <param name="because">Why the handle must be open at this point.</param>
    internal static void AssertLocked(string path, string because)
    {
        try
        {
            using var probe = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None);
        }
        catch (IOException)
        {
            return;
        }

        Assert.Fail(
            $"'{Path.GetFileName(path)}' was expected to be locked: {because}. The probe opened it "
            + "with FileShare.None, so no other handle is open — this case cannot demonstrate "
            + "handle release, and FileShare.None may have no effect on this platform. If the "
            + "latter, the gate is reporting nothing here and the platform claim in the "
            + "negative-control test needs to say so.");
    }

    /// <summary>
    /// Probes <paramref name="path"/>, runs <paramref name="open"/> and <paramref name="release"/>,
    /// then probes again.
    /// <para>
    /// The leading probe is what makes a red verdict attributable: it separates "an earlier case
    /// leaked this file" from "this case leaked it" instead of leaving an unlocalized red.
    /// </para>
    /// </summary>
    /// <typeparam name="T">The disposable-or-frame type the entry point returns.</typeparam>
    /// <param name="path">The backing file the entry point opens.</param>
    /// <param name="consumer">The entry point's name, for the failure message.</param>
    /// <param name="open">Opens the consumer.</param>
    /// <param name="release">Releases it. Called only if <paramref name="open"/> returned non-null.</param>
    internal static void AssertReleasesAfter<T>(string path, string consumer, Func<T> open, Action<T> release)
    {
        AssertUnlocked(path, $"a handle was already open before {consumer} ran");

        var consumerInstance = open();
        Assert.That(consumerInstance, Is.Not.Null, $"{consumer} returned nothing to dispose");
        release(consumerInstance!);

        AssertUnlocked(path, $"{consumer} left the file handle open");
    }
}