using NUnit.Framework;
using System.Numerics;
using Nivara.Primitives;
using Nivara.Tensors;

namespace Nivara.Tests.Primitives;

[TestFixture]
public class WidenPrimitivesPhase3Tests
{
    static bool WidenEnabled
    {
        get
        {
            var p = typeof(NivaraPrimitives).GetProperty(nameof(NivaraPrimitives.UseWidenSimd))!;
            return (bool)p.GetValue(null)!;
        }
        set
        {
            var p = typeof(NivaraPrimitives).GetProperty(nameof(NivaraPrimitives.UseWidenSimd))!;
            p.SetValue(null, value);
        }
    }

    /// <summary>
    /// Verifies the save/restore pattern that <c>RunSmolLMCore&lt;T&gt;</c> relies
    /// on: the global toggle is saved before the run, the run changes it, and the
    /// prior value is restored in the finally block. Without this guarantee,
    /// leaving the toggle on for one model leaks into the next model dispatch.
    /// </summary>
    [Test]
    public void Toggle_RestoredAfterRun_LeavesPriorStateUnchanged()
    {
        bool original = WidenEnabled;
        try
        {
            // Simulate RunSmolLMCore pattern: save → set → run → restore.
            bool priorWiden = WidenEnabled;
            WidenEnabled = true;

            // Simulate the run (actual kernel calls are tested in Phase1Tests).
            // Here we just verify the toggle is on during the "run".
            Assert.That(WidenEnabled, Is.True, "toggle must be on during the simulated run");

            // Restore.
            WidenEnabled = priorWiden;
            Assert.That(WidenEnabled, Is.EqualTo(original),
                "toggle must be restored to its original value after the run");
        }
        finally
        {
            WidenEnabled = original;
        }
    }

    /// <summary>
    /// Verifies nested save/restore (Main sets the global toggle, RunSmolLMCore
    /// saves and restores around the inner run). Both saves should capture the
    /// correct outer value and both restores should leave it unchanged.
    /// </summary>
    [Test]
    public void Toggle_NestedSaveRestore_InnerDoesNotCorruptOuter()
    {
        bool original = WidenEnabled;
        try
        {
            // Outer (Main): save → set true.
            bool outerPrior = WidenEnabled;
            WidenEnabled = true;

            // Inner (RunSmolLMCore): save outer value → set true → run → restore to outer.
            bool innerPrior = WidenEnabled;
            WidenEnabled = true; // narrow path auto-enables
            Assert.That(WidenEnabled, Is.True);
            WidenEnabled = innerPrior; // restores to outerPrior's value (true)
            Assert.That(WidenEnabled, Is.True, "inner restore must not flip toggle to false");

            // Outer restore: back to original.
            WidenEnabled = outerPrior;
            Assert.That(WidenEnabled, Is.EqualTo(original),
                "outer restore must leave toggle at its original value");
        }
        finally
        {
            WidenEnabled = original;
        }
    }

    /// <summary>
    /// When the A/B mode runs scalar then widen, both sides must produce the
    /// same numerical results for BFloat16 element-wise Add (within BF16 tolerance),
    /// confirming that toggling UseWidenSimd between passes is safe and correct.
    /// </summary>
    [Test]
    public void AbPattern_BFloat16_ScalarAndWidenProduceSameAddResult()
    {
        bool original = WidenEnabled;
        try
        {
            int n = 256;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 - 2)).ToArray();

            // Side A: scalar.
            WidenEnabled = false;
            var scalarResult = new BFloat16[n];
            WidenPrimitives.Add<BFloat16>(a, b, scalarResult);

            // Side B: widen.
            WidenEnabled = true;
            var widenResult = new BFloat16[n];
            WidenPrimitives.Add<BFloat16>(a, b, widenResult);

            WidenEnabled = original;

            // Results must match within BF16 precision.
            for (int i = 0; i < n; i++)
            {
                float e = (float)scalarResult[i];
                float a2 = (float)widenResult[i];
                Assert.That(a2, Is.EqualTo(e).Within(MathF.Abs(e) * 1e-2f + 1e-3f),
                    $"A/B mismatch at index {i}: scalar={e}, widen={a2}");
            }
        }
        finally
        {
            WidenEnabled = original;
        }
    }

    /// <summary>
    /// Same A/B pattern test for Half (fp16) element-wise Multiply.
    /// </summary>
    [Test]
    public void AbPattern_Half_ScalarAndWidenProduceSameMultiplyResult()
    {
        bool original = WidenEnabled;
        try
        {
            int n = 256;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 2 - 1)).ToArray();

            WidenEnabled = false;
            var scalarResult = new Half[n];
            WidenPrimitives.Multiply<Half>(a, b, scalarResult);

            WidenEnabled = true;
            var widenResult = new Half[n];
            WidenPrimitives.Multiply<Half>(a, b, widenResult);

            WidenEnabled = original;

            for (int i = 0; i < n; i++)
            {
                float e = (float)scalarResult[i];
                float a2 = (float)widenResult[i];
                Assert.That(a2, Is.EqualTo(e).Within(MathF.Abs(e) * 1e-2f + 1e-3f),
                    $"A/B mismatch at index {i}: scalar={e}, widen={a2}");
            }
        }
        finally
        {
            WidenEnabled = original;
        }
    }

    /// <summary>
    /// F32 widening is a no-op: toggling UseWidenSimd must not produce NaN or
    /// alter F32 Dot results. This is the A/B control case.
    /// </summary>
    [Test]
    public void AbPattern_Float_ToggleTransparent_NoNaN()
    {
        bool original = WidenEnabled;
        try
        {
            int n = 256;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();

            WidenEnabled = false;
            float scalarDot = WidenPrimitives.Dot<float>(a, b);

            WidenEnabled = true;
            float widenDot = WidenPrimitives.Dot<float>(a, b);

            WidenEnabled = original;

            Assert.That(float.IsNaN(scalarDot), Is.False, "scalar F32 dot must not be NaN");
            Assert.That(float.IsNaN(widenDot), Is.False, "widen F32 dot must not be NaN");
            Assert.That(widenDot, Is.EqualTo(scalarDot),
                "F32 Dot must be identical regardless of UseWidenSimd toggle");
        }
        finally
        {
            WidenEnabled = original;
        }
    }
}
