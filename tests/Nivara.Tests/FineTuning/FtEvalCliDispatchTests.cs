using NivaraFineTuning;
using NUnit.Framework;

namespace Nivara.Tests.FineTuning;

[TestFixture]
[NonParallelizable]
public sealed class FtEvalCliDispatchTests
{
    [Test]
    public void RunFtEval_SilkTrue_DispatchesToSilkPathWithoutCpuFallback()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "nivara-ft-eval-dispatch-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        var originalOut = Console.Out;
        var originalErr = Console.Error;
        using var stdout = new StringWriter();
        using var stderr = new StringWriter();

        try
        {
            Console.SetOut(stdout);
            Console.SetError(stderr);

            var testData = Path.Combine(tempDir, "test.jsonl");
            File.WriteAllText(testData, """{"instruction":"i","input":"in","output":"out"}""" + Environment.NewLine);

            var cli = new CliArgs
            {
                Mode = "ft-eval",
                Silk = true,
                ModelDir = tempDir,
                TestData = testData,
                OutDir = tempDir,
                Template = "qwen",
                MaxExamples = 1,
            };

            Exception? thrown = null;
            try
            {
                NivaraFineTuning.Program.RunFtEval(cli);
            }
            catch (Exception ex)
            {
                thrown = ex;
            }

            var output = stdout.ToString() + stderr.ToString();

            Assert.That(output, Does.Contain("=== Nivara LoRA SFT Eval (Silk) ==="),
                "Silk ft-eval should enter the Silk path and print the Silk header.");

            var outputWithoutSilkHeader = output.Replace("=== Nivara LoRA SFT Eval (Silk) ===", string.Empty, StringComparison.Ordinal);
            Assert.That(outputWithoutSilkHeader, Does.Not.Contain("=== Nivara LoRA SFT Eval ==="),
                "Silk ft-eval must not silently run the CPU eval path.");
            Assert.That(output, Does.Not.Contain("Backend=\"cpu\""),
                "Silk ft-eval dispatch must not produce CPU backend output.");

            var cpuReport = Path.Combine(tempDir, "ft-eval-report.json");
            if (File.Exists(cpuReport))
            {
                var report = File.ReadAllText(cpuReport);
                Assert.That(report, Does.Not.Contain("\"backend\": \"cpu\"").IgnoreCase,
                    "A CPU ft-eval report must not be written when --silk dispatches to Silk.");
            }

            // The temp dir has no tokenizer/model files, so Silk path is expected to fail
            // before opening a session. The failure must come from Silk dispatch, not CPU eval.
            var failedInSilkPath = thrown != null
                || output.Contains("ERROR: OpenCL/Silk unavailable", StringComparison.Ordinal);
            Assert.That(failedInSilkPath, Is.True,
                "Silk dispatch should fail on missing model assets or unavailable OpenCL, not silently succeed on CPU.");
        }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalErr);
            try { Directory.Delete(tempDir, recursive: true); } catch { /* best-effort cleanup */ }
        }
    }
}
