using System.IO;
using System.Text.RegularExpressions;
using NUnit.Framework;

namespace Nivara.Tests.Quality;

[TestFixture]
public class ConfigureAwaitEnforcementTests
{
    [Test]
    public void EverySimpleAwait_HasConfigureAwaitFalse()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !Directory.Exists(Path.Combine(dir.FullName, "src", "Nivara")))
            dir = dir.Parent;
        var root = dir == null ? string.Empty : Path.Combine(dir.FullName, "src", "Nivara");

        var violations = new List<string>();
        foreach (var file in Directory.EnumerateFiles(root, "*.cs", SearchOption.AllDirectories))
        {
            var lines = File.ReadAllLines(file);
            for (int i = 0; i < lines.Length; i++)
            {
                var line = lines[i];
                var trimmed = line.TrimStart();
                if (trimmed.StartsWith("//")) continue;
                if (!Regex.IsMatch(line, @"\bawait\b")) continue;
                var window = string.Join(" ", lines.Skip(i).Take(20));
                if (window.Contains("ConfigureAwait(false)")) continue;
                if (line.Contains("await foreach")) continue;
                if (line.Contains("await using")) continue;
                violations.Add($"{Path.GetRelativePath(root, file)}:{i + 1}: {line.Trim()}");
            }
        }

        Assert.That(violations, Is.Empty, string.Join(Environment.NewLine, violations));
    }
}
