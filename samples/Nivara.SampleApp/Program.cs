using Nivara.SampleApp;

Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║              Nivara DataFrame Library                        ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
Console.WriteLine();

// Run the diagnostics example
DiagnosticsExample.RunExample();

Console.WriteLine();
Console.WriteLine();

// Run the aggregate functions example
AggregateExample.Run();

Console.WriteLine();
Console.WriteLine();

// Run the automatic differentiation example
AutoDiffExample.Run();
