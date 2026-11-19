namespace NivaraChat.Helpers;

/// <summary>
/// Renders the per-component backend matrix for a run. Reporting only.
/// </summary>
public static class BackendMatrixPrinter
{
    public static void Print(BackendMatrix matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        Console.WriteLine("Backend matrix:");
        Console.WriteLine($"  {matrix.Embeddings.Component,-18}: {matrix.Embeddings.Backend,-24} {matrix.Embeddings.Precision}");
        Console.WriteLine($"  {matrix.Llm.Component,-18}: {matrix.Llm.Backend,-24} {matrix.Llm.Precision}");
        foreach (var cb in matrix.Classifiers)
            Console.WriteLine($"  {cb.Component,-18}: {cb.Backend,-24} {cb.Precision}");
        Console.WriteLine($"  {matrix.LayaEncoder.Component,-18}: {matrix.LayaEncoder.Backend,-24} {matrix.LayaEncoder.Precision}");
        Console.WriteLine($"  {matrix.LayaHeadTraining.Component,-18}: {matrix.LayaHeadTraining.Backend,-24} {matrix.LayaHeadTraining.Precision}");
        Console.WriteLine();
    }
}