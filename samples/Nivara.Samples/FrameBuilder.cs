namespace Nivara.Samples;

public static class FrameBuilder
{
    public static NivaraFrame BuildTokenClassificationFrame(
        int[] tokens, int[] labels, int count, int seqLen)
    {
        var columns = new List<(string Name, IColumn Column)>();

        for (int d = 0; d < seqLen; d++)
        {
            var tokData = new float[count];
            var lblData = new float[count];
            for (int i = 0; i < count; i++)
            {
                tokData[i] = tokens[i * seqLen + d];
                lblData[i] = labels[i * seqLen + d];
            }
            columns.Add(($"tok_{d}", NivaraColumn<float>.Create(tokData)));
            columns.Add(($"lbl_{d}", NivaraColumn<float>.Create(lblData)));
        }

        return NivaraFrame.Create(columns.ToArray());
    }

    public static NivaraFrame BuildDocumentClassificationFrame(
        int[] tokens, int[] labels, int count, int seqLen)
    {
        var columns = new List<(string Name, IColumn Column)>();

        for (int d = 0; d < seqLen; d++)
        {
            var colData = new float[count];
            for (int i = 0; i < count; i++)
                colData[i] = tokens[i * seqLen + d];
            columns.Add(($"tok_{d}", NivaraColumn<float>.Create(colData)));
        }

        var labelData = new float[count];
        for (int i = 0; i < count; i++)
            labelData[i] = labels[i];
        columns.Add(("label", NivaraColumn<float>.Create(labelData)));

        return NivaraFrame.Create(columns.ToArray());
    }

    public static NivaraFrame BuildDocumentClassificationFrameDouble(
        int[] tokens, int[] labels, int count, int seqLen)
    {
        var columns = new List<(string Name, IColumn Column)>();

        for (int d = 0; d < seqLen; d++)
        {
            var colData = new double[count];
            for (int i = 0; i < count; i++)
                colData[i] = tokens[i * seqLen + d];
            columns.Add(($"tok_{d}", NivaraColumn<double>.Create(colData)));
        }

        var labelData = new double[count];
        for (int i = 0; i < count; i++)
            labelData[i] = labels[i];
        columns.Add(("label", NivaraColumn<double>.Create(labelData)));

        return NivaraFrame.Create(columns.ToArray());
    }
}
