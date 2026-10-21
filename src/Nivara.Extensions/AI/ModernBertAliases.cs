using System.Numerics;

namespace Nivara.AI;

public sealed class ModernBertAttention<T> : ModernBertAttention<T, T> where T : struct, IFloatingPointIeee754<T>
{
    public ModernBertAttention(ModernBertConfig config, float ropeTheta) : base(config, ropeTheta) { }
}

public sealed class ModernBertMlp<T> : ModernBertMlp<T, T> where T : struct, IFloatingPointIeee754<T>
{
    public ModernBertMlp(ModernBertConfig config) : base(config) { }
}

public sealed class ModernBertLayer<T> : ModernBertLayer<T, T> where T : struct, IFloatingPointIeee754<T>
{
    public ModernBertLayer(ModernBertConfig config, int layerIndex) : base(config, layerIndex) { }
}

public sealed class ModernBertEncoder<T> : ModernBertEncoder<T, T> where T : struct, IFloatingPointIeee754<T>
{
    public ModernBertEncoder(ModernBertConfig config) : base(config) { }
}
