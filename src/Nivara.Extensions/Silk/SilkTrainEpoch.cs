namespace Nivara.Silk;

/// <summary>Per-epoch training outcome: mean cross-entropy loss and wall-clock duration.
/// Both Silk classifiers (<see cref="GpuTextLinearClassifier"/>, <see cref="GpuTextConvClassifier"/>)
/// return this from <c>Train</c>.</summary>
public readonly record struct SilkTrainEpoch(int Epoch, float Loss, TimeSpan Elapsed);
