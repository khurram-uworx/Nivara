using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Minimal ONNX export context used by <see cref="Module{T}"/> implementations to emit
/// nodes into a graph without taking a dependency on the ONNX protobuf layer in core.
/// </summary>
public interface IOnnxExportContext<T> where T : struct, IFloatingPointIeee754<T>
{
    string Linear(string inputName, Linear<T> module);
    string Conv1d(string inputName, Conv1d<T> module);
    string Conv2d(string inputName, Conv2d<T> module);
    string Embedding(string inputName, Embedding<T> module);
    string BatchNorm1d(string inputName, BatchNorm1d<T> module);
    string BatchNorm2d(string inputName, BatchNorm2d<T> module, int? n = null, int? h = null, int? w = null);
    string LayerNorm(string inputName, LayerNorm<T> module);
    string RMSNorm(string inputName, RMSNorm<T> module);
    string TransformerBlock(string inputName, TransformerBlock<T> module, int sequenceLength);
    string MultiheadAttention(string inputName, MultiheadAttention<T> module);
    string Sequential(string inputName, Sequential<T> module);
    string Relu(string inputName);
    string Gelu(string inputName);
    string Sigmoid(string inputName);
    string Tanh(string inputName);
    string Add(string left, string right);
}