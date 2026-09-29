using System.Reflection;
using System.Reflection.Emit;
using Nivara.Samples.Gpu;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Enforces "at most one shared-memory allocation per kernel body" on the compiled IL of the
/// GPU kernels, which is the invariant that survives ILGPU's OpenCL lowering (#468).
/// </summary>
/// <remarks>
/// Two <c>SharedMemory.Allocate2D</c> calls in one kernel whose extents disagree get the second
/// tile mis-placed by the generated OpenCL C, and the symptom is plausible wrong numbers rather
/// than a compile error — the shape of defect that a tolerance-based gate argues its way past and
/// that no amount of code review catches, because the C# is correct. See
/// docs/ACCELERATION.md lesson 19.
///
/// <para>
/// The rule is deliberately "one allocation", not "allocations with matching extents". Matching
/// extents is what the four #440 kernels were first written against, on the reasoning that the
/// incumbent Row4 escaped because its A tile was square. That reasoning is contradicted by the
/// issue's own 1x8@KT16 row, which has a square 16x16 A tile and still fails, and it rests on one
/// observed success (2x2@KT32, identical 32x32 extents) that is equally consistent with "this
/// build happens to place them correctly". An exemption list keyed on the second reading would be
/// a rot vector: the next geometry nobody checked gets exempted on the same discredited premise.
/// </para>
///
/// <para>
/// Scanning compiled IL rather than source text is what makes the guard survive reformatting,
/// comment edits and a moved file, and matching on <c>DeclaringType.FullName</c> rather than on an
/// ILGPU reference means the test needs no compile-time dependency on the pinned ILGPU version.
/// The name prefix <c>Allocate</c> covers every allocation entry point ILGPU 1.5.3 exposes —
/// <c>Allocate</c>, <c>Allocate1D/2D/3D</c> and their <c>DenseX/Y/ZY</c> shorthands — so a mixed
/// pair like <c>Allocate</c> + <c>Allocate2D</c> is caught, and a future overload is caught without
/// editing this file.
/// </para>
///
/// <para>
/// Stated limitation: a kernel that moved its allocation into a helper method and called that
/// helper twice would not be caught, because the scan sees one call in the kernel body. No such
/// shape exists in this tree.
/// </para>
/// </remarks>
[TestFixture]
public class SharedMemoryAllocationTests
{
    const string SharedMemoryTypeName = "ILGPU.SharedMemory";

    const string AllocationNamePrefix = "Allocate";

    /// <summary>Floor for the non-vacuity assertion: GemmKernels declares ten
    /// <c>TiledGemmKernel*</c> methods today. A rename that dropped them below this must fail the
    /// test loudly rather than leave it scanning nothing and reporting a clean run.</summary>
    const int MinimumGemmKernels = 10;

    static readonly Dictionary<short, OpCode> s_opCodes = BuildOpCodeTable();

    [Test]
    public void GpuKernels_AtMostOneSharedMemoryAllocationPerMethod()
    {
        Type[] gpuTypes = typeof(GemmKernels).Assembly
            .GetTypes()
            .Where(type => type.Namespace?.StartsWith("Nivara.Samples.Gpu", StringComparison.Ordinal) == true)
            .ToArray();

        Assert.That(gpuTypes, Is.Not.Empty,
            "no Nivara.Samples.Gpu types were found, so nothing was scanned; the guard would pass vacuously.");

        var scans = new List<AllocationScan>();
        foreach (Type type in gpuTypes)
        {
            foreach (MethodBase method in type.GetMethods(
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly))
            {
                scans.Add(Scan(method));
            }
        }

        int gemmKernels = scans.Count(scan => scan.Method.Contains("TiledGemmKernel", StringComparison.Ordinal));
        Assert.That(gemmKernels, Is.GreaterThanOrEqualTo(MinimumGemmKernels),
            $"only {gemmKernels} TiledGemmKernel* methods were scanned, expected at least {MinimumGemmKernels}. " +
            "The kernels were renamed or moved out of Nivara.Samples.Gpu, so this guard is no longer " +
            "looking at the code it exists to protect.");

        // Its own verdict, separate from the allocation count: a call token this scan could not
        // resolve means the allocation count for that method is not established, which is not the
        // same as it being within the limit. Never counted as a pass, never folded into the
        // offenders below.
        var unresolved = scans.Where(scan => scan.UnresolvedCalls > 0).ToList();
        Assert.That(unresolved.Select(scan => $"{scan.Method} ({scan.UnresolvedCalls} unresolvable call tokens)"), Is.Empty,
            "some call tokens in these methods could not be resolved to a target, so their shared-allocation " +
            "count is unknown rather than verified. This usually means a new dependency needs a generic " +
            "instantiation context passed to Module.ResolveMethod, or that an assembly is missing at run time.");

        var offenders = scans.Where(scan => scan.Allocations > 1).ToList();
        Assert.That(offenders.Select(scan => $"{scan.Method} ({scan.Allocations} allocations)"), Is.Empty,
            "these kernel methods make more than one SharedMemory allocation, which ILGPU's OpenCL lowering " +
            "can mis-place (#468). Stage every tile through a single allocation with hand-computed offsets - " +
            "see the banner in samples/Nivara.Samples/Gpu/GemmKernels.cs and docs/ACCELERATION.md lesson 19.");
    }

    /// <summary>What one method's IL says about its shared allocations.</summary>
    /// <param name="Method">Type and method name, for the failure message.</param>
    /// <param name="Allocations">Calls into <c>ILGPU.SharedMemory.Allocate*</c>.</param>
    /// <param name="UnresolvedCalls">Call tokens whose target could not be resolved, so the
    /// allocation count for this method is unknown rather than low.</param>
    sealed record AllocationScan(string Method, int Allocations, int UnresolvedCalls);

    static AllocationScan Scan(MethodBase method)
    {
        string name = $"{method.DeclaringType?.Name}.{method.Name}";
        MethodBody? body = method.GetMethodBody();
        byte[]? il = body?.GetILAsByteArray();
        if (il is null)
            return new AllocationScan(name, 0, 0);

        int allocations = 0;
        int unresolved = 0;
        int position = 0;

        while (position < il.Length)
        {
            OpCode op = ReadOpCode(il, ref position);

            if (op.OperandType == OperandType.InlineSwitch)
            {
                position += sizeof(int) + (sizeof(int) * BitConverter.ToInt32(il, position));
                continue;
            }

            if (op == OpCodes.Call || op == OpCodes.Callvirt)
            {
                MethodBase? target = ResolveOrNull(method.Module, BitConverter.ToInt32(il, position));
                if (target is null)
                    unresolved++;
                else if (IsSharedAllocation(target))
                    allocations++;
            }

            position += OperandSize(op);
        }

        return new AllocationScan(name, allocations, unresolved);
    }

    /// <summary>
    /// Resolves a call token, or null if this scan cannot establish what it points at. Both the
    /// kernels and the ILGPU allocation methods carry no enclosing generic instantiation, so
    /// null type and method arguments are the documented pairing; a token that needs a context
    /// lands on the null path rather than being guessed at.
    /// </summary>
    static MethodBase? ResolveOrNull(Module module, int token)
    {
        try
        {
            return module.ResolveMethod(token, genericTypeArguments: null, genericMethodArguments: null);
        }
        catch (Exception ex) when (ex is ArgumentException
            or BadImageFormatException
            or MissingMethodException
            or TypeLoadException
            or FileNotFoundException)
        {
            return null;
        }
    }

    static bool IsSharedAllocation(MethodBase target)
        => target.DeclaringType?.FullName == SharedMemoryTypeName
           && target.Name.StartsWith(AllocationNamePrefix, StringComparison.Ordinal);

    static OpCode ReadOpCode(byte[] il, ref int position)
    {
        int start = position;

        // Two-byte opcodes are 0xFE followed by the second byte, and System.Reflection.Emit keys
        // them as 0xFE00 | second.
        if (il[position] == 0xFE)
        {
            position += 2;
            short twoByte = (short)(0xFE00 | il[position - 1]);
            if (!s_opCodes.TryGetValue(twoByte, out OpCode prefixed))
                throw new InvalidOperationException($"unrecognised two-byte opcode 0x{il[start]:X2}{il[start + 1]:X2} at IL offset {start}");
            return prefixed;
        }

        position += 1;
        if (!s_opCodes.TryGetValue(il[position - 1], out OpCode single))
            throw new InvalidOperationException($"unrecognised opcode 0x{il[start]:X2} at IL offset {start}");
        return single;
    }

    static int OperandSize(OpCode op) => op.OperandType switch
    {
        OperandType.InlineNone => 0,
        OperandType.ShortInlineBrTarget => sizeof(sbyte),
        OperandType.ShortInlineI => sizeof(sbyte),
        OperandType.ShortInlineVar => sizeof(byte),
        OperandType.InlineVar => sizeof(short),
        OperandType.InlineBrTarget => sizeof(int),
        OperandType.InlineField => sizeof(int),
        OperandType.InlineI => sizeof(int),
        OperandType.InlineMethod => sizeof(int),
        OperandType.InlineSig => sizeof(int),
        OperandType.InlineString => sizeof(int),
        OperandType.InlineTok => sizeof(int),
        OperandType.InlineType => sizeof(int),
        OperandType.ShortInlineR => sizeof(float),
        OperandType.InlineI8 => sizeof(long),
        OperandType.InlineR => sizeof(double),
        OperandType.InlineSwitch => throw new InvalidOperationException("switch operands are sized by their case count, not a fixed width"),
        _ => throw new InvalidOperationException($"unhandled operand type {op.OperandType} for opcode {op.Name}"),
    };

    static Dictionary<short, OpCode> BuildOpCodeTable()
    {
        var table = new Dictionary<short, OpCode>();
        foreach (FieldInfo field in typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (field.GetValue(null) is OpCode op)
                table[op.Value] = op;
        }

        return table;
    }
}
