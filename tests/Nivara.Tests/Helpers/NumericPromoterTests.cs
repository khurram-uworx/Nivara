using Nivara.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Helpers;

[TestFixture]
public class NumericPromoterTests
{
    [Test]
    public void GetPromotedType_ReturnsCSharpPromotedType()
    {
        var cases = new (Type Left, Type Right, Type Expected)[]
        {
            (typeof(int), typeof(int), typeof(int)),
            (typeof(double), typeof(double), typeof(double)),
            (typeof(int), typeof(double), typeof(double)),
            (typeof(double), typeof(int), typeof(double)),
            (typeof(int), typeof(long), typeof(long)),
            (typeof(long), typeof(int), typeof(long)),
            (typeof(int), typeof(float), typeof(float)),
            (typeof(float), typeof(double), typeof(double)),
            (typeof(int), typeof(decimal), typeof(decimal)),
            (typeof(decimal), typeof(int), typeof(decimal)),
            (typeof(decimal), typeof(double), typeof(double)),
            (typeof(byte), typeof(int), typeof(int)),
            (typeof(short), typeof(int), typeof(int)),
            (typeof(byte), typeof(short), typeof(int)),
            (typeof(sbyte), typeof(ushort), typeof(int)),
            (typeof(int), typeof(uint), typeof(long)),
            (typeof(uint), typeof(long), typeof(long)),
            (typeof(byte), typeof(ulong), typeof(ulong)),
            (typeof(uint), typeof(ulong), typeof(ulong)),
            (typeof(int), typeof(ulong), typeof(double)),
            (typeof(long), typeof(ulong), typeof(double)),
            (typeof(sbyte), typeof(sbyte), typeof(int)),
            (typeof(byte), typeof(byte), typeof(int)),
            (typeof(short), typeof(short), typeof(int)),
            (typeof(ushort), typeof(ushort), typeof(int)),
            (typeof(char), typeof(char), typeof(int)),
            (typeof(char), typeof(byte), typeof(int)),
            (typeof(char), typeof(sbyte), typeof(int)),
            (typeof(char), typeof(short), typeof(int)),
            (typeof(char), typeof(ushort), typeof(int)),
            (typeof(char), typeof(int), typeof(int)),
            (typeof(char), typeof(uint), typeof(uint)),
            (typeof(char), typeof(long), typeof(long)),
            (typeof(char), typeof(ulong), typeof(ulong)),
            (typeof(char), typeof(float), typeof(float)),
            (typeof(char), typeof(double), typeof(double)),
            (typeof(char), typeof(decimal), typeof(decimal)),
            (typeof(char), typeof(Half), typeof(double)),
            (typeof(byte), typeof(uint), typeof(uint)),
            (typeof(ushort), typeof(uint), typeof(uint)),
            (typeof(uint), typeof(uint), typeof(uint)),
            (typeof(decimal), typeof(decimal), typeof(decimal)),
            (typeof(float), typeof(float), typeof(float)),
            (typeof(Half), typeof(Half), typeof(Half))
        };

        foreach (var (left, right, expected) in cases)
        {
            Assert.That(NumericPromoter.GetPromotedType(left, right), Is.EqualTo(expected), $"{left.Name} + {right.Name}");
            Assert.That(NumericPromoter.GetPromotedType(right, left), Is.EqualTo(expected), $"{right.Name} + {left.Name}");
        }
    }

    [Test]
    public void GetPromotedType_ExtendedNumericDomain_ResolvesNativeAndWidePairs()
    {
        // All cases verified against the C# compiler (non-constant operands). Pairs that are
        // binding-time errors in C# resolve to the safe superset double (repo convention).
        var cases = new (Type Left, Type Right, Type Expected)[]
        {
            // nint absorbs small signed/unsigned integrals and int
            (typeof(nint), typeof(byte), typeof(nint)),
            (typeof(nint), typeof(sbyte), typeof(nint)),
            (typeof(nint), typeof(ushort), typeof(nint)),
            (typeof(nint), typeof(short), typeof(nint)),
            (typeof(nint), typeof(char), typeof(nint)),
            (typeof(nint), typeof(int), typeof(nint)),
            // nint promotes to long with uint/long
            (typeof(nint), typeof(uint), typeof(long)),
            (typeof(nint), typeof(long), typeof(long)),
            // nint mixes with ulong/nuint are binding-time errors -> double
            (typeof(nint), typeof(ulong), typeof(double)),
            (typeof(nint), typeof(nuint), typeof(double)),
            // nint with floating/decimal stays with the wider float family
            (typeof(nint), typeof(float), typeof(float)),
            (typeof(nint), typeof(double), typeof(double)),
            (typeof(nint), typeof(decimal), typeof(decimal)),
            (typeof(nint), typeof(Half), typeof(double)),
            // nint vs 128-bit
            (typeof(nint), typeof(Int128), typeof(Int128)),
            (typeof(nint), typeof(UInt128), typeof(double)),

            // nuint absorbs unsigned integrals
            (typeof(nuint), typeof(byte), typeof(nuint)),
            (typeof(nuint), typeof(ushort), typeof(nuint)),
            (typeof(nuint), typeof(char), typeof(nuint)),
            (typeof(nuint), typeof(uint), typeof(nuint)),
            (typeof(nuint), typeof(nuint), typeof(nuint)),
            // nuint + ulong stays ulong (implicit nuint -> ulong)
            (typeof(nuint), typeof(ulong), typeof(ulong)),
            // nuint with signed integrals is a binding-time error -> double
            (typeof(nuint), typeof(sbyte), typeof(double)),
            (typeof(nuint), typeof(short), typeof(double)),
            (typeof(nuint), typeof(int), typeof(double)),
            (typeof(nuint), typeof(long), typeof(double)),
            (typeof(nuint), typeof(nint), typeof(double)),
            // nuint with floating/decimal
            (typeof(nuint), typeof(float), typeof(float)),
            (typeof(nuint), typeof(double), typeof(double)),
            (typeof(nuint), typeof(decimal), typeof(decimal)),
            (typeof(nuint), typeof(Half), typeof(double)),
            // nuint vs 128-bit
            (typeof(nuint), typeof(Int128), typeof(Int128)),
            (typeof(nuint), typeof(UInt128), typeof(UInt128)),

            // Int128 absorbs every integral type
            (typeof(Int128), typeof(byte), typeof(Int128)),
            (typeof(Int128), typeof(sbyte), typeof(Int128)),
            (typeof(Int128), typeof(ushort), typeof(Int128)),
            (typeof(Int128), typeof(short), typeof(Int128)),
            (typeof(Int128), typeof(char), typeof(Int128)),
            (typeof(Int128), typeof(int), typeof(Int128)),
            (typeof(Int128), typeof(uint), typeof(Int128)),
            (typeof(Int128), typeof(long), typeof(Int128)),
            (typeof(Int128), typeof(ulong), typeof(Int128)),
            (typeof(Int128), typeof(nint), typeof(Int128)),
            (typeof(Int128), typeof(nuint), typeof(Int128)),
            (typeof(Int128), typeof(Int128), typeof(Int128)),
            // Int128 with UInt128/float/double/decimal/Half is a binding-time error -> double
            (typeof(Int128), typeof(UInt128), typeof(double)),
            (typeof(Int128), typeof(float), typeof(double)),
            (typeof(Int128), typeof(double), typeof(double)),
            (typeof(Int128), typeof(decimal), typeof(double)),
            (typeof(Int128), typeof(Half), typeof(double)),

            // UInt128 absorbs unsigned integrals only
            (typeof(UInt128), typeof(byte), typeof(UInt128)),
            (typeof(UInt128), typeof(ushort), typeof(UInt128)),
            (typeof(UInt128), typeof(char), typeof(UInt128)),
            (typeof(UInt128), typeof(uint), typeof(UInt128)),
            (typeof(UInt128), typeof(ulong), typeof(UInt128)),
            (typeof(UInt128), typeof(nuint), typeof(UInt128)),
            (typeof(UInt128), typeof(UInt128), typeof(UInt128)),
            // UInt128 with signed integrals/128 is a binding-time error -> double
            (typeof(UInt128), typeof(sbyte), typeof(double)),
            (typeof(UInt128), typeof(short), typeof(double)),
            (typeof(UInt128), typeof(int), typeof(double)),
            (typeof(UInt128), typeof(long), typeof(double)),
            (typeof(UInt128), typeof(nint), typeof(double)),
            (typeof(UInt128), typeof(Int128), typeof(double)),
            (typeof(UInt128), typeof(float), typeof(double)),
            (typeof(UInt128), typeof(double), typeof(double)),
            (typeof(UInt128), typeof(decimal), typeof(double)),
            (typeof(UInt128), typeof(Half), typeof(double))
        };

        foreach (var (left, right, expected) in cases)
        {
            Assert.That(NumericPromoter.GetPromotedType(left, right), Is.EqualTo(expected), $"{left.Name} + {right.Name}");
            Assert.That(NumericPromoter.GetPromotedType(right, left), Is.EqualTo(expected), $"{right.Name} + {left.Name}");
        }
    }

    [Test]
    public void GetPromotedType_NonNumericOperand_ReturnsNull()
    {
        Assert.That(NumericPromoter.GetPromotedType(typeof(int), typeof(string)), Is.Null);
        Assert.That(NumericPromoter.GetPromotedType(typeof(string), typeof(int)), Is.Null);
        Assert.That(NumericPromoter.GetPromotedType(typeof(bool), typeof(int)), Is.Null);
        Assert.That(NumericPromoter.GetPromotedType(typeof(Guid), typeof(double)), Is.Null);
        Assert.That(NumericPromoter.GetPromotedType(typeof(DateTime), typeof(DateTime)), Is.Null);
    }
}
