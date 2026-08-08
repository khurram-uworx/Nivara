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
            (typeof(long), typeof(ulong), typeof(double))
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
