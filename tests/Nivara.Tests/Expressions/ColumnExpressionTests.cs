using NUnit.Framework;
using Nivara.Expressions;
using Nivara.Exceptions;

namespace Nivara.Tests.Expressions;

[TestFixture]
public class ColumnExpressionTests
{
    private Schema testSchema;

    [SetUp]
    public void SetUp()
    {
        testSchema = new Schema(new[]
        {
            ("Name", typeof(string)),
            ("Age", typeof(int)),
            ("Salary", typeof(double)),
            ("IsActive", typeof(bool))
        });
    }

    [Test]
    public void ColumnReference_ValidColumn_CreatesCorrectReference()
    {
        var colRef = new ColumnReference("Name", typeof(string));
        
        Assert.That(colRef.ColumnName, Is.EqualTo("Name"));
        Assert.That(colRef.ResultType, Is.EqualTo(typeof(string)));
        Assert.That(colRef.Name, Is.EqualTo("Name"));
        Assert.That(colRef.ToString(), Is.EqualTo("Col(Name)"));
    }

    [Test]
    public void ColumnReference_NullOrEmptyName_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new ColumnReference(null!));
        Assert.Throws<ArgumentException>(() => new ColumnReference(""));
        Assert.Throws<ArgumentException>(() => new ColumnReference("   "));
    }

    [Test]
    public void ColumnReference_Validate_ValidColumn_DoesNotThrow()
    {
        var colRef = new ColumnReference("Name", typeof(string));
        
        Assert.DoesNotThrow(() => colRef.Validate(testSchema));
    }

    [Test]
    public void ColumnReference_Validate_InvalidColumn_ThrowsSchemaValidationException()
    {
        var colRef = new ColumnReference("NonExistent");
        
        var ex = Assert.Throws<SchemaValidationException>(() => colRef.Validate(testSchema));
        Assert.That(ex.Message, Does.Contain("Column 'NonExistent' not found in schema"));
        Assert.That(ex.Message, Does.Contain("Available columns: Name, Age, Salary, IsActive"));
    }

    [Test]
    public void ColumnReference_Validate_WrongType_ThrowsSchemaValidationException()
    {
        var colRef = new ColumnReference("Age", typeof(string)); // Age is int, not string
        
        var ex = Assert.Throws<SchemaValidationException>(() => colRef.Validate(testSchema));
        Assert.That(ex.Message, Does.Contain("Column 'Age' has type Int32 but expected String"));
    }

    [Test]
    public void LiteralExpression_CreatesCorrectLiteral()
    {
        var literal = new LiteralExpression(42);
        
        Assert.That(literal.Value, Is.EqualTo(42));
        Assert.That(literal.ResultType, Is.EqualTo(typeof(int)));
        Assert.That(literal.Name, Is.EqualTo("42"));
        Assert.That(literal.ToString(), Is.EqualTo("42"));
    }

    [Test]
    public void LiteralExpression_NullValue_HandlesCorrectly()
    {
        var literal = new LiteralExpression(null);
        
        Assert.That(literal.Value, Is.Null);
        Assert.That(literal.ResultType, Is.EqualTo(typeof(object)));
        Assert.That(literal.Name, Is.EqualTo("null"));
        Assert.That(literal.ToString(), Is.EqualTo("null"));
    }

    [Test]
    public void LiteralExpression_Validate_AlwaysSucceeds()
    {
        var literal = new LiteralExpression("test");
        
        Assert.DoesNotThrow(() => literal.Validate(testSchema));
    }

    [Test]
    public void BinaryExpression_Addition_CreatesCorrectExpression()
    {
        var left = new ColumnReference("Age");
        var right = new ColumnReference("Salary");
        
        var expr = left + right;
        
        Assert.That(expr, Is.TypeOf<BinaryExpression>());
        var binaryExpr = (BinaryExpression)expr;
        Assert.That(binaryExpr.Operator, Is.EqualTo(BinaryOperator.Add));
        Assert.That(binaryExpr.Left, Is.EqualTo(left));
        Assert.That(binaryExpr.Right, Is.EqualTo(right));
        Assert.That(binaryExpr.Name, Is.EqualTo("(Age + Salary)"));
    }

    [Test]
    public void BinaryExpression_AllArithmeticOperators_CreateCorrectExpressions()
    {
        var left = new ColumnReference("Age");
        var right = new ColumnReference("Salary");
        
        var addExpr = (BinaryExpression)(left + right);
        var subExpr = (BinaryExpression)(left - right);
        var mulExpr = (BinaryExpression)(left * right);
        var divExpr = (BinaryExpression)(left / right);
        
        Assert.That(addExpr.Operator, Is.EqualTo(BinaryOperator.Add));
        Assert.That(subExpr.Operator, Is.EqualTo(BinaryOperator.Subtract));
        Assert.That(mulExpr.Operator, Is.EqualTo(BinaryOperator.Multiply));
        Assert.That(divExpr.Operator, Is.EqualTo(BinaryOperator.Divide));
        
        Assert.That(addExpr.Name, Is.EqualTo("(Age + Salary)"));
        Assert.That(subExpr.Name, Is.EqualTo("(Age - Salary)"));
        Assert.That(mulExpr.Name, Is.EqualTo("(Age * Salary)"));
        Assert.That(divExpr.Name, Is.EqualTo("(Age / Salary)"));
    }

    [Test]
    public void ComparisonExpression_AllOperators_CreateCorrectExpressions()
    {
        var left = new ColumnReference("Age");
        var right = new ColumnReference("Salary");
        
        var gtExpr = (ComparisonExpression)(left > right);
        var ltExpr = (ComparisonExpression)(left < right);
        var gteExpr = (ComparisonExpression)(left >= right);
        var lteExpr = (ComparisonExpression)(left <= right);
        var eqExpr = (ComparisonExpression)(left == right);
        var neqExpr = (ComparisonExpression)(left != right);
        
        Assert.That(gtExpr.Operator, Is.EqualTo(ComparisonOperator.GreaterThan));
        Assert.That(ltExpr.Operator, Is.EqualTo(ComparisonOperator.LessThan));
        Assert.That(gteExpr.Operator, Is.EqualTo(ComparisonOperator.GreaterThanOrEqual));
        Assert.That(lteExpr.Operator, Is.EqualTo(ComparisonOperator.LessThanOrEqual));
        Assert.That(eqExpr.Operator, Is.EqualTo(ComparisonOperator.Equal));
        Assert.That(neqExpr.Operator, Is.EqualTo(ComparisonOperator.NotEqual));
        
        Assert.That(gtExpr.ResultType, Is.EqualTo(typeof(bool)));
        Assert.That(gtExpr.Name, Is.EqualTo("(Age > Salary)"));
    }

    [Test]
    public void ScalarExpression_ArithmeticOperators_CreateCorrectExpressions()
    {
        var column = new ColumnReference("Age");
        
        var addExpr = (ScalarExpression)(column + 5);
        var subExpr = (ScalarExpression)(column - 10);
        var mulExpr = (ScalarExpression)(column * 2);
        var divExpr = (ScalarExpression)(column / 3);
        
        Assert.That(addExpr.Operator, Is.EqualTo(BinaryOperator.Add));
        Assert.That(addExpr.Scalar, Is.EqualTo(5));
        Assert.That(addExpr.Name, Is.EqualTo("(Age + 5)"));
        
        Assert.That(subExpr.Operator, Is.EqualTo(BinaryOperator.Subtract));
        Assert.That(subExpr.Scalar, Is.EqualTo(10));
        
        Assert.That(mulExpr.Operator, Is.EqualTo(BinaryOperator.Multiply));
        Assert.That(mulExpr.Scalar, Is.EqualTo(2));
        
        Assert.That(divExpr.Operator, Is.EqualTo(BinaryOperator.Divide));
        Assert.That(divExpr.Scalar, Is.EqualTo(3));
    }

    [Test]
    public void ScalarComparison_AllOperators_CreateCorrectExpressions()
    {
        var column = new ColumnReference("Age");
        
        var gtExpr = (ComparisonExpression)(column > 30);
        var ltExpr = (ComparisonExpression)(column < 65);
        var gteExpr = (ComparisonExpression)(column >= 18);
        var lteExpr = (ComparisonExpression)(column <= 100);
        var eqExpr = (ComparisonExpression)(column == 25);
        var neqExpr = (ComparisonExpression)(column != 0);
        
        Assert.That(gtExpr.Operator, Is.EqualTo(ComparisonOperator.GreaterThan));
        Assert.That(gtExpr.Right, Is.TypeOf<LiteralExpression>());
        Assert.That(((LiteralExpression)gtExpr.Right).Value, Is.EqualTo(30));
        
        Assert.That(ltExpr.Operator, Is.EqualTo(ComparisonOperator.LessThan));
        Assert.That(gteExpr.Operator, Is.EqualTo(ComparisonOperator.GreaterThanOrEqual));
        Assert.That(lteExpr.Operator, Is.EqualTo(ComparisonOperator.LessThanOrEqual));
        Assert.That(eqExpr.Operator, Is.EqualTo(ComparisonOperator.Equal));
        Assert.That(neqExpr.Operator, Is.EqualTo(ComparisonOperator.NotEqual));
    }

    [Test]
    public void ExpressionComposition_ComplexExpression_BuildsCorrectly()
    {
        var age = new ColumnReference("Age");
        var salary = new ColumnReference("Salary");
        
        // (Age + 5) * Salary > 50000
        var complexExpr = (age + 5) * salary > 50000;
        
        Assert.That(complexExpr, Is.TypeOf<ComparisonExpression>());
        var comparison = (ComparisonExpression)complexExpr;
        Assert.That(comparison.Operator, Is.EqualTo(ComparisonOperator.GreaterThan));
        
        Assert.That(comparison.Left, Is.TypeOf<BinaryExpression>());
        var leftBinary = (BinaryExpression)comparison.Left;
        Assert.That(leftBinary.Operator, Is.EqualTo(BinaryOperator.Multiply));
        
        Assert.That(leftBinary.Left, Is.TypeOf<ScalarExpression>());
        var leftScalar = (ScalarExpression)leftBinary.Left;
        Assert.That(leftScalar.Operator, Is.EqualTo(BinaryOperator.Add));
        Assert.That(leftScalar.Scalar, Is.EqualTo(5));
    }

    [Test]
    public void BinaryExpression_Validate_ValidatesOperands()
    {
        var validLeft = new ColumnReference("Age");
        var invalidRight = new ColumnReference("NonExistent");
        
        var expr = new BinaryExpression(BinaryOperator.Add, validLeft, invalidRight);
        
        var ex = Assert.Throws<SchemaValidationException>(() => expr.Validate(testSchema));
        Assert.That(ex.Message, Does.Contain("NonExistent"));
    }

    [Test]
    public void ComparisonExpression_Validate_ValidatesOperands()
    {
        var validLeft = new ColumnReference("Age");
        var invalidRight = new ColumnReference("NonExistent");
        
        var expr = new ComparisonExpression(ComparisonOperator.Equal, validLeft, invalidRight);
        
        var ex = Assert.Throws<SchemaValidationException>(() => expr.Validate(testSchema));
        Assert.That(ex.Message, Does.Contain("NonExistent"));
    }

    [Test]
    public void ScalarExpression_Validate_ValidatesColumn()
    {
        var invalidColumn = new ColumnReference("NonExistent");
        
        var expr = new ScalarExpression(BinaryOperator.Add, invalidColumn, 5);
        
        var ex = Assert.Throws<SchemaValidationException>(() => expr.Validate(testSchema));
        Assert.That(ex.Message, Does.Contain("NonExistent"));
    }

    [Test]
    public void BinaryExpression_TypePromotion_WorksCorrectly()
    {
        var intCol = new ColumnReference("Age", typeof(int));
        var doubleCol = new ColumnReference("Salary", typeof(double));
        
        var expr = new BinaryExpression(BinaryOperator.Add, intCol, doubleCol);
        
        // Should promote to double (higher precision)
        Assert.That(expr.ResultType, Is.EqualTo(typeof(double)));
    }

    [Test]
    public void BinaryExpression_SameTypes_PreservesType()
    {
        var intCol1 = new ColumnReference("Age", typeof(int));
        var intCol2 = new ColumnReference("Years", typeof(int));
        
        var expr = new BinaryExpression(BinaryOperator.Add, intCol1, intCol2);
        
        Assert.That(expr.ResultType, Is.EqualTo(typeof(int)));
    }

    [Test]
    public void BinaryExpression_NonNumericTypes_ReturnsObjectType()
    {
        var stringCol = new ColumnReference("Name", typeof(string));
        var boolCol = new ColumnReference("IsActive", typeof(bool));
        
        var expr = new BinaryExpression(BinaryOperator.Add, stringCol, boolCol);
        
        Assert.That(expr.ResultType, Is.EqualTo(typeof(object)));
    }
}