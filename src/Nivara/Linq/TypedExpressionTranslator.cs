using Nivara.Exceptions;
using Nivara.Expressions;
using System.Linq.Expressions;
using System.Reflection;
using NivaraBinaryExpression = Nivara.Expressions.BinaryExpression;
using SystemBinaryExpression = System.Linq.Expressions.BinaryExpression;

namespace Nivara.Linq;

/// <summary>
/// Converts typed LINQ expression trees into the expression engine's <see cref="ColumnExpression"/>
/// model, enforcing the supported/unsupported expression rules (IDEA §6.2) at query-build time.
/// Allowed: property access, constant literals, arithmetic, comparisons, boolean logic, and unary
/// negation. Rejected (with a clear diagnostic): method calls, captured variables/closures, nested
/// property access, array/index access, invocations, ternaries, modulo, string concatenation, and
/// nested lambdas.
/// </summary>
sealed class TypedExpressionTranslator
{
    readonly IReadOnlyDictionary<string, string> propertyToColumn;

    /// <summary>
    /// Initializes a new instance of TypedExpressionTranslator
    /// </summary>
    /// <param name="propertyToColumn">Mapping from row property names to frame column names</param>
    public TypedExpressionTranslator(IReadOnlyDictionary<string, string> propertyToColumn)
    {
        this.propertyToColumn = propertyToColumn ?? throw new ArgumentNullException(nameof(propertyToColumn));
    }

    /// <summary>
    /// Translates an expression tree node into a column expression.
    /// </summary>
    /// <param name="expression">The expression to translate</param>
    /// <returns>The equivalent column expression</returns>
    /// <exception cref="UnsupportedQueryExpressionException">Thrown when the expression is not supported</exception>
    public ColumnExpression Translate(Expression expression)
    {
        ArgumentNullException.ThrowIfNull(expression);

        return expression switch
        {
            MemberExpression member => TranslateMember(member),
            ConstantExpression constant => new LiteralExpression(constant.Value),
            SystemBinaryExpression binary => TranslateBinary(binary),
            UnaryExpression unary => TranslateUnary(unary),
            ParameterExpression _ => throw Unsupported(expression, "a bare row parameter has no column meaning"),
            MethodCallExpression _ => throw Unsupported(expression, "method calls are not supported"),
            NewExpression _ => throw Unsupported(expression, "object construction is only supported in Select projections"),
            MemberInitExpression _ => throw Unsupported(expression, "object construction is only supported in Select projections"),
            ConditionalExpression _ => throw Unsupported(expression, "ternary (?:) expressions are not supported"),
            InvocationExpression _ => throw Unsupported(expression, "invocation expressions are not supported"),
            LambdaExpression _ => throw Unsupported(expression, "nested lambdas are not supported"),
            IndexExpression _ => throw Unsupported(expression, "index access is not supported"),
            NewArrayExpression _ => throw Unsupported(expression, "array access is not supported"),
            _ => throw Unsupported(expression, $"expression node '{expression.NodeType}' is not supported")
        };
    }

    /// <summary>
    /// Translates a member access. A direct property access on the row parameter maps to a column
    /// reference; a member over a constant is a closure capture; any deeper path is nested access.
    /// </summary>
    ColumnExpression TranslateMember(MemberExpression member)
    {
        if (member.Expression is ConstantExpression)
            throw Unsupported(member, "captured variables/closures are not supported; inline the literal value");

        if (member.Expression is not ParameterExpression)
            throw Unsupported(member, "nested property access is not supported; access only top-level properties of the row");

        if (member.Member is not PropertyInfo)
            throw Unsupported(member, "field access is not supported; use properties");

        if (!propertyToColumn.TryGetValue(member.Member.Name, out var columnName))
            throw Unsupported(member, $"property '{member.Member.Name}' does not map to a frame column");

        return new ColumnReference(columnName);
    }

    /// <summary>
    /// Translates a binary expression, mapping arithmetic, boolean-logic, and comparison operators
    /// onto the expression model and rejecting string concatenation, modulo, and coalescing.
    /// </summary>
    ColumnExpression TranslateBinary(SystemBinaryExpression binary)
    {
        if (binary.NodeType == ExpressionType.Add && (binary.Left.Type == typeof(string) || binary.Right.Type == typeof(string)))
            throw Unsupported(binary, "string concatenation is not supported");

        var left = Translate(binary.Left);
        var right = Translate(binary.Right);

        return binary.NodeType switch
        {
            ExpressionType.Add or ExpressionType.AddChecked => new NivaraBinaryExpression(BinaryOperator.Add, left, right),
            ExpressionType.Subtract or ExpressionType.SubtractChecked => new NivaraBinaryExpression(BinaryOperator.Subtract, left, right),
            ExpressionType.Multiply or ExpressionType.MultiplyChecked => new NivaraBinaryExpression(BinaryOperator.Multiply, left, right),
            ExpressionType.Divide => new NivaraBinaryExpression(BinaryOperator.Divide, left, right),
            ExpressionType.AndAlso => new NivaraBinaryExpression(BinaryOperator.And, left, right),
            ExpressionType.OrElse => new NivaraBinaryExpression(BinaryOperator.Or, left, right),
            ExpressionType.Equal => new ComparisonExpression(ComparisonOperator.Equal, left, right),
            ExpressionType.NotEqual => new ComparisonExpression(ComparisonOperator.NotEqual, left, right),
            ExpressionType.GreaterThan => new ComparisonExpression(ComparisonOperator.GreaterThan, left, right),
            ExpressionType.GreaterThanOrEqual => new ComparisonExpression(ComparisonOperator.GreaterThanOrEqual, left, right),
            ExpressionType.LessThan => new ComparisonExpression(ComparisonOperator.LessThan, left, right),
            ExpressionType.LessThanOrEqual => new ComparisonExpression(ComparisonOperator.LessThanOrEqual, left, right),
            ExpressionType.Coalesce => throw Unsupported(binary, "null-coalescing (??) expressions are not supported"),
            ExpressionType.Modulo => throw Unsupported(binary, "modulo (%) arithmetic is not supported"),
            _ => throw Unsupported(binary, $"binary operator '{binary.NodeType}' is not supported")
        };
    }

    /// <summary>
    /// Translates a unary expression, mapping logical negation to <see cref="NotExpression"/> and
    /// unwrapping benign conversions.
    /// </summary>
    ColumnExpression TranslateUnary(UnaryExpression unary)
    {
        return unary.NodeType switch
        {
            ExpressionType.Not => new NotExpression(Translate(unary.Operand)),
            ExpressionType.Convert or ExpressionType.ConvertChecked => Translate(unary.Operand),
            _ => throw Unsupported(unary, $"unary operator '{unary.NodeType}' is not supported")
        };
    }

    static UnsupportedQueryExpressionException Unsupported(Expression expression, string reason)
    {
        return new UnsupportedQueryExpressionException(
            $"Unsupported query expression '{expression}': {reason}.");
    }
}
