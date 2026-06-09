using Nivara.Exceptions;
using Nivara.Helpers;

namespace Nivara.Expressions;

/// <summary>
/// Binary operators for expressions
/// </summary>
public enum BinaryOperator
{
    Add,
    Subtract,
    Multiply,
    Divide,
    And,
    Or
}

/// <summary>
/// Comparison operators for expressions
/// </summary>
public enum ComparisonOperator
{
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual
}

/// <summary>
/// Base class for column expressions used in query operations.
/// Provides the foundation for building composable query expressions.
/// </summary>
public abstract class ColumnExpression
{
    /// <summary>
    /// Gets the result type of this expression
    /// </summary>
    public virtual Type ResultType { get; protected set; } = typeof(object);

    /// <summary>
    /// Gets the name of this expression for display purposes
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Validates this expression against the provided schema
    /// </summary>
    /// <param name="schema">The schema to validate against</param>
    /// <exception cref="SchemaValidationException">Thrown when the expression is invalid for the schema</exception>
    public abstract void Validate(Schema schema);

    /// <summary>
    /// Addition operator for column expressions
    /// </summary>
    /// <param name="left">The left operand</param>
    /// <param name="right">The right operand</param>
    /// <returns>A binary expression representing addition</returns>
    public static ColumnExpression operator +(ColumnExpression left, ColumnExpression right)
    {
        return new BinaryExpression(BinaryOperator.Add, left, right);
    }

    /// <summary>
    /// Subtraction operator for column expressions
    /// </summary>
    /// <param name="left">The left operand</param>
    /// <param name="right">The right operand</param>
    /// <returns>A binary expression representing subtraction</returns>
    public static ColumnExpression operator -(ColumnExpression left, ColumnExpression right)
    {
        return new BinaryExpression(BinaryOperator.Subtract, left, right);
    }

    /// <summary>
    /// Multiplication operator for column expressions
    /// </summary>
    /// <param name="left">The left operand</param>
    /// <param name="right">The right operand</param>
    /// <returns>A binary expression representing multiplication</returns>
    public static ColumnExpression operator *(ColumnExpression left, ColumnExpression right)
    {
        return new BinaryExpression(BinaryOperator.Multiply, left, right);
    }

    /// <summary>
    /// Division operator for column expressions
    /// </summary>
    /// <param name="left">The left operand</param>
    /// <param name="right">The right operand</param>
    /// <returns>A binary expression representing division</returns>
    public static ColumnExpression operator /(ColumnExpression left, ColumnExpression right)
    {
        return new BinaryExpression(BinaryOperator.Divide, left, right);
    }

    /// <summary>
    /// Greater than operator for column expressions
    /// </summary>
    /// <param name="left">The left operand</param>
    /// <param name="right">The right operand</param>
    /// <returns>A comparison expression representing greater than</returns>
    public static ColumnExpression operator >(ColumnExpression left, ColumnExpression right)
    {
        return new ComparisonExpression(ComparisonOperator.GreaterThan, left, right);
    }

    /// <summary>
    /// Less than operator for column expressions
    /// </summary>
    /// <param name="left">The left operand</param>
    /// <param name="right">The right operand</param>
    /// <returns>A comparison expression representing less than</returns>
    public static ColumnExpression operator <(ColumnExpression left, ColumnExpression right)
    {
        return new ComparisonExpression(ComparisonOperator.LessThan, left, right);
    }

    /// <summary>
    /// Greater than or equal operator for column expressions
    /// </summary>
    /// <param name="left">The left operand</param>
    /// <param name="right">The right operand</param>
    /// <returns>A comparison expression representing greater than or equal</returns>
    public static ColumnExpression operator >=(ColumnExpression left, ColumnExpression right)
    {
        return new ComparisonExpression(ComparisonOperator.GreaterThanOrEqual, left, right);
    }

    /// <summary>
    /// Less than or equal operator for column expressions
    /// </summary>
    /// <param name="left">The left operand</param>
    /// <param name="right">The right operand</param>
    /// <returns>A comparison expression representing less than or equal</returns>
    public static ColumnExpression operator <=(ColumnExpression left, ColumnExpression right)
    {
        return new ComparisonExpression(ComparisonOperator.LessThanOrEqual, left, right);
    }

    /// <summary>
    /// Equality operator for column expressions
    /// </summary>
    /// <param name="left">The left operand</param>
    /// <param name="right">The right operand</param>
    /// <returns>A comparison expression representing equality</returns>
    public static ColumnExpression operator ==(ColumnExpression left, ColumnExpression right)
    {
        return new ComparisonExpression(ComparisonOperator.Equal, left, right);
    }

    /// <summary>
    /// Inequality operator for column expressions
    /// </summary>
    /// <param name="left">The left operand</param>
    /// <param name="right">The right operand</param>
    /// <returns>A comparison expression representing inequality</returns>
    public static ColumnExpression operator !=(ColumnExpression left, ColumnExpression right)
    {
        return new ComparisonExpression(ComparisonOperator.NotEqual, left, right);
    }

    /// <summary>
    /// Scalar multiplication operator
    /// </summary>
    /// <param name="left">The column expression</param>
    /// <param name="scalar">The scalar value</param>
    /// <returns>A scalar expression representing multiplication</returns>
    public static ColumnExpression operator *(ColumnExpression left, object scalar)
    {
        return new ScalarExpression(BinaryOperator.Multiply, left, scalar);
    }

    /// <summary>
    /// Scalar addition operator
    /// </summary>
    /// <param name="left">The column expression</param>
    /// <param name="scalar">The scalar value</param>
    /// <returns>A scalar expression representing addition</returns>
    public static ColumnExpression operator +(ColumnExpression left, object scalar)
    {
        return new ScalarExpression(BinaryOperator.Add, left, scalar);
    }

    /// <summary>
    /// Scalar subtraction operator
    /// </summary>
    /// <param name="left">The column expression</param>
    /// <param name="scalar">The scalar value</param>
    /// <returns>A scalar expression representing subtraction</returns>
    public static ColumnExpression operator -(ColumnExpression left, object scalar)
    {
        return new ScalarExpression(BinaryOperator.Subtract, left, scalar);
    }

    /// <summary>
    /// Scalar division operator
    /// </summary>
    /// <param name="left">The column expression</param>
    /// <param name="scalar">The scalar value</param>
    /// <returns>A scalar expression representing division</returns>
    public static ColumnExpression operator /(ColumnExpression left, object scalar)
    {
        return new ScalarExpression(BinaryOperator.Divide, left, scalar);
    }

    /// <summary>
    /// Scalar comparison operator (greater than)
    /// </summary>
    /// <param name="left">The column expression</param>
    /// <param name="value">The value to compare against</param>
    /// <returns>A comparison expression</returns>
    public static ColumnExpression operator >(ColumnExpression left, object value)
    {
        return new ComparisonExpression(ComparisonOperator.GreaterThan, left, new LiteralExpression(value));
    }

    /// <summary>
    /// Scalar comparison operator (less than)
    /// </summary>
    /// <param name="left">The column expression</param>
    /// <param name="value">The value to compare against</param>
    /// <returns>A comparison expression</returns>
    public static ColumnExpression operator <(ColumnExpression left, object value)
    {
        return new ComparisonExpression(ComparisonOperator.LessThan, left, new LiteralExpression(value));
    }

    /// <summary>
    /// Scalar comparison operator (equality)
    /// </summary>
    /// <param name="left">The column expression</param>
    /// <param name="value">The value to compare against</param>
    /// <returns>A comparison expression</returns>
    public static ColumnExpression operator ==(ColumnExpression left, object value)
    {
        return new ComparisonExpression(ComparisonOperator.Equal, left, new LiteralExpression(value));
    }

    /// <summary>
    /// Scalar comparison operator (inequality)
    /// </summary>
    /// <param name="left">The column expression</param>
    /// <param name="value">The value to compare against</param>
    /// <returns>A comparison expression</returns>
    public static ColumnExpression operator !=(ColumnExpression left, object value)
    {
        return new ComparisonExpression(ComparisonOperator.NotEqual, left, new LiteralExpression(value));
    }

    /// <summary>
    /// Scalar comparison operator (greater than or equal)
    /// </summary>
    /// <param name="left">The column expression</param>
    /// <param name="value">The value to compare against</param>
    /// <returns>A comparison expression</returns>
    public static ColumnExpression operator >=(ColumnExpression left, object value)
    {
        return new ComparisonExpression(ComparisonOperator.GreaterThanOrEqual, left, new LiteralExpression(value));
    }

    /// <summary>
    /// Scalar comparison operator (less than or equal)
    /// </summary>
    /// <param name="left">The column expression</param>
    /// <param name="value">The value to compare against</param>
    /// <returns>A comparison expression</returns>
    public static ColumnExpression operator <=(ColumnExpression left, object value)
    {
        return new ComparisonExpression(ComparisonOperator.LessThanOrEqual, left, new LiteralExpression(value));
    }

    /// <summary>
    /// Required override for equality operator
    /// </summary>
    /// <param name="obj">The object to compare</param>
    /// <returns>True if equal, false otherwise</returns>
    public override bool Equals(object? obj)
    {
        return ReferenceEquals(this, obj);
    }

    /// <summary>
    /// Required override for equality operator
    /// </summary>
    /// <returns>Hash code for this expression</returns>
    public override int GetHashCode()
    {
        return base.GetHashCode();
    }

    /// <summary>
    /// Returns a string representation of this expression
    /// </summary>
    /// <returns>A string representation</returns>
    public override string ToString()
    {
        return Name;
    }
}

/// <summary>
/// Represents a reference to a column by name
/// </summary>
public sealed class ColumnReference : ColumnExpression
{
    /// <summary>
    /// Initializes a new instance of ColumnReference
    /// </summary>
    /// <param name="columnName">The name of the column</param>
    /// <param name="resultType">The expected result type (optional)</param>
    public ColumnReference(string columnName, Type? resultType = null)
    {
        if (string.IsNullOrWhiteSpace(columnName))
            throw new ArgumentException("Column name cannot be null or whitespace", nameof(columnName));

        ColumnName = columnName;
        ResultType = resultType ?? typeof(object);
    }

    /// <summary>
    /// Gets the name of the referenced column
    /// </summary>
    public string ColumnName { get; }

    /// <inheritdoc />
    public override Type ResultType { get; protected set; }

    /// <inheritdoc />
    public override string Name => ColumnName;

    /// <inheritdoc />
    public override void Validate(Schema schema)
    {
        if (!schema.HasColumn(ColumnName))
        {
            var availableColumns = string.Join(", ", schema.ColumnNames);
            throw new SchemaValidationException($"Column '{ColumnName}' not found in schema. Available columns: {availableColumns}");
        }

        var actualType = schema.GetColumnType(ColumnName);

        // If ResultType was not explicitly set (i.e., it's object), update it to the actual schema type
        if (ResultType == typeof(object))
        {
            ResultType = actualType;
        }
        else if (ResultType != actualType)
        {
            throw new SchemaValidationException($"Column '{ColumnName}' has type {actualType.Name} but expected {ResultType.Name}");
        }
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"Col({ColumnName})";
    }
}

/// <summary>
/// Represents a literal value in an expression
/// </summary>
public sealed class LiteralExpression : ColumnExpression
{
    /// <summary>
    /// Initializes a new instance of LiteralExpression
    /// </summary>
    /// <param name="value">The literal value</param>
    public LiteralExpression(object? value)
    {
        Value = value;
        ResultType = value?.GetType() ?? typeof(object);
    }

    /// <summary>
    /// Gets the literal value
    /// </summary>
    public object? Value { get; }

    /// <inheritdoc />
    public override Type ResultType { get; protected set; }

    /// <inheritdoc />
    public override string Name => Value?.ToString() ?? "null";

    /// <inheritdoc />
    public override void Validate(Schema schema)
    {
        // Literals are always valid
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return Value?.ToString() ?? "null";
    }
}

/// <summary>
/// Represents a binary operation between two expressions
/// </summary>
public sealed class BinaryExpression : ColumnExpression
{
    /// <summary>
    /// Initializes a new instance of BinaryExpression
    /// </summary>
    /// <param name="operator">The binary operator</param>
    /// <param name="left">The left operand</param>
    /// <param name="right">The right operand</param>
    public BinaryExpression(BinaryOperator @operator, ColumnExpression left, ColumnExpression right)
    {
        Operator = @operator;
        Left = left ?? throw new ArgumentNullException(nameof(left));
        Right = right ?? throw new ArgumentNullException(nameof(right));

        // Determine result type based on operands
        ResultType = DetermineResultType(left.ResultType, right.ResultType);
    }

    /// <summary>
    /// Gets the binary operator
    /// </summary>
    public BinaryOperator Operator { get; }

    /// <summary>
    /// Gets the left operand
    /// </summary>
    public ColumnExpression Left { get; }

    /// <summary>
    /// Gets the right operand
    /// </summary>
    public ColumnExpression Right { get; }

    /// <inheritdoc />
    public override Type ResultType { get; protected set; }

    /// <inheritdoc />
    public override string Name => $"({Left.Name} {GetOperatorSymbol(Operator)} {Right.Name})";

    /// <inheritdoc />
    public override void Validate(Schema schema)
    {
        Left.Validate(schema);
        Right.Validate(schema);

        // Validate type compatibility for arithmetic operations
        if (Operator == BinaryOperator.Add || Operator == BinaryOperator.Subtract ||
            Operator == BinaryOperator.Multiply || Operator == BinaryOperator.Divide)
        {
            if (!TypeCompatibilityValidator.AreArithmeticCompatible(Left.ResultType, Right.ResultType))
            {
                throw new SchemaValidationException(
                    $"Binary {Operator} operation requires arithmetic-compatible types. " +
                    $"Left operand type: {Left.ResultType.Name}, Right operand type: {Right.ResultType.Name}");
            }
        }
    }

    private static Type DetermineResultType(Type leftType, Type rightType)
    {
        // Simple type promotion rules
        if (leftType == rightType)
            return leftType;

        // Numeric type promotion
        var numericTypes = new[] { typeof(double), typeof(float), typeof(long), typeof(int), typeof(short), typeof(byte) };

        var leftIndex = Array.IndexOf(numericTypes, leftType);
        var rightIndex = Array.IndexOf(numericTypes, rightType);

        if (leftIndex >= 0 && rightIndex >= 0)
            return numericTypes[Math.Min(leftIndex, rightIndex)];

        return typeof(object);
    }

    private static string GetOperatorSymbol(BinaryOperator op)
    {
        return op switch
        {
            BinaryOperator.Add => "+",
            BinaryOperator.Subtract => "-",
            BinaryOperator.Multiply => "*",
            BinaryOperator.Divide => "/",
            BinaryOperator.And => "&&",
            BinaryOperator.Or => "||",
            _ => op.ToString()
        };
    }
}

/// <summary>
/// Represents a comparison operation between two expressions
/// </summary>
public sealed class ComparisonExpression : ColumnExpression
{
    /// <summary>
    /// Initializes a new instance of ComparisonExpression
    /// </summary>
    /// <param name="operator">The comparison operator</param>
    /// <param name="left">The left operand</param>
    /// <param name="right">The right operand</param>
    public ComparisonExpression(ComparisonOperator @operator, ColumnExpression left, ColumnExpression right)
    {
        Operator = @operator;
        Left = left ?? throw new ArgumentNullException(nameof(left));
        Right = right ?? throw new ArgumentNullException(nameof(right));
    }

    /// <summary>
    /// Gets the comparison operator
    /// </summary>
    public ComparisonOperator Operator { get; }

    /// <summary>
    /// Gets the left operand
    /// </summary>
    public ColumnExpression Left { get; }

    /// <summary>
    /// Gets the right operand
    /// </summary>
    public ColumnExpression Right { get; }

    /// <inheritdoc />
    public override Type ResultType => typeof(bool);

    /// <inheritdoc />
    public override string Name => $"({Left.Name} {GetOperatorSymbol(Operator)} {Right.Name})";

    /// <inheritdoc />
    public override void Validate(Schema schema)
    {
        // Validate left first so we know its resolved type
        Left.Validate(schema);

        // Validate right operand normally (literals are no-ops)
        Right.Validate(schema);

        // Validate type compatibility for comparison operations
        bool compatible = TypeCompatibilityValidator.AreComparisonCompatible(Left.ResultType, Right.ResultType);

        // If not compatible, allow literal coercion where reasonable
        if (!compatible)
        {
            if (Right is LiteralExpression lit && Left.ResultType != typeof(object) && lit.Value != null)
            {
                try
                {
                    Convert.ChangeType(lit.Value, Left.ResultType);
                    compatible = true;
                }
                catch
                {
                    // conversion failed, leave compatible=false
                }
            }

            if (!compatible && Left is LiteralExpression litLeft && Right.ResultType != typeof(object) && litLeft.Value != null)
            {
                try
                {
                    Convert.ChangeType(litLeft.Value, Right.ResultType);
                    compatible = true;
                }
                catch
                {
                    // conversion failed
                }
            }
        }

        if (!compatible)
        {
            throw new SchemaValidationException(
                $"Comparison {Operator} operation requires compatible types. Left operand type: {Left.ResultType.Name}, Right operand type: {Right.ResultType.Name}");
        }

        // Validate that both operand types support comparison
        if (!TypeCompatibilityValidator.SupportsComparison(Left.ResultType))
        {
            throw new SchemaValidationException(
                $"Comparison {Operator} operation: Left operand type {Left.ResultType.Name} does not support comparison operations");
        }

        if (!TypeCompatibilityValidator.SupportsComparison(Right.ResultType))
        {
            throw new SchemaValidationException(
                $"Comparison {Operator} operation: Right operand type {Right.ResultType.Name} does not support comparison operations");
        }
    }

    private static string GetOperatorSymbol(ComparisonOperator op)
    {
        return op switch
        {
            ComparisonOperator.Equal => "==",
            ComparisonOperator.NotEqual => "!=",
            ComparisonOperator.GreaterThan => ">",
            ComparisonOperator.LessThan => "<",
            ComparisonOperator.GreaterThanOrEqual => ">=",
            ComparisonOperator.LessThanOrEqual => "<=",
            _ => op.ToString()
        };
    }
}

/// <summary>
/// Represents a scalar operation (column with scalar value)
/// </summary>
public sealed class ScalarExpression : ColumnExpression
{
    /// <summary>
    /// Initializes a new instance of ScalarExpression
    /// </summary>
    /// <param name="operator">The binary operator</param>
    /// <param name="column">The column expression</param>
    /// <param name="scalar">The scalar value</param>
    public ScalarExpression(BinaryOperator @operator, ColumnExpression column, object scalar)
    {
        Operator = @operator;
        Column = column ?? throw new ArgumentNullException(nameof(column));
        Scalar = scalar;
        ResultType = column.ResultType; // Result type is same as column type
    }

    /// <summary>
    /// Gets the binary operator
    /// </summary>
    public BinaryOperator Operator { get; }

    /// <summary>
    /// Gets the column expression
    /// </summary>
    public ColumnExpression Column { get; }

    /// <summary>
    /// Gets the scalar value
    /// </summary>
    public object Scalar { get; }

    /// <inheritdoc />
    public override Type ResultType { get; protected set; }

    /// <inheritdoc />
    public override string Name => $"({Column.Name} {GetOperatorSymbol(Operator)} {Scalar})";

    /// <inheritdoc />
    public override void Validate(Schema schema)
    {
        Column.Validate(schema);

        // Validate type compatibility for scalar operations
        if (Scalar != null && (Operator == BinaryOperator.Add || Operator == BinaryOperator.Subtract ||
            Operator == BinaryOperator.Multiply || Operator == BinaryOperator.Divide))
        {
            if (!TypeCompatibilityValidator.AreArithmeticCompatible(Column.ResultType, Scalar.GetType()))
            {
                throw new SchemaValidationException(
                    $"Scalar {Operator} operation requires arithmetic-compatible types. " +
                    $"Column type: {Column.ResultType.Name}, Scalar type: {Scalar.GetType().Name}");
            }
        }
    }

    private static string GetOperatorSymbol(BinaryOperator op)
    {
        return op switch
        {
            BinaryOperator.Add => "+",
            BinaryOperator.Subtract => "-",
            BinaryOperator.Multiply => "*",
            BinaryOperator.Divide => "/",
            BinaryOperator.And => "&&",
            BinaryOperator.Or => "||",
            _ => op.ToString()
        };
    }
}

/// <summary>
/// Global function for creating column references
/// </summary>
public static class ColumnExpressions
{
    /// <summary>
    /// Creates a column reference expression
    /// </summary>
    /// <param name="name">The name of the column</param>
    /// <returns>A column reference expression</returns>
    public static ColumnExpression Col(string name)
    {
        return new ColumnReference(name);
    }

    /// <summary>
    /// Creates a strongly-typed column reference expression
    /// </summary>
    /// <typeparam name="T">The expected type of the column</typeparam>
    /// <param name="name">The name of the column</param>
    /// <returns>A column reference expression</returns>
    public static ColumnExpression Col<T>(string name)
    {
        return new ColumnReference(name, typeof(T));
    }

    /// <summary>
    /// Creates a literal expression
    /// </summary>
    /// <param name="value">The literal value</param>
    /// <returns>A literal expression</returns>
    public static ColumnExpression Lit(object? value)
    {
        return new LiteralExpression(value);
    }
}
