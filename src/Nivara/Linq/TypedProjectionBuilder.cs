using Nivara.Exceptions;
using Nivara.Expressions;
using System.Linq.Expressions;
using System.Reflection;

namespace Nivara.Linq;

/// <summary>
/// Builds a select projection (column expressions + output column names) from the body of a typed
/// Select lambda. Only anonymous-type constructions (<see cref="NewExpression"/> with members) and
/// member initializers (<see cref="MemberInitExpression"/>) are supported, per IDEA §6.2.
/// </summary>
internal static class TypedProjectionBuilder
{
    /// <summary>
    /// The result of building a projection: the translated column expressions and their output names.
    /// </summary>
    public readonly record struct Projection(ColumnExpression[] Columns, string[] OutputNames);

    /// <summary>
    /// Builds a projection from an expression body
    /// </summary>
    /// <param name="body">The Select lambda body</param>
    /// <param name="translator">The translator for row-property references</param>
    /// <returns>The column expressions and their output column names</returns>
    /// <exception cref="UnsupportedQueryExpressionException">Thrown when the body is not a supported projection</exception>
    public static Projection Build(Expression body, TypedExpressionTranslator translator)
    {
        ArgumentNullException.ThrowIfNull(body);
        ArgumentNullException.ThrowIfNull(translator);

        switch (body)
        {
            case NewExpression newExpression when newExpression.Members is { Count: > 0 }:
                return BuildAnonymous(newExpression, translator);

            case MemberInitExpression memberInit:
                return BuildMemberInit(memberInit, translator);

            case NewExpression:
                throw new UnsupportedQueryExpressionException(
                    $"Unsupported Select projection '{body}': constructor calls are only supported for anonymous types.");

            default:
                throw new UnsupportedQueryExpressionException(
                    $"Unsupported Select projection '{body}': projections must construct an object " +
                    "(an anonymous type such as new {{ p.City, Total = p.Age + 5 }} or a member initializer).");
        }
    }

    static Projection BuildAnonymous(NewExpression newExpression, TypedExpressionTranslator translator)
    {
        var count = newExpression.Arguments.Count;
        var columns = new ColumnExpression[count];
        var outputNames = new string[count];

        for (int i = 0; i < count; i++)
        {
            var member = newExpression.Members?[i]
                ?? throw new UnsupportedQueryExpressionException(
                    $"Unsupported Select projection member at position {i}: anonymous-type arguments must be named members.");

            columns[i] = translator.Translate(newExpression.Arguments[i]);
            outputNames[i] = member.Name;
        }

        return new Projection(columns, outputNames);
    }

    static Projection BuildMemberInit(MemberInitExpression memberInit, TypedExpressionTranslator translator)
    {
        var columns = new List<ColumnExpression>();
        var outputNames = new List<string>();

        foreach (var binding in memberInit.Bindings)
        {
            if (binding is not MemberAssignment assignment)
                throw new UnsupportedQueryExpressionException(
                    $"Unsupported Select projection member '{binding.Member.Name}': only property assignments are supported.");

            if (assignment.Member is not PropertyInfo property)
                throw new UnsupportedQueryExpressionException(
                    $"Unsupported Select projection member '{assignment.Member.Name}': fields are not supported.");

            columns.Add(translator.Translate(assignment.Expression));
            outputNames.Add(property.Name);
        }

        if (columns.Count == 0)
            throw new UnsupportedQueryExpressionException(
                $"Unsupported Select projection '{memberInit}': the member initializer binds no properties.");

        return new Projection(columns.ToArray(), outputNames.ToArray());
    }
}
