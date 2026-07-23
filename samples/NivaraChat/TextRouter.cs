using Microsoft.Agents.AI.Workflows;

namespace NivaraChat;

internal sealed class TextRouter : Executor<string, string>
{
    public TextRouter() : base("TextRouter") { }

    public override ValueTask<string> HandleAsync(string text, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        return ValueTask.FromResult(text);
    }
}
