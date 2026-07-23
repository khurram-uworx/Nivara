using Microsoft.Extensions.AI;

namespace NivaraChat;

internal sealed class PassthroughTextModel : ITextModel
{
    private readonly IChatClient _client;

    public string Name => "LLM";

    public PassthroughTextModel(IChatClient client)
    {
        _client = client;
    }

    public string Process(string input)
    {
        var response = _client.GetResponseAsync(input).GetAwaiter().GetResult();
        return response.ToString();
    }
}
