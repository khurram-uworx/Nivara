using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using NivaraChat.Shared;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public class OpenAiCompatibleChatClientTests
{
    sealed class FakeHandler : HttpMessageHandler
    {
        public HttpRequestMessage? LastRequest;
        public string LastBody = string.Empty;
        public Func<HttpRequestMessage, HttpResponseMessage> Responder = _ =>
            new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("{\"choices\":[{\"message\":{\"content\":\"hello\"}}]}", Encoding.UTF8, "application/json")
            };

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            LastRequest = request;
            if (request.Content is not null)
                LastBody = request.Content.ReadAsStringAsync(cancellationToken).GetAwaiter().GetResult();
            return Task.FromResult(Responder(request));
        }
    }

    [Test]
    public async Task GetResponseAsync_PostsToChatCompletionsWithBearerAndParsesContent()
    {
        var handler = new FakeHandler();
        using var client = new OpenAiCompatibleChatClient("http://localhost:11434/v1", "m", "secret", "bearer", handler);
        var response = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "hi")], new ChatOptions { Temperature = 0.2f });

        Assert.That(response.Text, Is.EqualTo("hello"));
        Assert.That(handler.LastRequest!.Method, Is.EqualTo(HttpMethod.Post));
        Assert.That(handler.LastRequest.RequestUri!.AbsoluteUri, Is.EqualTo("http://localhost:11434/v1/chat/completions"));
        Assert.That(handler.LastRequest.Headers.Authorization!.Scheme, Is.EqualTo("Bearer"));
        Assert.That(handler.LastRequest.Headers.Authorization.Parameter, Is.EqualTo("secret"));

        using var doc = JsonDocument.Parse(handler.LastBody);
        Assert.That(doc.RootElement.GetProperty("model").GetString(), Is.EqualTo("m"));
        Assert.That(doc.RootElement.GetProperty("messages")[0].GetProperty("role").GetString(), Is.EqualTo("user"));
        Assert.That(doc.RootElement.GetProperty("messages")[0].GetProperty("content").GetString(), Is.EqualTo("hi"));
        Assert.That(doc.RootElement.GetProperty("temperature").GetSingle(), Is.EqualTo(0.2f).Within(1e-6f));
    }

    [Test]
    public async Task GetResponseAsync_ApiKeyAuth_SendsApiKeyHeader()
    {
        var handler = new FakeHandler();
        using var client = new OpenAiCompatibleChatClient("https://r.openai.azure.com/openai/v1", "m", "akey", "api-key", handler);
        await client.GetResponseAsync([new ChatMessage(ChatRole.User, "hi")]);

        Assert.That(handler.LastRequest!.Headers.Contains("api-key"), Is.True);
        Assert.That(handler.LastRequest.Headers.GetValues("api-key").Single(), Is.EqualTo("akey"));
        Assert.That(handler.LastRequest.Headers.Authorization, Is.Null);
    }

    [Test]
    public async Task GetResponseAsync_NonSuccess_ThrowsWithStatusAndBody()
    {
        var handler = new FakeHandler
        {
            Responder = _ => new HttpResponseMessage(HttpStatusCode.Unauthorized)
            {
                ReasonPhrase = "Unauthorized",
                Content = new StringContent("nope", Encoding.UTF8, "application/json")
            }
        };
        using var client = new OpenAiCompatibleChatClient("http://localhost:11434/v1", "m", "bad", "bearer", handler);

        var ex = await Assert.ThrowsAsync<HttpRequestException>(async () =>
            await client.GetResponseAsync([new ChatMessage(ChatRole.User, "hi")]));
        Assert.That(ex!.Message, Does.Contain("401"));
        Assert.That(ex.Message, Does.Contain("nope"));
    }

    [Test]
    public async Task GetResponseAsync_ModelIdOverride_UsesOptionsModel()
    {
        var handler = new FakeHandler();
        using var client = new OpenAiCompatibleChatClient("http://localhost:11434/v1", "default", "k", "bearer", handler);
        await client.GetResponseAsync([new ChatMessage(ChatRole.User, "hi")], new ChatOptions { ModelId = "override" });

        using var doc = JsonDocument.Parse(handler.LastBody);
        Assert.That(doc.RootElement.GetProperty("model").GetString(), Is.EqualTo("override"));
    }

    [Test]
    public async Task GetResponseAsync_BaseUrlAlreadyChatCompletions_UsedAsIs()
    {
        var handler = new FakeHandler();
        using var client = new OpenAiCompatibleChatClient("http://localhost:11434/v1/chat/completions", "m", "k", "bearer", handler);
        await client.GetResponseAsync([new ChatMessage(ChatRole.User, "hi")]);

        Assert.That(handler.LastRequest!.RequestUri!.AbsoluteUri, Is.EqualTo("http://localhost:11434/v1/chat/completions"));
    }

    [Test]
    public async Task GetStreamingResponseAsync_YieldsDeltas()
    {
        var sse = "data: {\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\ndata: [DONE]\n";
        var handler = new FakeHandler
        {
            Responder = _ => new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(sse, Encoding.UTF8, "text/event-stream")
            }
        };
        using var client = new OpenAiCompatibleChatClient("http://localhost:11434/v1", "m", "k", "bearer", handler);

        var collected = new StringBuilder();
        await foreach (var update in client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, "hi")]))
            collected.Append(update.Text);

        Assert.That(collected.ToString(), Is.EqualTo("hello"));
    }
}
