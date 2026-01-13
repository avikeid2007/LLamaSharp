using LLama.Common;
using LLama.Transformers;

namespace LLama.Unittest.Transformers;

public class ModelHistoryTransformTests
{
    [Fact]
    public void Llama3_HistoryToText_SingleUserMessage()
    {
        var transformer = new Llama3HistoryTransform(addAssistantHeader: true);
        var history = new ChatHistory
        {
            Messages = [new ChatHistory.Message(AuthorRole.User, "Hello")]
        };

        var result = transformer.HistoryToText(history);

        Assert.Contains("user", result);
        Assert.Contains("Hello", result);
        Assert.Contains("assistant", result);
    }

    [Fact]
    public void Llama3_Clone_ReturnsNewInstance()
    {
        var transformer = new Llama3HistoryTransform();
        var clone = transformer.Clone();
        Assert.NotSame(transformer, clone);
    }

    [Fact]
    public void Phi_HistoryToText_SingleUserMessage()
    {
        var transformer = new PhiHistoryTransform(addAssistantHeader: true);
        var history = new ChatHistory
        {
            Messages = [new ChatHistory.Message(AuthorRole.User, "Hello")]
        };

        var result = transformer.HistoryToText(history);

        Assert.Contains("user", result);
        Assert.Contains("Hello", result);
    }

    [Fact]
    public void Phi_Clone_ReturnsNewInstance()
    {
        var transformer = new PhiHistoryTransform();
        var clone = transformer.Clone();
        Assert.NotSame(transformer, clone);
    }

    [Fact]
    public void Mistral_HistoryToText_SingleUserMessage()
    {
        var transformer = new MistralHistoryTransform();
        var history = new ChatHistory
        {
            Messages = [new ChatHistory.Message(AuthorRole.User, "Hello")]
        };

        var result = transformer.HistoryToText(history);

        Assert.Contains("[INST]", result);
        Assert.Contains("Hello", result);
        Assert.Contains("[/INST]", result);
    }

    [Fact]
    public void Mistral_Clone_ReturnsNewInstance()
    {
        var transformer = new MistralHistoryTransform();
        var clone = transformer.Clone();
        Assert.NotSame(transformer, clone);
    }

    [Fact]
    public void Qwen_HistoryToText_SingleUserMessage()
    {
        var transformer = new QwenHistoryTransform(addAssistantHeader: true);
        var history = new ChatHistory
        {
            Messages = [new ChatHistory.Message(AuthorRole.User, "Hello")]
        };

        var result = transformer.HistoryToText(history);

        Assert.Contains("user", result);
        Assert.Contains("Hello", result);
    }

    [Fact]
    public void Qwen_Clone_ReturnsNewInstance()
    {
        var transformer = new QwenHistoryTransform();
        var clone = transformer.Clone();
        Assert.NotSame(transformer, clone);
    }

    [Fact]
    public void AllTransformers_EmptyHistory_ReturnsEmpty()
    {
        var llama3 = new Llama3HistoryTransform();
        var phi = new PhiHistoryTransform();
        var mistral = new MistralHistoryTransform();
        var qwen = new QwenHistoryTransform();
        var empty = new ChatHistory();

        Assert.Equal(string.Empty, llama3.HistoryToText(empty));
        Assert.Equal(string.Empty, phi.HistoryToText(empty));
        Assert.Equal(string.Empty, mistral.HistoryToText(empty));
        Assert.Equal(string.Empty, qwen.HistoryToText(empty));
    }
}
