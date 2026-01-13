using LLama.Abstractions;
using LLama.Common;
using System.Text;

namespace LLama.Transformers;

/// <summary>
/// A history transformer for Microsoft Phi models (Phi-3, Phi-4).
/// Uses the Phi-specific chat format with role tokens.
/// </summary>
/// <remarks>
/// Format:
/// &lt;|system|&gt;
/// {system_message}&lt;|end|&gt;
/// &lt;|user|&gt;
/// {user_message}&lt;|end|&gt;
/// &lt;|assistant|&gt;
/// </remarks>
public class PhiHistoryTransform : IHistoryTransform
{
    /// <summary>
    /// Gets the name of this transformer.
    /// </summary>
    public string Name => "Phi";

    private readonly bool _addAssistantHeader;

    /// <summary>
    /// Creates a new instance of the Phi history transformer.
    /// </summary>
    /// <param name="addAssistantHeader">Whether to add the assistant header at the end to prompt for a response.</param>
    public PhiHistoryTransform(bool addAssistantHeader = true)
    {
        _addAssistantHeader = addAssistantHeader;
    }

    /// <inheritdoc/>
    public IHistoryTransform Clone()
    {
        return new PhiHistoryTransform(_addAssistantHeader);
    }

    /// <inheritdoc/>
    public string HistoryToText(ChatHistory history)
    {
        if (history.Messages.Count == 0)
            return string.Empty;

        var builder = new StringBuilder(256);

        foreach (var message in history.Messages)
        {
            var role = message.AuthorRole switch
            {
                AuthorRole.System => "system",
                AuthorRole.User => "user",
                AuthorRole.Assistant => "assistant",
                _ => "user"
            };

            builder.Append("<|")
                   .Append(role)
                   .Append("|>\n")
                   .Append(message.Content.Trim())
                   .Append("<|end|>\n");
        }

        if (_addAssistantHeader)
        {
            builder.Append("<|assistant|>\n");
        }

        return builder.ToString();
    }

    /// <inheritdoc/>
    public ChatHistory TextToHistory(AuthorRole role, string text)
    {
        return new ChatHistory([new ChatHistory.Message(role, text)]);
    }
}
