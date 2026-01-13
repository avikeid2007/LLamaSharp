using LLama.Abstractions;
using LLama.Common;
using System.Text;

namespace LLama.Transformers;

/// <summary>
/// A history transformer for Qwen models (Qwen 1.5, 2, 3).
/// Uses the ChatML format with im_start/im_end tokens.
/// </summary>
/// <remarks>
/// Format:
/// &lt;|im_start|&gt;system
/// {system_message}&lt;|im_end|&gt;
/// &lt;|im_start|&gt;user
/// {user_message}&lt;|im_end|&gt;
/// &lt;|im_start|&gt;assistant
/// </remarks>
public class QwenHistoryTransform : IHistoryTransform
{
    /// <summary>
    /// Gets the name of this transformer.
    /// </summary>
    public string Name => "Qwen";

    private readonly bool _addAssistantHeader;

    /// <summary>
    /// Creates a new instance of the Qwen history transformer.
    /// </summary>
    /// <param name="addAssistantHeader">Whether to add the assistant header at the end to prompt for a response.</param>
    public QwenHistoryTransform(bool addAssistantHeader = true)
    {
        _addAssistantHeader = addAssistantHeader;
    }

    /// <inheritdoc/>
    public IHistoryTransform Clone()
    {
        return new QwenHistoryTransform(_addAssistantHeader);
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

            builder.Append("<|im_start|>")
                   .Append(role)
                   .Append("\n")
                   .Append(message.Content.Trim())
                   .Append("<|im_end|>\n");
        }

        if (_addAssistantHeader)
        {
            builder.Append("<|im_start|>assistant\n");
        }

        return builder.ToString();
    }

    /// <inheritdoc/>
    public ChatHistory TextToHistory(AuthorRole role, string text)
    {
        return new ChatHistory([new ChatHistory.Message(role, text)]);
    }
}
