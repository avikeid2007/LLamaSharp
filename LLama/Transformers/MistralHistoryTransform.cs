using LLama.Abstractions;
using LLama.Common;
using System.Text;

namespace LLama.Transformers;

/// <summary>
/// A history transformer for Mistral Instruct models.
/// Uses the Mistral instruction format with [INST] tags.
/// </summary>
/// <remarks>
/// Format:
/// [INST] {system_message}
/// {user_message} [/INST] {assistant_response}
/// </remarks>
public class MistralHistoryTransform : IHistoryTransform
{
    /// <summary>
    /// Gets the name of this transformer.
    /// </summary>
    public string Name => "Mistral";

    /// <inheritdoc/>
    public IHistoryTransform Clone()
    {
        return new MistralHistoryTransform();
    }

    /// <inheritdoc/>
    public string HistoryToText(ChatHistory history)
    {
        if (history.Messages.Count == 0)
            return string.Empty;

        var builder = new StringBuilder(256);
        int i = 0;

        // Handle system message at the beginning
        if (history.Messages.Count > 0 && history.Messages[0].AuthorRole == AuthorRole.System)
        {
            builder.Append("<s>[INST] ")
                   .Append(history.Messages[0].Content.Trim())
                   .Append("\n\n");
            i++;

            // If there's a user message right after system, include it in the same [INST] block
            if (i < history.Messages.Count && history.Messages[i].AuthorRole == AuthorRole.User)
            {
                builder.Append(history.Messages[i].Content.Trim())
                       .Append(" [/INST]");
                i++;
            }
            else
            {
                builder.Append("[/INST]");
            }
        }

        // Handle remaining messages
        for (; i < history.Messages.Count; i++)
        {
            var message = history.Messages[i];

            if (message.AuthorRole == AuthorRole.User)
            {
                builder.Append(" [INST] ")
                       .Append(message.Content.Trim())
                       .Append(" [/INST]");
            }
            else if (message.AuthorRole == AuthorRole.Assistant)
            {
                builder.Append(" ")
                       .Append(message.Content.Trim())
                       .Append("</s>");
            }
        }

        return builder.ToString();
    }

    /// <inheritdoc/>
    public ChatHistory TextToHistory(AuthorRole role, string text)
    {
        return new ChatHistory([new ChatHistory.Message(role, text)]);
    }
}
