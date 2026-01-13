using LLama.Abstractions;
using LLama.Common;
using System.Text;

namespace LLama.Transformers;

/// <summary>
/// A history transformer for Meta LLaMA 3.x models (3.1, 3.2, 3.3).
/// Uses the official LLaMA 3 chat format with special header tokens.
/// </summary>
/// <remarks>
/// Format: <![CDATA[
/// <|start_header_id|>system<|end_header_id|>
/// {system_message}<|eot_id|>
/// <|start_header_id|>user<|end_header_id|>
/// {user_message}<|eot_id|>
/// <|start_header_id|>assistant<|end_header_id|>
/// ]]>
/// </remarks>
public class Llama3HistoryTransform : IHistoryTransform
{
    /// <summary>
    /// Gets the name of this transformer.
    /// </summary>
    public string Name => "Llama3";

    private readonly bool _addAssistantHeader;

    /// <summary>
    /// Creates a new instance of the LLaMA 3 history transformer.
    /// </summary>
    /// <param name="addAssistantHeader">Whether to add the assistant header at the end to prompt for a response.</param>
    public Llama3HistoryTransform(bool addAssistantHeader = true)
    {
        _addAssistantHeader = addAssistantHeader;
    }

    /// <inheritdoc/>
    public IHistoryTransform Clone()
    {
        return new Llama3HistoryTransform(_addAssistantHeader);
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

            builder.Append("<|start_header_id|>")
                   .Append(role)
                   .Append("<|end_header_id|>\n\n")
                   .Append(message.Content.Trim())
                   .Append("<|eot_id|>");
        }

        if (_addAssistantHeader)
        {
            builder.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }

        return builder.ToString();
    }

    /// <inheritdoc/>
    public ChatHistory TextToHistory(AuthorRole role, string text)
    {
        return new ChatHistory([new ChatHistory.Message(role, text)]);
    }
}
