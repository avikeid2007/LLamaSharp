using LLama.Common;
using LLama.Sampling;
using LLama.Transformers;

namespace LLama.Examples.Examples;

/// <summary>
/// This sample shows a simple chatbot using Microsoft Phi-3/Phi-4 models.
/// It demonstrates using the PhiHistoryTransform for correct prompt formatting.
/// </summary>
public class Phi4ChatSession
{
    public static async Task Run()
    {
        var modelPath = UserSettings.GetModelPath();
        var parameters = new ModelParams(modelPath)
        {
            GpuLayerCount = 32 // Use more GPU layers for faster inference
        };

        using var model = LLamaWeights.LoadFromFile(parameters);
        using var context = model.CreateContext(parameters);
        var executor = new InteractiveExecutor(context);

        // Start with a fresh chat history for Phi models
        var chatHistory = new ChatHistory();
        chatHistory.AddMessage(AuthorRole.System, "You are a helpful AI assistant.");

        ChatSession session = new(executor, chatHistory);

        // Use llama.cpp's built-in template (phi3 is natively supported)
        session.WithHistoryTransform(new PromptTemplateTransformer(model, withAssistant: true));

        // Add a transformer to eliminate printing the end of turn tokens, llama 3 specifically has an odd LF that gets printed sometimes
        session.WithOutputTransform(new LLamaTransforms.KeywordTextOutputStreamTransform(
            ["User:", "�"],
            redundancyLength: 5));

        var inferenceParams = new InferenceParams
        {
            SamplingPipeline = new DefaultSamplingPipeline
            {
                Temperature = 0.6f
            },

            MaxTokens = 512
        };

        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("The chat session has started (Phi-3/Phi-4 model).");

        // show the prompt
        Console.ForegroundColor = ConsoleColor.Green;
        Console.Write("User> ");
        var userInput = Console.ReadLine() ?? "";

        while (userInput != "exit")
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("Assistant> ");

            // as each token (partial or whole word is streamed back) print it to the console, stream to web client, etc
            await foreach (
                var text
                in session.ChatAsync(
                    new ChatHistory.Message(AuthorRole.User, userInput),
                    inferenceParams))
            {
                Console.ForegroundColor = ConsoleColor.White;
                Console.Write(text);
            }
            Console.WriteLine();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write("User> ");
            userInput = Console.ReadLine() ?? "";
        }
    }
}
