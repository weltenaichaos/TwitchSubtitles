using System.Net.WebSockets;
using System.Text;
using Whisper.net;
using Whisper.net.Ggml;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();
app.UseWebSockets();

string modelPath = "ggml-tiny.bin";
if (!File.Exists(modelPath))
{
    Console.WriteLine("Model not found. Downloading ggml-tiny.bin...");
    using var httpClient = new HttpClient();
    var response = await httpClient.GetAsync("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin");
    response.EnsureSuccessStatusCode();
    await using var fs = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.None);
    await response.Content.CopyToAsync(fs);
    Console.WriteLine("Model downloaded.");
}

// Initialize Whisper factory and builder once
using var whisperFactory = WhisperFactory.FromPath(modelPath);

app.Map("/listen", async context =>
{
    if (context.WebSockets.IsWebSocketRequest)
    {
        using var webSocket = await context.WebSockets.AcceptWebSocketAsync();

        using var processor = whisperFactory.CreateBuilder()
            .WithLanguage("de")
            .WithTranslate() // Translate to English
            .Build();

        var buffer = new byte[1024 * 4];
        var receiveResult = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);

        var audioData = new List<float>();
        var remainingBytes = new List<byte>();

        while (!receiveResult.CloseStatus.HasValue)
        {
            var totalBytes = remainingBytes.Count + receiveResult.Count;
            var completeFloatsCount = totalBytes / 4;
            var bytesToKeep = totalBytes % 4;

            var newFloatArray = new float[completeFloatsCount];
            var byteBuffer = new byte[totalBytes];

            // Copy remaining from last time
            if (remainingBytes.Count > 0)
            {
                remainingBytes.CopyTo(byteBuffer, 0);
            }

            // Copy new bytes
            Buffer.BlockCopy(buffer, 0, byteBuffer, remainingBytes.Count, receiveResult.Count);

            // Convert to floats (efficiently)
            Buffer.BlockCopy(byteBuffer, 0, newFloatArray, 0, completeFloatsCount * 4);
            audioData.AddRange(newFloatArray);

            // Store remaining incomplete bytes
            remainingBytes.Clear();
            for (int i = totalBytes - bytesToKeep; i < totalBytes; i++)
            {
                remainingBytes.Add(byteBuffer[i]);
            }

            // Process audio every ~2 seconds (16kHz * 2 = 32000 floats)
            if (audioData.Count >= 32000)
            {
                var audioToProcess = audioData.ToArray();
                audioData.Clear(); // Clear early to not block incoming audio, though process is sync here

                var results = processor.ProcessAsync(audioToProcess);

                var transcription = new StringBuilder();
                await foreach (var result in results)
                {
                    transcription.Append(result.Text);
                }

                if (transcription.Length > 0)
                {
                    var bytes = Encoding.UTF8.GetBytes(transcription.ToString());
                    await webSocket.SendAsync(new ArraySegment<byte>(bytes, 0, bytes.Length),
                        WebSocketMessageType.Text, true, CancellationToken.None);
                }
            }

            receiveResult = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
        }

        await webSocket.CloseAsync(
            receiveResult.CloseStatus.Value,
            receiveResult.CloseStatusDescription,
            CancellationToken.None);
    }
    else
    {
        context.Response.StatusCode = StatusCodes.Status400BadRequest;
    }
});

app.Run();
