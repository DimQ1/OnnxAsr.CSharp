using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxAsr;

/// <summary>
/// Utility methods for audio processing.
/// </summary>
public static class AudioUtils
{
    /// <summary>
    /// The default chunk length in seconds.
    /// </summary>
    public const int DefaultChunkLength = 30;

    /// <summary>
    /// Split audio into chunks of specified length.
    /// </summary>
    /// <param name="waveform">The input waveform tensor (batch_size, length).</param>
    /// <param name="waveformLen">The input waveform lengths tensor (batch_size).</param>
    /// <param name="sampleRate">The sample rate in Hz.</param>
    /// <param name="chunkLength">The chunk length in seconds.</param>
    /// <returns>A list of (waveform, length) tuples.</returns>
    public static IEnumerable<(Tensor<float>, Tensor<int>)> SplitAudio(
        Tensor<float> waveform,
        Tensor<int> waveformLen,
        int sampleRate,
        int chunkLength = DefaultChunkLength)
    {
        var samplesPerChunk = chunkLength * sampleRate;

        for (int b = 0; b < waveform.Dimensions[0]; b++)
        {
            var length = waveformLen[b];
            var offset = 0;

            while (offset < length)
            {
                var chunkSize = Math.Min(samplesPerChunk, length - offset);
                var chunkData = new float[chunkSize];

                // Copy data in smaller chunks to avoid memory issues
                var copySize = Math.Min(chunkSize, 10000); // Copy in 10k sample chunks
                var copied = 0;

                while (copied < chunkSize)
                {
                    var currentCopySize = Math.Min(copySize, chunkSize - copied);

                    for (int i = 0; i < currentCopySize; i++)
                    {
                        try
                        {
                            chunkData[copied + i] = waveform[b, offset + copied + i];
                        }
                        catch (Exception ex)
                        {
                            throw new InvalidOperationException(
                                $"Error accessing tensor at position [{b}, {offset + copied + i}]: {ex.Message}");
                        }
                    }

                    copied += currentCopySize;
                }

                yield return (
                    new DenseTensor<float>(chunkData, new int[] { 1, chunkSize }),
                    new DenseTensor<int>(new int[] { chunkSize }, new int[] { 1 }));

                offset += chunkSize;
            }
        }
    }

    /// <summary>
    /// Combine a sequence of TimestampedResult with proper timestamp adjustment.
    /// </summary>
    /// <param name="results">The sequence of results to combine.</param>
    /// <param name="sampleRate">The sample rate in Hz.</param>
    /// <param name="chunkLength">The chunk length in seconds.</param>
    /// <returns>A combined TimestampedResult.</returns>
    public static TimestampedResult CombineResults(
        IEnumerable<TimestampedResult> results,
        int sampleRate,
        int chunkLength = DefaultChunkLength)
    {
        var samplesPerChunk = chunkLength * sampleRate;
        var combinedResult = new TimestampedResult
        {
            Text = string.Empty,
            Tokens = Array.Empty<string>(),
            Timestamps = Array.Empty<float>()
        };

        var chunkIndex = 0;
        foreach (var result in results)
        {
            if (chunkIndex > 0)
            {
                combinedResult.Text += " ";
            }

            combinedResult.Text += result?.Text ?? string.Empty;

            var resultTokens = result?.Tokens ?? Array.Empty<string>();
            combinedResult.Tokens = combinedResult.Tokens.Concat(resultTokens).ToArray();

            var resultTimestamps = result?.Timestamps ?? Array.Empty<float>();
            var adjusted = resultTimestamps.Select(ts => ts + chunkIndex * chunkLength).ToArray();
            combinedResult.Timestamps = combinedResult.Timestamps.Concat(adjusted).ToArray();

            chunkIndex++;
        }

        return combinedResult;
    }
}