using NAudio.Wave;
using OnnxAsr;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.RegularExpressions;

Console.WriteLine("ONNX ASR Example");

if (args.Length < 3)
{
    Console.WriteLine("Usage: OnnxAsr.Example <model_name> <model_path> <wav_file>");
    return;
}

var modelName = args[0];
var modelPath = args[1];
var wavFile = args[2];

Console.WriteLine($"Loading model {modelName} from {modelPath}");
Console.WriteLine($"Processing {wavFile}");

// Load ASR model
Asr asr;
try
{
    asr = Loader.LoadModel(modelName, modelPath, sessOptions: GetSessionOptions());
}
catch (Exception ex)
{
    Console.WriteLine($"Failed to load model: {ex.Message}");
    return;
}

// Read WAV file and resample if needed
float[] waveform;
int sampleRate;
using (var reader = new AudioFileReader(wavFile))
{
    sampleRate = reader.WaveFormat.SampleRate;
    if (reader.WaveFormat.Channels != 1)
    {
        Console.WriteLine("Only mono audio is supported.");
        return;
    }

    // Resample to 16kHz if needed using simple linear interpolation
    if (sampleRate != 16000)
    {
        Console.WriteLine($"Resampling from {sampleRate}Hz to 16000Hz...");

        var samples = new List<float>();
        var buffer = new float[1024];
        int bytesRead;
        while ((bytesRead = reader.Read(buffer, 0, buffer.Length)) > 0)
        {
            for (int i = 0; i < bytesRead; i++)
            {
                samples.Add(buffer[i]);
            }
        }

        var originalSamples = samples.ToArray();
        var ratio = 16000.0 / sampleRate;
        var newLength = (int)(originalSamples.Length * ratio);
        waveform = new float[newLength];

        for (int i = 0; i < newLength; i++)
        {
            var originalIndex = i / ratio;
            var index = (int)originalIndex;
            var fraction = originalIndex - index;

            if (index < originalSamples.Length - 1)
            {
                waveform[i] = (float)(originalSamples[index] * (1 - fraction) + originalSamples[index + 1] * fraction);
            }
            else
            {
                waveform[i] = originalSamples[index];
            }
        }

        sampleRate = 16000;
    }
    else
    {
        var samples = new List<float>();
        var buffer = new float[1024];
        int bytesRead;
        while ((bytesRead = reader.Read(buffer, 0, buffer.Length)) > 0)
        {
            for (int i = 0; i < bytesRead; i++)
            {
                samples.Add(buffer[i]);
            }
        }
        waveform = samples.ToArray();
    }
}

Console.WriteLine($"Sample Rate: {sampleRate}, Length: {waveform.Length}");

// Process audio in smaller chunks manually to avoid memory issues
try
{
    Console.WriteLine("Processing audio in chunks...");

    var chunkResults = new List<TimestampedResult>();
    var chunkSize = 30 * sampleRate; // 30-second chunks
    var totalSamples = waveform.Length;

    for (int offset = 0; offset < totalSamples; offset += chunkSize)
    {
        var currentChunkSize = Math.Min(chunkSize, totalSamples - offset);

        if (currentChunkSize > 0)
        {
            Console.WriteLine($"Processing chunk from {offset} to {offset + currentChunkSize}...");

            // Extract chunk from the waveform array
            var chunkData = new float[currentChunkSize];
            Array.Copy(waveform, offset, chunkData, 0, currentChunkSize);

            var chunkWaveform = new DenseTensor<float>(chunkData, new int[] { 1, currentChunkSize });
            var chunkLen = new DenseTensor<int>(new int[] { currentChunkSize }, new int[] { 1 });

            var chunkResult = asr.RecognizeBatch(chunkWaveform, chunkLen, "en");

            // Adjust timestamps for chunk position
            var adjustedResults = chunkResult.Select(result =>
            {
                if (result.Timestamps != null)
                {
                    var adjustedTimestamps = result.Timestamps.Select(ts => ts + (float)offset / sampleRate).ToArray();
                    return new TimestampedResult
                    {
                        Text = result.Text,
                        Tokens = result.Tokens,
                        Timestamps = adjustedTimestamps
                    };
                }
                return result;
            });

            chunkResults.AddRange(adjustedResults);
        }
    }

    // Combine results
    if (chunkResults.Count > 0)
    {
        var combinedText = string.Join(" ", chunkResults.Select(r => r.Text).Where(t => !string.IsNullOrEmpty(t)));
        var combinedTokens = chunkResults.SelectMany(r => r.Tokens ?? Array.Empty<string>()).ToArray();
        var combinedTimestamps = chunkResults.SelectMany(r => r.Timestamps ?? Array.Empty<float>()).ToArray();

        Console.WriteLine($"Recognized: {combinedText}");

        // Segment into sentences by gaps in timestamps (no fallback)
        if (combinedTokens != null && combinedTimestamps != null &&
            combinedTokens.Length == combinedTimestamps.Length &&
            combinedTokens.Length > 0)
        {
            // First group tokens into words (split by spaces)
            var words = new List<(string word, float startTime, float endTime)>();
            var currentWord = new List<string>();
            var wordStart = combinedTimestamps[0];

            for (int i = 0; i < combinedTokens.Length; i++)
            {
                var token = combinedTokens[i];
                var timestamp = combinedTimestamps[i];

                if (token == " ")
                {
                    if (currentWord.Count > 0)
                    {
                        var wordText = string.Join("", currentWord);
                        words.Add((wordText, wordStart, timestamp));
                        currentWord.Clear();
                    }
                    wordStart = timestamp; // Next word starts after space
                }
                else
                {
                    currentWord.Add(token);
                }
            }

            // Add the last word if any
            if (currentWord.Count > 0)
            {
                var wordText = string.Join("", currentWord);
                words.Add((wordText, wordStart, combinedTimestamps[combinedTimestamps.Length - 1]));
            }

            Console.WriteLine($"Grouped into {words.Count} words");

            //// Now segment words by time gaps
            //var sentences = new List<string>();
            //var currentSentenceWords = new List<string>();

            //for (int i = 0; i < words.Count; i++)
            //{
            //    var (word, start, end) = words[i];
            //    currentSentenceWords.Add(word);

            //    // Check if there's a significant gap to next word
            //    if (i < words.Count - 1)
            //    {
            //        var nextStart = words[i + 1].startTime;
            //        var gap = nextStart - end;

            //        Console.WriteLine($"Word '{word}' ends at {end:F2}s, next starts at {nextStart:F2}s, gap: {gap:F2}s");

            //        if (gap > 0.05f) // Gap threshold
            //        {
            //            var sentence = string.Join(" ", currentSentenceWords);
            //            if (!string.IsNullOrWhiteSpace(sentence))
            //            {
            //                sentences.Add(sentence);
            //            }
            //            currentSentenceWords.Clear();
            //            Console.WriteLine($"Sentence break detected, sentences so far: {sentences.Count}");
            //        }
            //    }
            //}

            //// Add the last sentence
            //if (currentSentenceWords.Count > 0)
            //{
            //    var sentence = string.Join(" ", currentSentenceWords);
            //    if (!string.IsNullOrWhiteSpace(sentence))
            //    {
            //        sentences.Add(sentence);
            //    }
            //}

            //Console.WriteLine($"Total sentences: {sentences.Count}");

            //if (sentences.Count > 1)
            //{
            //    Console.WriteLine("\nSentences:");
            //    for (int i = 0; i < sentences.Count; i++)
            //    {
            //        Console.WriteLine($"{i + 1}. {sentences[i]}");
            //    }
            //}
        }

        if (combinedTimestamps != null && combinedTimestamps.Length > 0)
        {
            Console.WriteLine("\nWord timestamps:");
            for (int i = 0; i < combinedTimestamps.Length; i++)
            {
                Console.WriteLine($"{combinedTokens?[i]}: {combinedTimestamps[i]:F2}s");
            }
        }
    }
    else
    {
        Console.WriteLine("No recognition results returned.");
    }
}
catch (Exception ex)
{
    Console.WriteLine("Recognition failed: \n" + ex.ToString());
}

static Microsoft.ML.OnnxRuntime.SessionOptions GetSessionOptions()
{
    var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions();
    sessionOptions.AppendExecutionProvider_CUDA(0);

    return sessionOptions;
}