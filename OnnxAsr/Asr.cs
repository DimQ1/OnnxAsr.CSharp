using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxAsr;

/// <summary>
/// Base ASR class.
/// </summary>
public abstract class Asr
{
    /// <summary>
    /// Gets the config.
    /// </summary>
    protected AsrConfig Config { get; }

    /// <summary>
    /// Gets the preprocessor.
    /// </summary>
    protected Preprocessor Preprocessor { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="Asr"/> class.
    /// </summary>
    /// <param name="modelFiles">The model files.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    protected Asr(Dictionary<string, string> modelFiles, OnnxSessionOptions onnxOptions)
    {
        if (modelFiles.TryGetValue("config", out var configPath))
        {
            var configJson = File.ReadAllText(configPath);
            Config = System.Text.Json.JsonSerializer.Deserialize<AsrConfig>(configJson) ?? new AsrConfig();
        }
        else
        {
            Config = new AsrConfig();
        }

        Preprocessor = new Preprocessor(PreprocessorName, onnxOptions);
    }

    /// <summary>
    /// Gets the preprocessor name.
    /// </summary>
    protected abstract string PreprocessorName { get; }

    /// <summary>
    /// Recognizes a batch of waveforms.
    /// </summary>
    /// <param name="waveforms">The waveforms.</param>
    /// <param name="waveformsLen">The waveforms lengths.</param>
    /// <param name="language">The language.</param>
    /// <returns>The recognition results.</returns>
    public abstract IEnumerable<TimestampedResult> RecognizeBatch(Tensor<float> waveforms, Tensor<int> waveformsLen, string? language = null);
}

/// <summary>
/// Base ASR with decoding.
/// </summary>
public abstract class AsrWithDecoding : Asr
{
    /// <summary>
    /// Gets the vocab.
    /// </summary>
    protected Dictionary<int, string> Vocab { get; }

    /// <summary>
    /// Gets the vocab size.
    /// </summary>
    protected int VocabSize { get; }

    /// <summary>
    /// Gets the blank index.
    /// </summary>
    protected int BlankIdx { get; }

    /// <summary>
    /// Gets the subsampling factor.
    /// </summary>
    protected abstract int SubsamplingFactor { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="AsrWithDecoding"/> class.
    /// </summary>
    /// <param name="modelFiles">The model files.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    protected AsrWithDecoding(Dictionary<string, string> modelFiles, OnnxSessionOptions onnxOptions)
        : base(modelFiles, onnxOptions)
    {
        var vocabLines = File.ReadAllLines(modelFiles["vocab"]);
        var tokens = vocabLines.Select(line => line.Split(' ', 2)).ToDictionary(parts => parts[0], parts => int.Parse(parts[1]));
        Vocab = tokens.ToDictionary(kvp => kvp.Value, kvp => kvp.Key.Replace("\u2581", " "));
        VocabSize = Vocab.Count;
        BlankIdx = tokens["<blk>"];
    }

    /// <summary>
    /// Encodes the features.
    /// </summary>
    /// <param name="features">The features.</param>
    /// <param name="featuresLens">The features lengths.</param>
    /// <returns>The encoded output.</returns>
    protected abstract (Tensor<float>, Tensor<int>) Encode(Tensor<float> features, Tensor<int> featuresLens);

    /// <summary>
    /// Decodes the tokens.
    /// </summary>
    /// <param name="tokens">The tokens.</param>
    /// <param name="timestamps">The timestamps.</param>
    /// <returns>The timestamped result.</returns>
    protected TimestampedResult DecodeTokens(List<int> tokens, List<float> timestamps)
    {
        var text = string.Join("", tokens.Select(t => Vocab[t]));
        text = System.Text.RegularExpressions.Regex.Replace(text, @"(?<!\s)\s\B|\A\s|\s\B", m => m.Value == " " ? "" : " ");
        return new TimestampedResult { Text = text, Timestamps = timestamps.ToArray(), Tokens = tokens.Select(t => Vocab[t]).ToArray() };
    }

    /// <summary>
    /// Recognizes a batch of waveforms.
    /// </summary>
    /// <param name="waveforms">The waveforms.</param>
    /// <param name="waveformsLen">The waveforms lengths.</param>
    /// <param name="language">The language.</param>
    /// <returns>The recognition results.</returns>
    public override IEnumerable<TimestampedResult> RecognizeBatch(Tensor<float> waveforms, Tensor<int> waveformsLen, string? language = null)
    {
        var (features, featuresLens) = Preprocessor.Process(waveforms, waveformsLen);
        var (encoderOut, encoderOutLens) = Encode(features, featuresLens);
        return Decoding(encoderOut, encoderOutLens);
    }

    /// <summary>
    /// Performs decoding.
    /// </summary>
    /// <param name="encoderOut">The encoder output.</param>
    /// <param name="encoderOutLens">The encoder output lengths.</param>
    /// <returns>The results.</returns>
    protected abstract IEnumerable<TimestampedResult> Decoding(Tensor<float> encoderOut, Tensor<int> encoderOutLens);
}

/// <summary>
/// ASR with CTC decoding.
/// </summary>
public abstract class AsrWithCtcDecoding : AsrWithDecoding
{
    /// <summary>
    /// Initializes a new instance of the <see cref="AsrWithCtcDecoding"/> class.
    /// </summary>
    /// <param name="modelFiles">The model files.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    protected AsrWithCtcDecoding(Dictionary<string, string> modelFiles, OnnxSessionOptions onnxOptions)
        : base(modelFiles, onnxOptions)
    {
    }

    /// <summary>
    /// Performs CTC decoding.
    /// </summary>
    /// <param name="encoderOut">The encoder output.</param>
    /// <param name="encoderOutLens">The encoder output lengths.</param>
    /// <returns>The results.</returns>
    protected override IEnumerable<TimestampedResult> Decoding(Tensor<float> encoderOut, Tensor<int> encoderOutLens)
    {
        var encoderOutData = encoderOut.ToArray();
        var encoderOutLensData = encoderOutLens.ToArray();

        for (int b = 0; b < encoderOut.Dimensions[0]; b++)
        {
            var logProbs = encoderOutData.Skip(b * encoderOut.Dimensions[1] * encoderOut.Dimensions[2]).Take(encoderOut.Dimensions[1] * encoderOut.Dimensions[2]).ToArray();
            var len = encoderOutLensData[b];

            var tokens = new List<int>();
            var timestamps = new List<float>();

            for (int t = 0; t < len; t++)
            {
                var maxIdx = 0;
                var maxVal = float.MinValue;
                for (int v = 0; v < VocabSize; v++)
                {
                    var val = logProbs[t * VocabSize + v];
                    if (val > maxVal)
                    {
                        maxVal = val;
                        maxIdx = v;
                    }
                }

                if (maxIdx != BlankIdx)
                {
                    if (tokens.Count == 0 || tokens.Last() != maxIdx)
                    {
                        tokens.Add(maxIdx);
                        timestamps.Add(t * 0.01f * SubsamplingFactor);
                    }
                }
            }

            yield return DecodeTokens(tokens, timestamps);
        }
    }
}

/// <summary>
/// ASR with transducer decoding.
/// </summary>
public abstract class AsrWithTransducerDecoding<TState> : AsrWithDecoding
{
    /// <summary>
    /// Gets the max tokens per step.
    /// </summary>
    protected abstract int MaxTokensPerStep { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="AsrWithTransducerDecoding{TState}"/> class.
    /// </summary>
    /// <param name="modelFiles">The model files.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    protected AsrWithTransducerDecoding(Dictionary<string, string> modelFiles, OnnxSessionOptions onnxOptions)
        : base(modelFiles, onnxOptions)
    {
    }

    /// <summary>
    /// Creates the state.
    /// </summary>
    /// <returns>The state.</returns>
    protected abstract TState CreateState();

    /// <summary>
    /// Decodes a step.
    /// </summary>
    /// <param name="prevTokens">The previous tokens.</param>
    /// <param name="prevState">The previous state.</param>
    /// <param name="encoderOut">The encoder output.</param>
    /// <param name="t">The time index.</param>
    /// <returns>The probabilities, step, and state.</returns>
    protected abstract (Tensor<float>, int, TState) Decode(List<int> prevTokens, TState prevState, Tensor<float> encoderOut, int t);

    /// <summary>
    /// Performs transducer decoding.
    /// </summary>
    /// <param name="encoderOut">The encoder output.</param>
    /// <param name="encoderOutLens">The encoder output lengths.</param>
    /// <returns>The results.</returns>
    protected override IEnumerable<TimestampedResult> Decoding(Tensor<float> encoderOut, Tensor<int> encoderOutLens)
    {
        var encoderOutData = encoderOut.ToArray();
        var encoderOutLensData = encoderOutLens.ToArray();

        for (int b = 0; b < encoderOut.Dimensions[0]; b++)
        {
            var encodings = new DenseTensor<float>(encoderOutData.Skip(b * encoderOut.Dimensions[1] * encoderOut.Dimensions[2]).Take(encoderOut.Dimensions[1] * encoderOut.Dimensions[2]).ToArray(), new int[] { encoderOut.Dimensions[1], encoderOut.Dimensions[2] });
            var encodingsLen = (int)encoderOutLensData[b];

            var prevState = CreateState();
            var tokens = new List<int>();
            var timestamps = new List<float>();

            int t = 0;
            int emittedTokens = 0;
            while (t < encodingsLen)
            {
                var (probs, step, state) = Decode(tokens, prevState, encodings, t);
                var probsData = probs.ToArray();

                var token = Array.IndexOf(probsData, probsData.Max());

                if (token != BlankIdx)
                {
                    prevState = state;
                    tokens.Add(token);
                    timestamps.Add(t);
                    emittedTokens++;
                }

                if (step > 0)
                {
                    t += step;
                    emittedTokens = 0;
                }
                else if (token == BlankIdx || emittedTokens == MaxTokensPerStep)
                {
                    t++;
                    emittedTokens = 0;
                }
            }

            yield return DecodeTokens(tokens, timestamps.Select(ts => ts * 0.01f * SubsamplingFactor).ToList());
        }
    }
}