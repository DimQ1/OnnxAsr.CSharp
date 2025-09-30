using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxAsr;

/// <summary>
/// GigaAM v2 base class.
/// </summary>
public abstract class GigaamV2 : AsrWithDecoding
{
    /// <summary>
    /// Initializes a new instance of the <see cref="GigaamV2"/> class.
    /// </summary>
    /// <param name="modelFiles">The model files.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    protected GigaamV2(Dictionary<string, string> modelFiles, OnnxSessionOptions onnxOptions)
        : base(modelFiles, onnxOptions)
    {
    }

    /// <summary>
    /// Gets the preprocessor name.
    /// </summary>
    protected override string PreprocessorName => "gigaam";

    /// <summary>
    /// Gets the subsampling factor.
    /// </summary>
    protected override int SubsamplingFactor => Config.SubsamplingFactor ?? 4;

    /// <summary>
    /// Gets the model files.
    /// </summary>
    /// <param name="quantization">The quantization.</param>
    /// <returns>The model files.</returns>
    public static Dictionary<string, string> GetModelFiles(string? quantization = null)
    {
        return new Dictionary<string, string> { ["vocab"] = "v2_vocab.txt" };
    }
}

/// <summary>
/// GigaAM v2 CTC model.
/// </summary>
public class GigaamV2Ctc : AsrWithCtcDecoding
{
    private readonly InferenceSession _model;

    /// <summary>
    /// Initializes a new instance of the <see cref="GigaamV2Ctc"/> class.
    /// </summary>
    /// <param name="modelFiles">The model files.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    public GigaamV2Ctc(Dictionary<string, string> modelFiles, OnnxSessionOptions onnxOptions)
        : base(modelFiles, onnxOptions)
    {
        var sessionOptions = onnxOptions.SessionOptions ?? new SessionOptions();
        _model = new InferenceSession(modelFiles["model"], sessionOptions);
    }

    /// <summary>
    /// Gets the preprocessor name.
    /// </summary>
    protected override string PreprocessorName => "gigaam";

    /// <summary>
    /// Gets the subsampling factor.
    /// </summary>
    protected override int SubsamplingFactor => Config.SubsamplingFactor ?? 4;

    /// <summary>
    /// Gets the model files.
    /// </summary>
    /// <param name="quantization">The quantization.</param>
    /// <returns>The model files.</returns>
    public static Dictionary<string, string> GetModelFiles(string? quantization = null)
    {
        var suffix = quantization != null ? "?" + quantization : "";
        return new Dictionary<string, string>
        {
            ["model"] = $"v2_ctc{suffix}.onnx",
            ["vocab"] = "v2_vocab.txt",
            ["config"] = "config.json"
        };
    }

    /// <summary>
    /// Encodes the features.
    /// </summary>
    /// <param name="features">The features.</param>
    /// <param name="featuresLens">The features lengths.</param>
    /// <returns>The encoded output.</returns>
    protected override (Tensor<float>, Tensor<int>) Encode(Tensor<float> features, Tensor<int> featuresLens)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("features", features),
            OnnxSessionUtils.CreateLengthInput(_model, "feature_lengths", featuresLens)
        };

        using var results = _model.Run(inputs);
        var logProbs = results.First(r => r.Name == "log_probs").AsTensor<float>();
        var lengths = featuresLens.ToArray().Select(l => (l - 1) / SubsamplingFactor + 1).ToArray();
        var lengthsTensor = new DenseTensor<int>(lengths, new int[] { lengths.Length });
        return (logProbs, lengthsTensor);
    }
}

/// <summary>
/// GigaAM v2 RNN-T model.
/// </summary>
public class GigaamV2Rnnt : AsrWithTransducerDecoding<List<Tensor<float>>>
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly InferenceSession _joiner;

    /// <summary>
    /// Initializes a new instance of the <see cref="GigaamV2Rnnt"/> class.
    /// </summary>
    /// <param name="modelFiles">The model files.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    public GigaamV2Rnnt(Dictionary<string, string> modelFiles, OnnxSessionOptions onnxOptions)
        : base(modelFiles, onnxOptions)
    {
        var sessionOptions = onnxOptions.SessionOptions ?? new SessionOptions();
        _encoder = new InferenceSession(modelFiles["encoder"], sessionOptions);
        _decoder = new InferenceSession(modelFiles["decoder"], sessionOptions);
        _joiner = new InferenceSession(modelFiles["joint"], sessionOptions);
    }

    /// <summary>
    /// Gets the preprocessor name.
    /// </summary>
    protected override string PreprocessorName => "gigaam";

    /// <summary>
    /// Gets the subsampling factor.
    /// </summary>
    protected override int SubsamplingFactor => Config.SubsamplingFactor ?? 4;

    /// <summary>
    /// Gets the max tokens per step.
    /// </summary>
    protected override int MaxTokensPerStep => Config.MaxTokensPerStep ?? 3;

    /// <summary>
    /// Gets the model files.
    /// </summary>
    /// <param name="quantization">The quantization.</param>
    /// <returns>The model files.</returns>
    public static Dictionary<string, string> GetModelFiles(string? quantization = null)
    {
        var suffix = quantization != null ? "?" + quantization : "";
        return new Dictionary<string, string>
        {
            ["encoder"] = $"v2_rnnt_encoder{suffix}.onnx",
            ["decoder"] = $"v2_rnnt_decoder{suffix}.onnx",
            ["joint"] = $"v2_rnnt_joint{suffix}.onnx",
            ["vocab"] = "v2_vocab.txt"
        };
    }

    /// <summary>
    /// Encodes the features.
    /// </summary>
    /// <param name="features">The features.</param>
    /// <param name="featuresLens">The features lengths.</param>
    /// <returns>The encoded output.</returns>
    protected override (Tensor<float>, Tensor<int>) Encode(Tensor<float> features, Tensor<int> featuresLens)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("audio_signal", features),
            OnnxSessionUtils.CreateLengthInput(_encoder, "length", featuresLens)
        };

        using var results = _encoder.Run(inputs);
        var encoded = results.First(r => r.Name == "encoded").AsTensor<float>();
        var encodedLen = results.First(r => r.Name == "encoded_len").AsTensor<int>();

        // Transpose to (batch, time, dim)
        var encodedData = encoded.ToArray();
        var newShape = new int[] { encoded.Dimensions[0], encoded.Dimensions[2], encoded.Dimensions[1] };
        var transposed = new DenseTensor<float>(new float[encodedData.Length], newShape);
        for (int b = 0; b < encoded.Dimensions[0]; b++)
        {
            for (int t = 0; t < encoded.Dimensions[1]; t++)
            {
                for (int d = 0; d < encoded.Dimensions[2]; d++)
                {
                    transposed[b, d, t] = encoded[b, t, d];
                }
            }
        }

        return (transposed, encodedLen);
    }

    /// <summary>
    /// Creates the state.
    /// </summary>
    /// <returns>The state.</returns>
    protected override List<Tensor<float>> CreateState()
    {
        return new List<Tensor<float>>
        {
            new DenseTensor<float>(new float[1 * 1 * 320], new int[] { 1, 1, 320 }),
            new DenseTensor<float>(new float[1 * 1 * 320], new int[] { 1, 1, 320 })
        };
    }

    /// <summary>
    /// Decodes a step.
    /// </summary>
    /// <param name="prevTokens">The previous tokens.</param>
    /// <param name="prevState">The previous state.</param>
    /// <param name="encoderOut">The encoder output.</param>
    /// <param name="t">The time index.</param>
    /// <returns>The probabilities, step, and state.</returns>
    protected override (Tensor<float>, int, List<Tensor<float>>) Decode(List<int> prevTokens, List<Tensor<float>> prevState, Tensor<float> encoderOut, int t)
    {
        Tensor<float> decoderOut;
        List<Tensor<float>> newState;

        if (prevState[0].Dimensions[0] == 0)
        {
            // First call
            var decoderInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("x", new DenseTensor<long>(new long[] { prevTokens.Count > 0 ? prevTokens.Last() : BlankIdx }, new int[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("h.1", prevState[0]),
                NamedOnnxValue.CreateFromTensor("c.1", prevState[1])
            };

            using var decoderResults = _decoder.Run(decoderInputs);
            decoderOut = decoderResults.First(r => r.Name == "dec").AsTensor<float>();
            var state1 = decoderResults.First(r => r.Name == "h").AsTensor<float>();
            var state2 = decoderResults.First(r => r.Name == "c").AsTensor<float>();
            newState = new List<Tensor<float>> { decoderOut, state1, state2 };
        }
        else
        {
            decoderOut = prevState[0];
            newState = prevState;
        }

        // Extract encoder out at time t
        var encoderSlice = new DenseTensor<float>(new float[encoderOut.Dimensions[2]], new int[] { 1, 1, encoderOut.Dimensions[2] });
        for (int d = 0; d < encoderOut.Dimensions[2]; d++)
        {
            encoderSlice[0, 0, d] = encoderOut[0, t, d];
        }

        var joinerInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("enc", encoderSlice),
            NamedOnnxValue.CreateFromTensor("dec", decoderOut)
        };

        using var joinerResults = _joiner.Run(joinerInputs);
        var joint = joinerResults.First(r => r.Name == "joint").AsTensor<float>();

        return (joint, -1, newState);
    }
}