using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxAsr;

/// <summary>
/// Kaldi Transducer model.
/// </summary>
public class KaldiTransducer : AsrWithTransducerDecoding<Dictionary<string, Tensor<float>>>
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly InferenceSession _joiner;

    /// <summary>
    /// Initializes a new instance of the <see cref="KaldiTransducer"/> class.
    /// </summary>
    /// <param name="modelFiles">The model files.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    public KaldiTransducer(Dictionary<string, string> modelFiles, OnnxSessionOptions onnxOptions)
        : base(modelFiles, onnxOptions)
    {
        var sessionOptions = onnxOptions.SessionOptions ?? new SessionOptions();
        _encoder = new InferenceSession(modelFiles["encoder"], sessionOptions);
        _decoder = new InferenceSession(modelFiles["decoder"], sessionOptions);
        _joiner = new InferenceSession(modelFiles["joiner"], sessionOptions);
    }

    /// <summary>
    /// Gets the preprocessor name.
    /// </summary>
    protected override string PreprocessorName => "kaldi";

    /// <summary>
    /// Gets the subsampling factor.
    /// </summary>
    protected override int SubsamplingFactor => Config.SubsamplingFactor ?? 4;

    /// <summary>
    /// Gets the max tokens per step.
    /// </summary>
    protected override int MaxTokensPerStep => Config.MaxTokensPerStep ?? 1;

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
            ["encoder"] = $"*/encoder{suffix}.onnx",
            ["decoder"] = $"*/decoder{suffix}.onnx",
            ["joiner"] = $"*/joiner{suffix}.onnx",
            ["vocab"] = "*/tokens.txt"
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
            NamedOnnxValue.CreateFromTensor("x", features),
            OnnxSessionUtils.CreateLengthInput(_encoder, "x_lens", featuresLens)
        };

        using var results = _encoder.Run(inputs);
        var encoderOut = results.First(r => r.Name == "encoder_out").AsTensor<float>();
        var encoderOutLensLong = results.First(r => r.Name == "encoder_out_lens").AsTensor<long>();
        var encoderOutLens = new DenseTensor<int>(encoderOutLensLong.Dimensions);
        var encoderOutLensData = encoderOutLensLong.ToArray();
        for (int i = 0; i < encoderOutLensData.Length; i++)
        {
            encoderOutLens[i] = (int)encoderOutLensData[i];
        }

        return (encoderOut, encoderOutLens);
    }

    /// <summary>
    /// Creates the state.
    /// </summary>
    /// <returns>The state.</returns>
    protected override Dictionary<string, Tensor<float>> CreateState()
    {
        return new Dictionary<string, Tensor<float>>();
    }

    /// <summary>
    /// Decodes a step.
    /// </summary>
    /// <param name="prevTokens">The previous tokens.</param>
    /// <param name="prevState">The previous state.</param>
    /// <param name="encoderOut">The encoder output.</param>
    /// <param name="t">The time index.</param>
    /// <returns>The probabilities, step, and state.</returns>
    protected override (Tensor<float>, int, Dictionary<string, Tensor<float>>) Decode(List<int> prevTokens, Dictionary<string, Tensor<float>> prevState, Tensor<float> encoderOut, int t)
    {
        var context = prevTokens.TakeLast(2).Prepend(-1).Append(BlankIdx).TakeLast(2).Select(i => (long)i).ToArray();
        var contextKey = string.Join(",", context);

        Tensor<float> decoderOut;
        if (!prevState.TryGetValue(contextKey, out decoderOut))
        {
            // Create context tensor matching decoder input element type
            var decoderInputMeta = _decoder.InputMetadata["y"];
            var decoderElemType = decoderInputMeta.ElementType;

            NamedOnnxValue yInput;
            if (decoderElemType == typeof(long) || decoderElemType == typeof(System.Int64))
            {
                var contextLong = context;
                yInput = NamedOnnxValue.CreateFromTensor("y", new DenseTensor<long>(contextLong, new int[] { 1, context.Length }));
            }
            else
            {
                var contextInt = context.Select(i => (int)i).ToArray();
                yInput = NamedOnnxValue.CreateFromTensor("y", new DenseTensor<int>(contextInt, new int[] { 1, context.Length }));
            }

            var decoderInputs = new List<NamedOnnxValue>
            {
                yInput
            };

            using var decoderResults = _decoder.Run(decoderInputs);
            decoderOut = decoderResults.First(r => r.Name == "decoder_out").AsTensor<float>();
            prevState[contextKey] = decoderOut;
        }

        // Extract encoder out at time t
        var encoderSlice = new DenseTensor<float>(new float[encoderOut.Dimensions[1]], new int[] { 1, encoderOut.Dimensions[1] });
        for (int d = 0; d < encoderOut.Dimensions[1]; d++)
        {
            encoderSlice[0, d] = encoderOut[t, d];
        }

        var joinerInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_out", encoderSlice),
            NamedOnnxValue.CreateFromTensor("decoder_out", decoderOut)
        };

        using var joinerResults = _joiner.Run(joinerInputs);
        var logit = joinerResults.First(r => r.Name == "logit").AsTensor<float>();

        return (logit, -1, prevState);
    }
}