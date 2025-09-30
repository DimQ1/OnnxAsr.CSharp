using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxAsr;

/// <summary>
/// NeMo Conformer base class.
/// </summary>
public abstract class NemoConformer : AsrWithDecoding
{
    /// <summary>
    /// Initializes a new instance of the <see cref="NemoConformer"/> class.
    /// </summary>
    /// <param name="modelFiles">The model files.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    protected NemoConformer(Dictionary<string, string> modelFiles, OnnxSessionOptions onnxOptions)
        : base(modelFiles, onnxOptions)
    {
    }

    /// <summary>
    /// Gets the preprocessor name.
    /// </summary>
    protected override string PreprocessorName => $"nemo{Config.FeaturesSize ?? 80}";

    /// <summary>
    /// Gets the subsampling factor.
    /// </summary>
    protected override int SubsamplingFactor => Config.SubsamplingFactor ?? 8;

    /// <summary>
    /// Gets the model files.
    /// </summary>
    /// <param name="quantization">The quantization.</param>
    /// <returns>The model files.</returns>
    public static Dictionary<string, string> GetModelFiles(string? quantization = null)
    {
        return new Dictionary<string, string> { ["vocab"] = "vocab.txt" };
    }
}

/// <summary>
/// NeMo Conformer CTC model.
/// </summary>
public class NemoConformerCtc : AsrWithCtcDecoding
{
    private readonly InferenceSession _model;

    /// <summary>
    /// Initializes a new instance of the <see cref="NemoConformerCtc"/> class.
    /// </summary>
    /// <param name="modelFiles">The model files.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    public NemoConformerCtc(Dictionary<string, string> modelFiles, OnnxSessionOptions onnxOptions)
        : base(modelFiles, onnxOptions)
    {
        var sessionOptions = onnxOptions.SessionOptions ?? new SessionOptions();
        _model = new InferenceSession(modelFiles["model"], sessionOptions);
    }

    /// <summary>
    /// Gets the preprocessor name.
    /// </summary>
    protected override string PreprocessorName => $"nemo{Config.FeaturesSize ?? 80}";

    /// <summary>
    /// Gets the subsampling factor.
    /// </summary>
    protected override int SubsamplingFactor => Config.SubsamplingFactor ?? 8;

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
            ["model"] = $"model{suffix}.onnx",
            ["vocab"] = "vocab.txt"
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
            NamedOnnxValue.CreateFromTensor("length", featuresLens)
        };

        using var results = _model.Run(inputs);
        var logprobs = results.First(r => r.Name == "logprobs").AsTensor<float>();
        var lengths = featuresLens.ToArray().Select(l => (l - 1) / SubsamplingFactor + 1).ToArray();
        var lengthsTensor = new DenseTensor<int>(lengths, new int[] { lengths.Length });
        return (logprobs, lengthsTensor);
    }
}

/// <summary>
/// NeMo Conformer RNN-T model.
/// </summary>
public class NemoConformerRnnt : AsrWithTransducerDecoding<(Tensor<float>, Tensor<float>)>
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoderJoint;

    /// <summary>
    /// Initializes a new instance of the <see cref="NemoConformerRnnt"/> class.
    /// </summary>
    /// <param name="modelFiles">The model files.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    public NemoConformerRnnt(Dictionary<string, string> modelFiles, OnnxSessionOptions onnxOptions)
        : base(modelFiles, onnxOptions)
    {
        var sessionOptions = onnxOptions.SessionOptions ?? new SessionOptions();
        _encoder = new InferenceSession(modelFiles["encoder"], sessionOptions);
        _decoderJoint = new InferenceSession(modelFiles["decoder_joint"], sessionOptions);
    }

    /// <summary>
    /// Gets the preprocessor name.
    /// </summary>
    protected override string PreprocessorName => $"nemo{Config.FeaturesSize ?? 80}";

    /// <summary>
    /// Gets the subsampling factor.
    /// </summary>
    protected override int SubsamplingFactor => Config.SubsamplingFactor ?? 8;

    /// <summary>
    /// Gets the max tokens per step.
    /// </summary>
    protected override int MaxTokensPerStep => Config.MaxTokensPerStep ?? 10;

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
            ["encoder"] = $"encoder-model{suffix}.onnx",
            ["decoder_joint"] = $"decoder_joint-model{suffix}.onnx",
            ["vocab"] = "vocab.txt"
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
            NamedOnnxValue.CreateFromTensor("length", featuresLens)
        };

        using var results = _encoder.Run(inputs);
        var outputs = results.First(r => r.Name == "outputs").AsTensor<float>();
        var encodedLengths = results.First(r => r.Name == "encoded_lengths").AsTensor<int>();

        // Transpose to (batch, time, dim)
        var outputsData = outputs.ToArray();
        var newShape = new int[] { outputs.Dimensions[0], outputs.Dimensions[2], outputs.Dimensions[1] };
        var transposed = new DenseTensor<float>(new float[outputsData.Length], newShape);
        for (int b = 0; b < outputs.Dimensions[0]; b++)
        {
            for (int t = 0; t < outputs.Dimensions[1]; t++)
            {
                for (int d = 0; d < outputs.Dimensions[2]; d++)
                {
                    transposed[b, d, t] = outputs[b, t, d];
                }
            }
        }

        return (transposed, encodedLengths);
    }

    /// <summary>
    /// Creates the state.
    /// </summary>
    /// <returns>The state.</returns>
    protected override (Tensor<float>, Tensor<float>) CreateState()
    {
        // Hardcoded shapes based on typical NeMo models
        return (
            new DenseTensor<float>(new float[1 * 1 * 1024], new int[] { 1, 1, 1024 }),
            new DenseTensor<float>(new float[1 * 1 * 1024], new int[] { 1, 1, 1024 })
        );
    }

    /// <summary>
    /// Decodes a step.
    /// </summary>
    /// <param name="prevTokens">The previous tokens.</param>
    /// <param name="prevState">The previous state.</param>
    /// <param name="encoderOut">The encoder output.</param>
    /// <param name="t">The time index.</param>
    /// <returns>The probabilities, step, and state.</returns>
    protected override (Tensor<float>, int, (Tensor<float>, Tensor<float>)) Decode(List<int> prevTokens, (Tensor<float>, Tensor<float>) prevState, Tensor<float> encoderOut, int t)
    {
        // Extract encoder out at time t
        var encoderSlice = new DenseTensor<float>(new float[encoderOut.Dimensions[2]], new int[] { 1, 1, encoderOut.Dimensions[2] });
        for (int d = 0; d < encoderOut.Dimensions[2]; d++)
        {
            encoderSlice[0, 0, d] = encoderOut[0, t, d];
        }

        var decoderInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_outputs", encoderSlice),
            NamedOnnxValue.CreateFromTensor("targets", new DenseTensor<long>(new long[] { prevTokens.Count > 0 ? prevTokens.Last() : BlankIdx }, new int[] { 1, 1 })),
            NamedOnnxValue.CreateFromTensor("target_length", new DenseTensor<long>(new long[] { 1 }, new int[] { 1 })),
            NamedOnnxValue.CreateFromTensor("input_states_1", prevState.Item1),
            NamedOnnxValue.CreateFromTensor("input_states_2", prevState.Item2)
        };

        using var results = _decoderJoint.Run(decoderInputs);
        var outputs = results.First(r => r.Name == "outputs").AsTensor<float>();
        var outputStates1 = results.First(r => r.Name == "output_states_1").AsTensor<float>();
        var outputStates2 = results.First(r => r.Name == "output_states_2").AsTensor<float>();

        return (outputs, -1, (outputStates1, outputStates2));
    }
}

/// <summary>
/// NeMo Conformer TDT model.
/// </summary>
public class NemoConformerTdt : NemoConformerRnnt
{
    /// <summary>
    /// Initializes a new instance of the <see cref="NemoConformerTdt"/> class.
    /// </summary>
    /// <param name="modelFiles">The model files.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    public NemoConformerTdt(Dictionary<string, string> modelFiles, OnnxSessionOptions onnxOptions)
        : base(modelFiles, onnxOptions)
    {
    }

    /// <summary>
    /// Decodes a step.
    /// </summary>
    /// <param name="prevTokens">The previous tokens.</param>
    /// <param name="prevState">The previous state.</param>
    /// <param name="encoderOut">The encoder output.</param>
    /// <param name="t">The time index.</param>
    /// <returns>The probabilities, step, and state.</returns>
    protected override (Tensor<float>, int, (Tensor<float>, Tensor<float>)) Decode(List<int> prevTokens, (Tensor<float>, Tensor<float>) prevState, Tensor<float> encoderOut, int t)
    {
        var (output, _, state) = base.Decode(prevTokens, prevState, encoderOut, t);
        var outputData = output.ToArray();
        var vocabSize = VocabSize;
        var probs = new DenseTensor<float>(outputData.Take(vocabSize).ToArray(), new int[] { 1, vocabSize });
        var duration = (int)outputData[vocabSize];
        return (probs, duration, state);
    }
}