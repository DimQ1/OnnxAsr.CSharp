using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxAsr;

/// <summary>
/// Whisper ASR model.
/// </summary>
public class Whisper : Asr
{
    private readonly InferenceSession _model;
    private readonly Dictionary<string, int> _tokens;
    private readonly Dictionary<int, string> _vocab;
    private readonly int _bosTokenId;
    private readonly int _eosTokenId;

    /// <summary>
    /// Initializes a new instance of the <see cref="Whisper"/> class.
    /// </summary>
    /// <param name="modelFiles">The model files.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    public Whisper(Dictionary<string, string> modelFiles, OnnxSessionOptions onnxOptions)
        : base(modelFiles, onnxOptions)
    {
        // Load vocab
        var vocabJson = File.ReadAllText(modelFiles["vocab"]);
        _tokens = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson) ?? new();

        var addedTokensJson = File.ReadAllText(modelFiles["added_tokens"]);
        var addedTokens = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, int>>(addedTokensJson) ?? new();
        foreach (var token in addedTokens)
        {
            _tokens[token.Key] = token.Value;
        }

        _vocab = _tokens.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        _bosTokenId = _tokens["<|startoftranscript|>"];
        _eosTokenId = _tokens["<|endoftext|>"];

        // Load model
        var sessionOptions = onnxOptions.SessionOptions ?? new SessionOptions();
        _model = new InferenceSession(modelFiles["model"], sessionOptions);

        // Debug: print model input metadata
        Console.WriteLine("[Whisper] model inputs:");
        foreach (var kv in _model.InputMetadata)
        {
            var name = kv.Key;
            var meta = kv.Value;
            Console.WriteLine($"  {name}: elementType={meta.ElementType}, dims={string.Join(',', meta.Dimensions ?? Array.Empty<int>())}");
        }
    }

    /// <summary>
    /// Gets the preprocessor name.
    /// </summary>
    protected override string PreprocessorName => $"whisper{Config.FeaturesSize ?? 80}";

    /// <summary>
    /// Recognizes a batch of waveforms.
    /// </summary>
    /// <param name="waveforms">The waveforms.</param>
    /// <param name="waveformsLen">The waveforms lengths.</param>
    /// <param name="language">The language.</param>
    /// <returns>The recognition results.</returns>
    public override IEnumerable<TimestampedResult> RecognizeBatch(Tensor<float> waveforms, Tensor<int> waveformsLen, string? language = null)
    {
        var (features, _) = Preprocessor.Process(waveforms, waveformsLen);

        // Prepare inputs for decoding
        var transcribeInput = new int[] { _bosTokenId, _eosTokenId, _tokens["<|transcribe|>"], _tokens["<|notimestamps|>"] };
        if (language != null && _tokens.TryGetValue($"<|{language}|>", out var langToken))
        {
            transcribeInput[1] = langToken;
        }

        var tokens = new DenseTensor<int>(transcribeInput, new int[] { 1, transcribeInput.Length });

        var featuresFloat = features.ToArray();
        Console.WriteLine($"[Whisper] features dims: [{string.Join(',', features.Dimensions.ToArray())}], features length: {featuresFloat.Length}");
        // Build a float tensor for input_features (model expects float)
        var featuresTensor = new DenseTensor<float>(featuresFloat, features.Dimensions.ToArray());

        // Prepare additional BeamSearch parameters (match Python defaults)
        int maxLengthVal = 448;
        var maxLength = new DenseTensor<int>(new int[] { maxLengthVal }, new int[] { 1 });
        var minLength = new DenseTensor<int>(new int[] { 0 }, new int[] { 1 });
        var numBeams = new DenseTensor<int>(new int[] { 1 }, new int[] { 1 });
        var numReturnSeq = new DenseTensor<int>(new int[] { 1 }, new int[] { 1 });
        var lengthPenalty = new DenseTensor<float>(new float[] { 1.0f }, new int[] { 1 });
        var repetitionPenalty = new DenseTensor<float>(new float[] { 1.0f }, new int[] { 1 });

        // Run model with input_features, decoder_input_ids and BeamSearch params
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_features", featuresTensor),
            NamedOnnxValue.CreateFromTensor("decoder_input_ids", tokens),
            NamedOnnxValue.CreateFromTensor("max_length", maxLength),
            NamedOnnxValue.CreateFromTensor("min_length", minLength),
            NamedOnnxValue.CreateFromTensor("num_beams", numBeams),
            NamedOnnxValue.CreateFromTensor("num_return_sequences", numReturnSeq),
            NamedOnnxValue.CreateFromTensor("length_penalty", lengthPenalty),
            NamedOnnxValue.CreateFromTensor("repetition_penalty", repetitionPenalty),
        };

        using var results = _model.Run(inputs);
        var sequences = results.First(r => r.Name == "sequences").AsTensor<int>();

        // Decode tokens
        var sequencesData = sequences.ToArray();
        var seqLength = sequences.Dimensions[2];
        var seq = new int[seqLength];
        Array.Copy(sequencesData, 0, seq, 0, seqLength);
        var text = string.Join("", seq.Where(id => _vocab.TryGetValue(id, out var token) && !token.StartsWith("<|")).Select(id => _vocab[id]));
        yield return new TimestampedResult { Text = text.TrimStart(' ') };
    }

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
            ["model"] = $"whisper-base_beamsearch{suffix}.onnx",
            ["vocab"] = "vocab.json",
            ["added_tokens"] = "added_tokens.json"
        };
    }
}