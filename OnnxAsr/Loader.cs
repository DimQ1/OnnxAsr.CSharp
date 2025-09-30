using Microsoft.ML.OnnxRuntime;

namespace OnnxAsr;

/// <summary>
/// Loader for ASR models.
/// </summary>
public static class Loader
{
    /// <summary>
    /// Loads an ASR model.
    /// </summary>
    /// <param name="model">The model name.</param>
    /// <param name="path">The path to model files.</param>
    /// <param name="quantization">The quantization.</param>
    /// <param name="sessOptions">The session options.</param>
    /// <param name="providers">The providers.</param>
    /// <param name="providerOptions">The provider options.</param>
    /// <param name="cpuPreprocessing">Whether to use CPU preprocessing.</param>
    /// <returns>The ASR model.</returns>
    public static Asr LoadModel(
        string model,
        string? path = null,
        string? quantization = null,
        SessionOptions? sessOptions = null,
        string[]? providers = null,
        ProviderOptions[]? providerOptions = null,
        bool cpuPreprocessing = true)
    {
        var onnxOptions = new OnnxSessionOptions
        {
            SessionOptions = sessOptions,
            Providers = providers ?? new[] { "CPUExecutionProvider" },
            ProviderOptions = providerOptions,
            CpuPreprocessing = cpuPreprocessing
        };

        Dictionary<string, string> modelFiles;
        Asr asr;

        switch (model)
        {
            case "gigaam-v2-ctc":
                modelFiles = GigaamV2Ctc.GetModelFiles(quantization);
                if (path != null)
                {
                    modelFiles = modelFiles.ToDictionary(
                        kvp => kvp.Key,
                        kvp => Path.Combine(path, kvp.Value));
                }
                asr = new GigaamV2Ctc(modelFiles, onnxOptions);
                break;
            case "gigaam-v2-rnnt":
                modelFiles = GigaamV2Rnnt.GetModelFiles(quantization);
                if (path != null)
                {
                    modelFiles = modelFiles.ToDictionary(
                        kvp => kvp.Key,
                        kvp => Path.Combine(path, kvp.Value));
                }
                asr = new GigaamV2Rnnt(modelFiles, onnxOptions);
                break;
            case "kaldi-rnnt":
            case "vosk":
                modelFiles = KaldiTransducer.GetModelFiles(quantization);
                if (path != null) modelFiles = modelFiles.ToDictionary(kvp => kvp.Key, kvp => Path.Combine(path, kvp.Value));
                asr = new KaldiTransducer(modelFiles, onnxOptions);
                break;
            case "nemo-conformer-ctc":
                modelFiles = NemoConformerCtc.GetModelFiles(quantization);
                if (path != null) modelFiles = modelFiles.ToDictionary(kvp => kvp.Key, kvp => Path.Combine(path, kvp.Value));
                asr = new NemoConformerCtc(modelFiles, onnxOptions);
                break;
            case "nemo-conformer-rnnt":
                modelFiles = NemoConformerRnnt.GetModelFiles(quantization);
                if (path != null) modelFiles = modelFiles.ToDictionary(kvp => kvp.Key, kvp => Path.Combine(path, kvp.Value));
                asr = new NemoConformerRnnt(modelFiles, onnxOptions);
                break;
            case "nemo-conformer-tdt":
                modelFiles = NemoConformerRnnt.GetModelFiles(quantization); // TDT uses same files
                if (path != null) modelFiles = modelFiles.ToDictionary(kvp => kvp.Key, kvp => Path.Combine(path, kvp.Value));
                asr = new NemoConformerTdt(modelFiles, onnxOptions);
                break;
            case "whisper":
                modelFiles = Whisper.GetModelFiles(quantization);
                if (path != null) modelFiles = modelFiles.ToDictionary(kvp => kvp.Key, kvp => Path.Combine(path, kvp.Value));
                asr = new Whisper(modelFiles, onnxOptions);
                break;
            default:
                throw new ArgumentException($"Model '{model}' not supported!");
        }

        return asr;
    }
}