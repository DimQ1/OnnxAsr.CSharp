namespace OnnxAsr;

/// <summary>
/// Whisper preprocessor base class.
/// </summary>
public abstract class WhisperPreprocessor : Preprocessor
{
    /// <summary>
    /// Constants used in the Whisper preprocessing.
    /// </summary>
    protected const int ChunkLength = 30;
    protected const int SampleRate = 16000;
    protected const int WindowLength = 400;
    protected const int HopLength = 160;

    /// <summary>
    /// Initializes a new instance of the <see cref="WhisperPreprocessor"/> class.
    /// </summary>
    /// <param name="name">The preprocessor name.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    protected WhisperPreprocessor(string name, OnnxSessionOptions onnxOptions)
        : base(name, onnxOptions)
    {
    }
}

/// <summary>
/// LogMelSpectrogram feature extractor for Whisper models with 80 mel frequency bands.
/// </summary>
public class Whisper80Preprocessor : WhisperPreprocessor
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Whisper80Preprocessor"/> class.
    /// </summary>
    /// <param name="onnxOptions">The ONNX options.</param>
    public Whisper80Preprocessor(OnnxSessionOptions onnxOptions)
        : base("WhisperPreprocessor80", onnxOptions)
    {
    }
}

/// <summary>
/// LogMelSpectrogram feature extractor for Whisper models with 128 mel frequency bands.
/// </summary>
public class Whisper128Preprocessor : WhisperPreprocessor
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Whisper128Preprocessor"/> class.
    /// </summary>
    /// <param name="onnxOptions">The ONNX options.</param>
    public Whisper128Preprocessor(OnnxSessionOptions onnxOptions)
        : base("WhisperPreprocessor128", onnxOptions)
    {
    }
}