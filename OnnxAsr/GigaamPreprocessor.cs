namespace OnnxAsr;

/// <summary>
/// GigaAM preprocessor - Fbank feature extractor for GigaSpeech.
/// </summary>
public class GigaamPreprocessor : Preprocessor
{
    /// <summary>
    /// Initializes a new instance of the <see cref="GigaamPreprocessor"/> class.
    /// </summary>
    /// <param name="onnxOptions">The ONNX options.</param>
    public GigaamPreprocessor(OnnxSessionOptions onnxOptions)
        : base("Gigaam", onnxOptions)
    {
    }
}