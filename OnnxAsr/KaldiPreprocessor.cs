namespace OnnxAsr;

/// <summary>
/// Kaldi preprocessor - Fbank feature extractor with delta and delta-delta features.
/// </summary>
public class KaldiPreprocessor : Preprocessor
{
    /// <summary>
    /// Initializes a new instance of the <see cref="KaldiPreprocessor"/> class.
    /// </summary>
    /// <param name="onnxOptions">The ONNX options.</param>
    public KaldiPreprocessor(OnnxSessionOptions onnxOptions)
        : base("Kaldi", onnxOptions)
    {
    }
}