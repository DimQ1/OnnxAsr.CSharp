namespace OnnxAsr;

/// <summary>
/// NeMo preprocessor - LogMel feature extractor with normalization.
/// </summary>
public class NemoPreprocessor : Preprocessor
{
    /// <summary>
    /// Initializes a new instance of the <see cref="NemoPreprocessor"/> class.
    /// </summary>
    /// <param name="onnxOptions">The ONNX options.</param>
    public NemoPreprocessor(OnnxSessionOptions onnxOptions)
        : base("Nemo", onnxOptions)
    {
    }
}