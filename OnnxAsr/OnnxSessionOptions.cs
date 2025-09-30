using Microsoft.ML.OnnxRuntime;

namespace OnnxAsr;

/// <summary>
/// Options for ONNX runtime InferenceSession.
/// </summary>
public class OnnxSessionOptions
{
    /// <summary>
    /// Gets or sets the session options.
    /// </summary>
    public SessionOptions? SessionOptions { get; set; }

    /// <summary>
    /// Gets or sets the providers.
    /// </summary>
    public string[]? Providers { get; set; }

    /// <summary>
    /// Gets or sets the provider options.
    /// </summary>
    public ProviderOptions[]? ProviderOptions { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether to use CPU preprocessing.
    /// </summary>
    public bool CpuPreprocessing { get; set; }
}

/// <summary>
/// Provider options.
/// </summary>
public class ProviderOptions
{
    /// <summary>
    /// Gets or sets the provider name.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the options.
    /// </summary>
    public Dictionary<string, string> Options { get; set; } = new();
}