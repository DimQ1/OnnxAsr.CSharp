using System.Text.Json.Serialization;

namespace OnnxAsr;

/// <summary>
/// Config for ASR model.
/// </summary>
public class AsrConfig
{
    /// <summary>
    /// Gets or sets the model type.
    /// </summary>
    [JsonPropertyName("model_type")]
    public string? ModelType { get; set; }

    /// <summary>
    /// Gets or sets the features size.
    /// </summary>
    [JsonPropertyName("features_size")]
    public int? FeaturesSize { get; set; }

    /// <summary>
    /// Gets or sets the subsampling factor.
    /// </summary>
    [JsonPropertyName("subsampling_factor")]
    public int? SubsamplingFactor { get; set; }

    /// <summary>
    /// Gets or sets the max tokens per step.
    /// </summary>
    [JsonPropertyName("max_tokens_per_step")]
    public int? MaxTokensPerStep { get; set; }
}