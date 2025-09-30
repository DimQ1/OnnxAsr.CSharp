namespace OnnxAsr;

/// <summary>
/// Timestamped recognition result.
/// </summary>
public class TimestampedResult
{
    /// <summary>
    /// Gets or sets the recognized text.
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamps.
    /// </summary>
    public float[]? Timestamps { get; set; }

    /// <summary>
    /// Gets or sets the tokens.
    /// </summary>
    public string[]? Tokens { get; set; }
}