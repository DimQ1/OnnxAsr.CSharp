using System.Reflection;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxAsr;

/// <summary>
/// ASR preprocessor base class.
/// </summary>
public class Preprocessor : IDisposable
{
    private readonly InferenceSession _session;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="Preprocessor"/> class.
    /// </summary>
    /// <param name="name">The preprocessor name.</param>
    /// <param name="onnxOptions">The ONNX options.</param>
    public Preprocessor(string name, OnnxSessionOptions onnxOptions)
    {
        var modelName = $"{name}.onnx";
        var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "preprocessors", modelName);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Preprocessor model {modelName} not found at {modelPath}");
        }

        var sessionOptions = new SessionOptions();
        if (onnxOptions?.SessionOptions != null)
        {
            sessionOptions = onnxOptions.SessionOptions;
        }

        if (onnxOptions?.CpuPreprocessing == true)
        {
            onnxOptions = new OnnxSessionOptions { SessionOptions = onnxOptions.SessionOptions };
        }

        _session = new InferenceSession(modelPath, sessionOptions);
    }

    /// <summary>
    /// Processes the waveforms.
    /// </summary>
    /// <param name="waveforms">The waveforms.</param>
    /// <param name="waveformsLens">The waveforms lengths.</param>
    /// <returns>The features and lengths.</returns>
    public (Tensor<float>, Tensor<int>) Process(Tensor<float> waveforms, Tensor<int> waveformsLens)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(Preprocessor));
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("waveforms", waveforms),
            OnnxSessionUtils.CreateLengthInput(_session, "waveforms_lens", waveformsLens)
        };

        using var results = _session.Run(inputs);
        var features = results.First(r => r.Name == "features").AsTensor<float>();
        var featuresLensLong = results.First(r => r.Name == "features_lens").AsTensor<long>();
        var featuresLens = new DenseTensor<int>(featuresLensLong.Dimensions);
        var featuresLensData = featuresLensLong.ToArray();
        for (int i = 0; i < featuresLensData.Length; i++)
        {
            featuresLens[i] = (int)featuresLensData[i];
        }

        return (features, featuresLens);
    }

    /// <summary>
    /// Disposes the preprocessor.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes the preprocessor.
    /// </summary>
    /// <param name="disposing">True if disposing.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
        {
            return;
        }

        if (disposing)
        {
            _session?.Dispose();
        }

        _disposed = true;
    }
}