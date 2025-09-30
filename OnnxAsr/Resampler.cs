using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxAsr;

/// <summary>
/// Waveform resampler to 16 kHz implementation.
/// </summary>
public class Resampler : IDisposable
{
    private readonly Dictionary<int, InferenceSession> _preprocessors = new();
    private readonly int[] _supportedSampleRates = new[] { 8000, 32000, 44100, 48000 };
    private const int TargetSampleRate = 16000;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="Resampler"/> class.
    /// </summary>
    /// <param name="onnxOptions">The ONNX options.</param>
    public Resampler(OnnxSessionOptions onnxOptions)
    {
        foreach (var sampleRate in _supportedSampleRates)
        {
            var modelName = $"resample{sampleRate / 1000}.onnx";
            var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "preprocessors", modelName);

            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Resampler model {modelName} not found at {modelPath}");
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

            _preprocessors[sampleRate] = new InferenceSession(modelPath, sessionOptions);
        }
    }

    /// <summary>
    /// Resample waveform to 16 kHz.
    /// </summary>
    /// <param name="waveforms">The input waveforms.</param>
    /// <param name="waveformsLens">The input waveforms lengths.</param>
    /// <param name="sampleRate">The input sample rate.</param>
    /// <returns>The resampled waveforms and lengths.</returns>
    public (Tensor<float>, Tensor<int>) Process(Tensor<float> waveforms, Tensor<int> waveformsLens, int sampleRate)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(Resampler));
        }

        if (sampleRate == TargetSampleRate)
        {
            return (waveforms, waveformsLens);
        }

        if (!_preprocessors.TryGetValue(sampleRate, out var preprocessor))
        {
            throw new ArgumentException($"Unsupported sample rate: {sampleRate}. Supported rates: {string.Join(", ", _supportedSampleRates)}");
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("waveforms", waveforms),
            OnnxSessionUtils.CreateLengthInput(preprocessor, "waveforms_lens", waveformsLens)
        };

        using var results = preprocessor.Run(inputs);
        var resampled = results.First(r => r.Name == "resampled").AsTensor<float>();
        var resampledLensLong = results.First(r => r.Name == "resampled_lens").AsTensor<long>();
        var resampledLens = new DenseTensor<int>(resampledLensLong.Dimensions);
        var resampledLensData = resampledLensLong.ToArray();
        for (int i = 0; i < resampledLensData.Length; i++)
        {
            resampledLens[i] = (int)resampledLensData[i];
        }

        return (resampled, resampledLens);
    }

    /// <summary>
    /// Disposes the resampler.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes the resampler.
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
            foreach (var preprocessor in _preprocessors.Values)
            {
                preprocessor.Dispose();
            }
            _preprocessors.Clear();
        }

        _disposed = true;
    }
}