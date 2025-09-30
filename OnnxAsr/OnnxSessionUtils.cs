using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxAsr;

internal static class OnnxSessionUtils
{
    // Creates a NamedOnnxValue for a length tensor, matching the model input's element type
    public static NamedOnnxValue CreateLengthInput(InferenceSession session, string inputName, Tensor<int> lengths)
    {
        var inputMeta = session.InputMetadata[inputName];
        var elemType = inputMeta.ElementType; // CLR Type (e.g., typeof(System.Int64) or typeof(System.Int32))

        if (elemType == typeof(long) || elemType == typeof(System.Int64))
        {
            var data = lengths.ToArray();
            var longData = Array.ConvertAll(data, i => (long)i);
            return NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<long>(longData, lengths.Dimensions));
        }
        else if (elemType == typeof(int) || elemType == typeof(System.Int32))
        {
            return NamedOnnxValue.CreateFromTensor(inputName, lengths);
        }
        else
        {
            throw new InvalidOperationException($"Unsupported length element type: {elemType}");
        }
    }
}
