using Forge.Core;

namespace Forge.Neural;

public class Linear : IModule
{
    private Tensor Weight;
    private Tensor Bias;

    public Linear(int nin, int nout, int seed = 0, bool useHeInit = true) // Add flag
    {
        Weight = Tensor.Random(nin, nout, seed);
        
        // He Init (for ReLU) vs Xavier Init (for Tanh)
        double factor = useHeInit ? 2.0 : 1.0; 
        double scale = Math.Sqrt(factor / nin); 
        
        for(int i=0; i < Weight.Data.Length; i++) Weight.Data[i] *= scale;

        Bias = Tensor.Zeros(1, nout);
    }

    public Tensor Forward(Tensor input)
    {
        var outMat = input.MatMul(Weight);
        return outMat + Bias;
    }

    public List<Tensor> Parameters()
    {
        return new List<Tensor> { Weight, Bias };
    }
}