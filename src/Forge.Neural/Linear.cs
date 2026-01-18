using Forge.Core;

namespace Forge.Neural;

public class Linear : IModule
{
    private Tensor Weight;
    private Tensor Bias;

    public Linear(int nin, int nout, int seed = 0)
    {
        Weight = Tensor.Random(nin, nout, seed);
        
        double scale = 1.0 / Math.Sqrt(nin);
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