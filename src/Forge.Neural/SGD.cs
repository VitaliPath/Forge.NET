using Forge.Core;

namespace Forge.Neural;

public class SGD
{
    private double _lr;

    public SGD(double learningRate)
    {
        _lr = learningRate;
    }

    public void Step(List<Tensor> parameters)
    {
        foreach (var p in parameters)
        {
            for(int i=0; i<p.Data.Length; i++)
            {
                p.Data[i] -= _lr * p.Grad[i];
            }
        }
    }

    public void ZeroGrad(List<Tensor> parameters)
    {
        foreach (var p in parameters)
        {
            Array.Clear(p.Grad);
        }
    }
}