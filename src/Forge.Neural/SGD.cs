using Forge.Core;

namespace Forge.Neural;

public interface IOptimizer
{
    void Step(IEnumerable<Value> parameters);
    void ZeroGrad(IEnumerable<Value> parameters);
}

public class SGD : IOptimizer
{
    private double _learningRate;

    public SGD(double learningRate)
    {
        _learningRate = learningRate;
    }

    public void Step(IEnumerable<Value> parameters)
    {
        foreach (var p in parameters)
        {
            p.Data -= p.Grad * _learningRate;
        }
    }

    public void ZeroGrad(IEnumerable<Value> parameters)
    {
        foreach (var p in parameters)
        {
            p.Grad = 0.0;
        }
    }
}
