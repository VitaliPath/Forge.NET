using Forge.Core;

namespace Forge.Neural;

public class Tanh : IModule
{
    public Tensor Forward(Tensor input)
    {
        return input.Tanh();
    }

    public List<Tensor> Parameters()
    {
        return new List<Tensor>();
    }
}