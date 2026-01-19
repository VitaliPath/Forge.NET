using Forge.Core;

namespace Forge.Neural;

public class Relu : IModule
{
    public Tensor Forward(Tensor input)
    {
        return input.Relu();
    }

    public List<Tensor> Parameters()
    {
        return new List<Tensor>();
    }
}