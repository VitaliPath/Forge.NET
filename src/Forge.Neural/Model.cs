using Forge.Core;

namespace Forge.Neural;

public interface IModule
{
    Tensor Forward(Tensor input);
    List<Tensor> Parameters();
}

public class Sequential : IModule
{
    private List<IModule> _layers = new List<IModule>();

    public void Add(IModule layer)
    {
        _layers.Add(layer);
    }

    public Tensor Forward(Tensor input)
    {
        var current = input;
        foreach (var layer in _layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    public List<Tensor> Parameters()
    {
        var paramsList = new List<Tensor>();
        foreach (var layer in _layers)
        {
            paramsList.AddRange(layer.Parameters());
        }
        return paramsList;
    }
}