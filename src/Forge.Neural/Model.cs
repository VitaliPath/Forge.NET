using Forge.Core;

namespace Forge.Neural;

public interface IModelLayer
{
    Value[] Forward(Value[] input);
    List<Value> Parameters();
}

public class Model : IModelLayer
{
    private List<IModelLayer> _layers = new List<IModelLayer>();

    public void Add(IModelLayer layer)
    {
        _layers.Add(layer);
    }

    public Value[] Forward(Value[] input)
    {
        var current = input;
        for (var i = 0; i < _layers.Count(); i++)
        {
            current = _layers[i].Forward(current);
        }

        return current;
    }

    public List<Value> Parameters()
    {
        var paramsList = new List<Value>();
        foreach (var layer in _layers)
        {
            paramsList.AddRange(layer.Parameters());
        }
        return paramsList;
    }
}
