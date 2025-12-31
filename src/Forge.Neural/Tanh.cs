using Forge.Core;

namespace Forge.Neural;

public class Tanh : IModelLayer
{
    public Value[] Forward(Value[] input)
    {
        var output = new Value[input.Length];
        for (var i = 0; i < input.Length; i++)
        {
            output[i] = input[i].Tanh();
        }
        return output;
    }

    public List<Value> Parameters()
    {
        return new List<Value>();
    }
}
