using Forge.Core;

namespace Forge.Neural;

public class Neuron
{
    private List<Value> Weights;
    private Value Bias;

    public Neuron(int numInputs)
    {
        var rng = new Random();
        Weights = new List<Value>(numInputs);
        Bias = new Value(rng.NextDouble() * 2 - 1);
        for (var i = 0; i < numInputs; i++)
        {
            var randomValue = rng.NextDouble() * 2 - 1;
            Weights.Add(new Value(randomValue));
        }
    }

    public Value Forward(Value[] inputs)
    {
        var Sum = new Value(0.0, "sum0");
        for (var i = 0; i < Weights.Count; i++)
        {
            Sum += Weights[i] * inputs[i];
        }

        Sum += Bias;
        return Sum;
    }

    public List<Value> Parameters()
    {
        var p = new List<Value>();
        p.AddRange(Weights);
        p.Add(Bias);
        return p;
    }
}
