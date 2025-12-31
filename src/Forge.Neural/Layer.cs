using Forge.Core;

namespace Forge.Neural;

public class Layer : IModelLayer
{
    public List<Neuron> Neurons { get; }

    public Layer(int nin, int nout)
    {
        Neurons = new List<Neuron>(nout);
        for (int i = 0; i < nout; i++)
        {
            Neurons.Add(new Neuron(nin));
        }
    }

    public Value[] Forward(Value[] inputs)
    {
        var output = new Value[Neurons.Count];
        for (int i = 0; i < Neurons.Count; i++)
        {
            output[i] = Neurons[i].Forward(inputs);
        }
        return output;
    }

    public List<Value> Parameters()
    {
        var paramsList = new List<Value>();
        foreach (var neuron in Neurons)
        {
            paramsList.AddRange(neuron.Parameters());
        }
        return paramsList;
    }
}
