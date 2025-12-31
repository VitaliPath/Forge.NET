namespace Forge.Core;

public class Value
{
    public double Data;
    public double Grad;
    public string Label;

    private HashSet<Value> _prev;
    private Action _backward;

    public Value(double data, string label = "", IEnumerable<Value>? children = null)
    {
        Data = data;
        Label = label;
        Grad = 0.0;
        _prev = children != null ? new HashSet<Value>(children) : new HashSet<Value>();
        _backward = () => { };
    }

    public static Value operator +(Value a, Value b)
    {
        var output = new Value(a.Data + b.Data, "", new List<Value> { a, b });
        output._backward = () =>
        {
            a.Grad += 1.0 * output.Grad;
            b.Grad += 1.0 * output.Grad;
        };

        return output;
    }

    public static Value operator *(Value a, Value b)
    {
        var output = new Value(a.Data * b.Data, "", new List<Value> { a, b });
        output._backward = () =>
        {
            a.Grad += b.Data * output.Grad;
            b.Grad += a.Data * output.Grad;
        };

        return output;
    }

    public Value Tanh()
    {
        var x = this.Data;
        var t = Math.Tanh(x);
        var output = new Value(t, "tanh", new List<Value> { this });
        
        output._backward = () =>
        {
            var localGrad = (1.0 - t*t);
            this.Grad += localGrad * output.Grad;
        };

        return output;
    }

    public void Backward()
    {
        var visited = new HashSet<Value>();
        var topo = new List<Value>();

        void Build(Value node)
        {
            if (visited.Contains(node))
            {
                return;
            }

            visited.Add(node);
            foreach (var child in node._prev)
            {
                Build(child);
            }

            topo.Add(node);
        }

        this.Grad = 1.0;
        Build(this);

        topo.Reverse();

        foreach (var node in topo)
        {
            node._backward();
        }
    }
}
