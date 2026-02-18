namespace Forge.Graph.Algorithms;

public interface IConnectedComponents<T>
{
    List<List<string>> Execute(GraphCsr csr, Func<int, double, bool>? predicate = null);
}