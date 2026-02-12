using BenchmarkDotNet.Attributes;
using Forge.Graph;
using Forge.Graph.Algorithms;

namespace Forge.Benchmarks;

[MemoryDiagnoser]
public class GraphBenchmarks
{
    private Graph<string> _graph = null!;
    private GraphCsr _csr;
    private ConnectedComponents<string> _algo = null!;

    [Params(1000)] // Number of nodes
    public int NodeCount;

    [Params(10)] // Average degree per node
    public int EdgeDegree;

    [GlobalSetup]
    public void Setup()
    {
        _graph = new Graph<string>();
        _algo = new ConnectedComponents<string>();

        // Generate a synthetic linked-chain/mesh graph
        for (int i = 0; i < NodeCount; i++)
        {
            _graph.AddNode(i.ToString(), $"data_{i}");
        }

        for (int i = 0; i < NodeCount; i++)
        {
            for (int j = 1; j <= EdgeDegree; j++)
            {
                int target = (i + j) % NodeCount;
                _graph.AddEdge(i.ToString(), target.ToString(), 1.0);
            }
        }

        // Compile once for the CSR benchmark
        _csr = _graph.CompileCsr();
    }

    [Benchmark(Baseline = true)]
    public void StandardGraph_ConnectedComponents()
    {
        _algo.Execute(_graph);
    }

    [Benchmark]
    public void CsrGraph_ConnectedComponents()
    {
        _algo.Execute(_csr);
    }
}