using Xunit;
using Forge.Graph;
using Forge.Graph.Algorithms;
using System.Diagnostics;
using System.Linq;

namespace Forge.Tests;

public class CentralityTests
{
    [Fact]
    public void GodObject_Detection_Exceeds_Three_Sigma()
    {
        // Arrange: Create 1 'God' node and 99 'Leaf' nodes
        var graph = new Graph<string>();
        string godId = "AnalystConsole";
        graph.AddNode(godId, "The Hub");

        for (int i = 0; i < 99; i++)
        {
            string leafId = $"Leaf_{i}";
            graph.AddNode(leafId, "Worker");
            // Every leaf depends on the God Object
            graph.AddEdge(leafId, godId, 1.0f); 
            // God Object depends on every leaf (Bidirectional)
            graph.AddEdge(godId, leafId, 1.0f);
        }

        var csr = graph.CompileCsr();

        // Act
        float[] scores = csr.CalculateCentrality();

        // Assert
        int godIdx = csr.IdToIndex[godId];
        float godScore = scores[godIdx];

        // Calculate Mean and StdDev of the leaf scores
        var leafScores = scores.Where((s, i) => i != godIdx).ToArray();
        double mean = leafScores.Average();
        double sumSqDiff = leafScores.Sum(s => Math.Pow(s - mean, 2));
        double stdDev = Math.Sqrt(sumSqDiff / leafScores.Length);

        // Verification: God score must be > 3 sigma above the leaf mean
        double threshold = mean + (3 * stdDev);
        
        Assert.True(godScore > threshold, 
            $"God Object gravity ({godScore:F4}) was not significantly higher than leaf mean ({mean:F4} + 3σ).");
    }

    [Fact]
    public void Performance_Benchmark_LargeGraph_Under_75ms()
    {
        // Arrange: 10k nodes, 100k edges (Average degree 10)
        var graph = new Graph<string>();
        int nodeCount = 10000;
        int edgeCount = 100000;

        for (int i = 0; i < nodeCount; i++) graph.AddNode(i.ToString(), "data");
        
        var rnd = new Random(42);
        for (int i = 0; i < edgeCount; i++)
        {
            graph.AddEdge(
                rnd.Next(nodeCount).ToString(), 
                rnd.Next(nodeCount).ToString(), 
                1.0f);
        }

        var csr = graph.CompileCsr();
        var sw = new Stopwatch();

        // Act
        sw.Start();
        var scores = csr.CalculateCentrality();
        sw.Stop();

        // Assert
        Assert.Equal(nodeCount, scores.Length);
        Assert.True(sw.ElapsedMilliseconds < 75, 
            $"Performance regression: Kernel took {sw.ElapsedMilliseconds}ms (Target: <75ms).");
    }
}