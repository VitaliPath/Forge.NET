using Xunit;
using Forge.Graph;
using Forge.Graph.Algorithms;

namespace Forge.Tests;

public class FlowRankingTests
{
    [Fact]
    public void CalculateFlowRank_Orders_3Tier_Pipeline_Correctly()
    {
        // Arrange: Ingestion -> Intelligence -> UI
        var graph = new Graph<string>();
        graph.AddNode("Ingest", "Source");
        graph.AddNode("Intel", "Processing");
        graph.AddNode("UI", "Sink");

        graph.AddEdge("Ingest", "Intel", 1.0f, RelationshipType.Structural);
        graph.AddEdge("Intel", "UI", 1.0f, RelationshipType.Structural);

        var csr = graph.CompileCsr();
        var subset = new[] { csr.IdToIndex["UI"], csr.IdToIndex["Intel"], csr.IdToIndex["Ingest"] };

        // Act
        var ranks = csr.CalculateFlowRank(subset);

        // Assert
        Assert.Equal(0, ranks[2]); // Ingest (index 2 in subset)
        Assert.Equal(1, ranks[1]); // Intel (index 1 in subset)
        Assert.Equal(2, ranks[0]); // UI (index 0 in subset)
    }

[Fact]
    public void Graph_Prevents_Circular_Structural_Dependencies()
    {
        // Arrange: A -> B
        var graph = new Graph<string>();
        graph.AddNode("A", "n");
        graph.AddNode("B", "n");
        graph.AddEdge("A", "B", 1.0f, RelationshipType.Structural);

        // Act & Assert: B -> A (Illegal)
        // This confirms the FORGE-065 invariant is active.
        var exception = Assert.Throws<InvalidOperationException>(() => 
            graph.AddEdge("B", "A", 1.0f, RelationshipType.Structural));
            
        Assert.Contains("Circular structural dependency", exception.Message);
    }

    [Fact]
    public void CalculateFlowRank_Ignores_Associative_Bridges()
    {
        // Arrange: Root -> Child, but Child --(Associative)--> Independent
        var graph = new Graph<string>();
        graph.AddNode("Root", "r");
        graph.AddNode("Child", "c");
        graph.AddNode("Indie", "i");

        graph.AddEdge("Root", "Child", 1.0f, RelationshipType.Structural);
        graph.AddEdge("Child", "Indie", 1.0f, RelationshipType.Associative);

        var csr = graph.CompileCsr();
        var subset = new[] { 0, 1, 2 };

        // Act
        var ranks = csr.CalculateFlowRank(subset);

        // Assert: Indie should be Rank 0 because the link from Child is non-structural
        Assert.Equal(0, ranks[csr.IdToIndex["Root"]]);
        Assert.Equal(1, ranks[csr.IdToIndex["Child"]]);
        Assert.Equal(0, ranks[csr.IdToIndex["Indie"]]);
    }
}