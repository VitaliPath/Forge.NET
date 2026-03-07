using Xunit;
using Forge.Graph;
using Forge.Graph.Algorithms;

namespace Forge.Tests;

public class DependencyDepthTests
{
    [Fact]
    public void Calculate_Respects_Associative_Penalty()
    {
        // Arrange
        var graph = new Graph<string>();
        graph.AddNode("A", "root");
        graph.AddNode("B", "child");
        graph.AddNode("C", "reference");

        // A --(Structural)--> B  [Depth 1]
        graph.AddEdge("A", "B", 1.0f, RelationshipType.Structural);
        
        // A --(Associative)--> C [Depth 3 if penalty is 3]
        graph.AddEdge("A", "C", 1.0f, RelationshipType.Associative);

        var csr = graph.CompileCsr();
        int rootIdx = csr.IdToIndex["A"];

        // Act
        int[] depths = csr.CalculateDependencyDepth(new[] { rootIdx }, associativePenalty: 3);

        // Assert
        Assert.Equal(0, depths[csr.IdToIndex["A"]]);
        Assert.Equal(1, depths[csr.IdToIndex["B"]]);
        Assert.Equal(3, depths[csr.IdToIndex["C"]]);
    }

    [Fact]
    public void Calculate_Ignores_Associative_When_Penalty_Is_Negative()
    {
        var graph = new Graph<string>();
        graph.AddNode("A", "root");
        graph.AddNode("B", "assoc");
        graph.AddEdge("A", "B", 1.0f, RelationshipType.Associative);

        var csr = graph.CompileCsr();
        
        // Act: Set penalty to -1 (Structural Only)
        int[] depths = csr.CalculateDependencyDepth(new[] { csr.IdToIndex["A"] }, associativePenalty: -1);

        // Assert: B should be unreachable
        Assert.Equal(-1, depths[csr.IdToIndex["B"]]);
    }
}