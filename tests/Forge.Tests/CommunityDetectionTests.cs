using Xunit;
using Forge.Graph;
using Forge.Graph.Algorithms;
using System.Collections.Generic;
using System.Linq;

namespace Forge.Tests
{
    public class CommunityDetectionTests
    {
        [Fact]
        public void Detects_Disconnected_Islands()
        {
            // Arrange: Create a graph with two separate islands
            var graph = new Graph<string>();

            // Island 1: A-B
            // Fix: Pass "A" as both ID and Data. 
            // Fix: AddEdge takes IDs (strings), not Node objects.
            graph.AddNode("A", "A");
            graph.AddNode("B", "B");
            graph.AddEdge("A", "B", 1.0f);

            // Island 2: C-D-E
            graph.AddNode("C", "C");
            graph.AddNode("D", "D");
            graph.AddNode("E", "E");
            graph.AddEdge("C", "D", 1.0f);
            graph.AddEdge("D", "E", 1.0f);

            // Act: Run detection
            var detector = new ConnectedComponents<string>(); // Namespace match
            List<List<Node<string>>> communities = detector.Execute(graph);

            // Assert
            Assert.Equal(2, communities.Count);
            
            // Verify content of islands
            var island1 = communities.First(c => c.Any(n => n.Data == "A"));
            Assert.Contains(island1, n => n.Data == "B");
            Assert.DoesNotContain(island1, n => n.Data == "C");

            var island2 = communities.First(c => c.Any(n => n.Data == "C"));
            Assert.Equal(3, island2.Count);
        }

        [Fact]
        public void Single_Node_Is_Community()
        {
            var graph = new Graph<string>();
            graph.AddNode("Loner", "Loner");

            var detector = new ConnectedComponents<string>();
            var result = detector.Execute(graph);

            Assert.Single(result);
            Assert.Equal("Loner", result[0][0].Data);
        }

        [Fact]
        public void Predicate_Threshold_Splits_Weak_Bridges()
        {
            // Arrange: Create two islands connected by a weak "Noise" edge
            var graph = new Graph<string>();

            // Island 1 (Strongly linked)
            graph.AddNode("A", "A");
            graph.AddNode("B", "B");
            graph.AddEdge("A", "B", 1.0f);

            // Island 2 (Strongly linked)
            graph.AddNode("C", "C");
            graph.AddNode("D", "D");
            graph.AddEdge("C", "D", 1.0f);

            // The Bridge (Weak noise link)
            graph.AddEdge("B", "C", 0.05f);

            var detector = new ConnectedComponents<string>();

            // Act 1: Standard binary BFS (should find 1 giant component)
            var giantResult = detector.Execute(graph);

            // Act 2: Weighted BFS with threshold > 0.1 (should find 2 islands)
            var splitResult = detector.Execute(graph, e => e.Weight > 0.1);

            // Assert
            Assert.Single(giantResult); // The bridge holds in binary mode
            Assert.Equal(2, splitResult.Count); // The bridge snaps at threshold

            // Verify content of split islands
            var island1 = splitResult.First(c => c.Any(n => n.Id == "A"));
            Assert.Contains(island1, n => n.Id == "B");
            Assert.DoesNotContain(island1, n => n.Id == "C");
        }

        [Fact]
        public void ParallelDSU_Maintains_Parity_With_Standard_DFS()
        {
            // Arrange
            var graph = new Graph<string>();
            for (int i = 0; i < 1000; i++) graph.AddNode(i.ToString(), "data");
            for (int i = 0; i < 999; i++) graph.AddEdge(i.ToString(), (i + 1).ToString(), 1.0f);

            var csr = graph.CompileCsr();
            var sequential = new ConnectedComponents<string>();
            var parallel = new ParallelConnectedComponents<string>();

            // Act
            var seqResult = sequential.Execute(csr);
            var parResult = parallel.Execute(csr);

            // Assert
            Assert.Single(seqResult);
            Assert.Single(parResult);
            Assert.Equal(seqResult[0].Count, parResult[0].Count);
        }
    }
}