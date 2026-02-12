using Xunit;
using Forge.Graph;
using System.Linq;
using System.Threading.Tasks;
using Forge.Graph.Algorithms;

namespace Forge.Tests
{
    public class GraphTests
    {
        [Fact]
        public void AccumulateEdgeWeight_Correctly_Sums_Weights()
        {
            // Arrange
            var graph = new Graph<string>();
            graph.AddNode("file_a", "content_a");
            graph.AddNode("file_b", "content_b");

            // Act
            graph.AccumulateEdgeWeight("file_a", "file_b", 0.5);
            graph.AccumulateEdgeWeight("file_a", "file_b", 0.5);

            // Assert
            var source = graph.GetNode("file_a");
            var edge = source.Neighbors.First(e => e.Target.Id == "file_b");
            
            Assert.Equal(1.0, edge.Weight, precision: 5);
            Assert.Single(source.Neighbors); // Should NOT have created a second edge
        }

        [Fact]
        public async Task AccumulateEdgeWeight_Is_ThreadSafe()
        {
            // Arrange
            var graph = new Graph<string>();
            graph.AddNode("source", "src");
            graph.AddNode("target", "dest");
            int iterations = 1000;

            // Act: Fire 1000 increments of 1.0 in parallel
            await Task.Run(() => Parallel.For(0, iterations, _ =>
            {
                graph.AccumulateEdgeWeight("source", "target", 1.0);
            }));

            // Assert
            var weight = graph.GetNode("source").Neighbors.First().Weight;
            Assert.Equal(1000.0, weight);
        }

        [Fact]
        public void AccumulateEdgeWeight_Maintains_Most_Recent_Timestamp()
        {
            // Arrange
            var graph = new Graph<string>();
            graph.AddNode("file_a", "data_a");
            graph.AddNode("file_b", "data_b");

            // Act: Initial reinforcement at Time 1000
            graph.AccumulateEdgeWeight("file_a", "file_b", 1.0, 1000);

            // Act: Reinforcement from an "older" commit (Time 500)
            graph.AccumulateEdgeWeight("file_a", "file_b", 1.0, 500);

            // Assert
            var source = graph.GetNode("file_a");
            var edge = source.Neighbors.First(e => e.Target.Id == "file_b");

            Assert.Equal(2.0, edge.Weight);

            Assert.Equal(1000, edge.LastModified);
        }

        [Fact]
        public void Csr_Compilation_And_Traversal_Parity()
        {
            // Arrange
            var graph = new Graph<string>();
            graph.AddNode("A", "data");
            graph.AddNode("B", "data");
            graph.AddNode("C", "data");

            // This creates TWO directed edges: A->B and B->A
            graph.AddEdge("A", "B", 1.0);

            // Act
            var csr = graph.CompileCsr();
            var detector = new ConnectedComponents<string>();

            // Traversal 1: Standard (returns List<List<Node<string>>>)
            var standardResult = detector.Execute(graph);

            // Traversal 2: CSR (returns List<List<string>>)
            var csrResult = detector.Execute(csr);

            // Assert
            Assert.Equal(2, csr.EdgeCount);
            Assert.Equal(4, csr.RowPtr.Length); // |V| + 1 = 3 + 1

            // Parity check: Both should find 2 islands
            Assert.Equal(standardResult.Count, csrResult.Count);

            // Deep parity: Ensure the island content matches
            var standardIds = standardResult
                .Select(island => island.Select(n => n.Id).OrderBy(id => id).ToList())
                .OrderBy(island => island.First())
                .ToList();

            var csrIds = csrResult
                .Select(island => island.OrderBy(id => id).ToList())
                .OrderBy(island => island.First())
                .ToList();

            for (int i = 0; i < standardIds.Count; i++)
            {
                Assert.Equal(standardIds[i], csrIds[i]);
            }
        }

        [Fact]
        public void CompileCsr_Handles_Nodes_With_Zero_Edges()
        {
            // Arrange
            var graph = new Graph<string>();
            graph.AddNode("Loner", "I have no friends");

            // Act
            var csr = graph.CompileCsr();

            // Assert
            Assert.Equal(1, csr.NodeCount);
            Assert.Equal(0, csr.EdgeCount);
            Assert.Equal(0, csr.ColIdx.Length);
            Assert.Equal(0, csr.RowPtr[0]);
            Assert.Equal(0, csr.RowPtr[1]); // RowPtr[i] == RowPtr[i+1] means 0 degree
        }
    }
}