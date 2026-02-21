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
            Assert.Empty(csr.ColIdx);
            Assert.Equal(0, csr.RowPtr[0]);
            Assert.Equal(0, csr.RowPtr[1]); // RowPtr[i] == RowPtr[i+1] means 0 degree
        }

        [Fact]
        public async Task AccumulateEdgeWeight_Prevents_Deadlock_On_Bidirectional_Updates()
        {
            // Arrange
            var graph = new Graph<string>();
            graph.AddNode("A", "DataA");
            graph.AddNode("B", "DataB");
            int iterations = 10000;

            // Act: Two tasks updating the same edge from opposite 'directions'
            var task1 = Task.Run(() =>
            {
                for (int i = 0; i < iterations; i++)
                    graph.AccumulateEdgeWeight("A", "B", 1.0);
            });

            var task2 = Task.Run(() =>
            {
                for (int i = 0; i < iterations; i++)
                    graph.AccumulateEdgeWeight("B", "A", 1.0);
            });

            // Assert: Use WaitAsync to ensure the test fails if it hangs for > 5 seconds.
            await Task.WhenAll(task1, task2).WaitAsync(TimeSpan.FromSeconds(5));

            var edgeAB = graph.GetNode("A").Neighbors.First(e => e.Target.Id == "B");
            var edgeBA = graph.GetNode("B").Neighbors.First(e => e.Target.Id == "A");

            // Total weight should be 20,000 (10k from task1 + 10k from task2 bidirectional logic)
            Assert.Equal(20000.0, edgeAB.Weight);
            Assert.Equal(20000.0, edgeBA.Weight);
        }

        [Fact]
        public void RemoveNode_Performs_Symmetric_Cleanup()
        {
            // Arrange
            var graph = new Graph<string>();
            graph.AddNode("A", "DataA");
            graph.AddNode("B", "DataB");
            graph.AddEdge("A", "B", 1.0);

            // Act
            bool removed = graph.RemoveNode("A");

            // Assert
            Assert.True(removed);
            Assert.DoesNotContain("A", graph.GetAllIds());
            
            var nodeB = graph.GetNode("B");
            Assert.Empty(nodeB.Neighbors); // Symmetric snip verified
        }

        [Fact]
        public void RemoveNode_Maintains_CSR_Integrity()
        {
            // Arrange
            var graph = new Graph<string>();
            graph.AddNode("A", "A");
            graph.AddNode("B", "B");
            graph.AddNode("C", "C");
            graph.AddEdge("A", "B", 1.0);
            graph.AddEdge("B", "C", 1.0);

            // Act: Remove the central pivot
            graph.RemoveNode("B");

            // Assert: CSR compilation should not find orphans
            var csr = graph.CompileCsr();
            Assert.Equal(2, csr.NodeCount); // A and C remain
            Assert.Equal(0, csr.EdgeCount); // All edges touched B
        }

        [Fact]
        public void ApplyDecay_Correctly_Ages_Weights()
        {
            // Arrange: 138.6 days is exactly 1 half-life for lambda 0.005
            var graph = new Graph<string>();
            graph.AddNode("A", "A");
            graph.AddNode("B", "B");

            long now = 200000;
            long then = now - (long)(138.629 * 86400); // 1 half-life ago

            graph.AccumulateEdgeWeight("A", "B", 10.0, then);

            // Act
            graph.ApplyDecay(0.005, now);

            // Assert
            var edge = graph.GetNode("A").Neighbors.First();
            Assert.Equal(5.0, edge.Weight, precision: 1);
        }

        [Fact]
        public void Csr_WeightsAsTensor_Aliasing_Verified()
        {
            // Arrange
            var graph = new Graph<string>();
            graph.AddNode("A", "A");
            graph.AddNode("B", "B");
            graph.AddEdge("A", "B", 1.0);
            var csr = graph.CompileCsr();

            // Act: Mutate the weight via the Tensor view
            var tensorView = csr.WeightsAsTensor;
            for (int i = 0; i < tensorView.Data.Length; i++) tensorView.Data[i] = 42.0;

            // Assert: Verify the original CSR weights changed
            Assert.Equal(42.0, csr.Weights[0]);
        }
    }
}