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
            graph.AccumulateEdgeWeight("file_a", "file_b", 0.5f);
            graph.AccumulateEdgeWeight("file_a", "file_b", 0.5f);

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
                graph.AccumulateEdgeWeight("source", "target", 1.0f);
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
            graph.AccumulateEdgeWeight("file_a", "file_b", 1.0f, 1000);

            // Act: Reinforcement from an "older" commit (Time 500)
            graph.AccumulateEdgeWeight("file_a", "file_b", 1.0f, 500);

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
            graph.AddEdge("A", "B", 1.0f);

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
                    graph.AccumulateEdgeWeight("A", "B", 1.0f);
            });

            var task2 = Task.Run(() =>
            {
                for (int i = 0; i < iterations; i++)
                    graph.AccumulateEdgeWeight("B", "A", 1.0f);
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
            graph.AddEdge("A", "B", 1.0f);

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
            graph.AddEdge("A", "B", 1.0f);
            graph.AddEdge("B", "C", 1.0f);

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

            graph.AccumulateEdgeWeight("A", "B", 10.0f, then);

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
            graph.AddEdge("A", "B", 1.0f);
            var csr = graph.CompileCsr();

            // Act: Mutate the weight via the Tensor view
            var tensorView = csr.WeightsAsTensor;
            for (int i = 0; i < tensorView.Data.Length; i++) tensorView.Data[i] = 42.0f;

            // Assert: Verify the original CSR weights changed
            Assert.Equal(42.0, csr.Weights[0]);
        }

        [Fact]
        public void ParallelProjectNodes_Returns_Correct_Count_And_Data()
        {
            // Arrange
            var graph = new Graph<string>();
            for (int i = 0; i < 1000; i++)
            {
                graph.AddNode($"id_{i}", $"label_{i}");
            }

            // Act: Project only the labels
            var labels = graph.ParallelProjectNodes(n => n.Data);

            // Assert
            Assert.Equal(1000, labels.Count());
            Assert.Contains("label_500", labels);
        }

        [Fact]
        public void ParallelScanNodes_Performs_Concurrent_Mutations_Safely()
        {
            // Arrange
            var graph = new Graph<int>();
            for (int i = 0; i < 1000; i++) graph.AddNode(i.ToString(), 0);

            // Act: Increment every node's data in parallel
            graph.ParallelScanNodes(node =>
            {
                node.Data = 1;
            });

            // Assert
            foreach (var node in graph.Nodes)
            {
                Assert.Equal(1, node.Data);
            }
        }

        // tests/Forge.Tests/GraphTests.cs additions

        [Fact]
        public async Task GetOrAddNode_Concurrently_MaintainsIdentityStability()
        {
            // Arrange
            var graph = new Graph<string>();
            const string sharedId = "IDENTITY-CONCURRENCY-TEST";
            const int threadCount = 100;

            // We use a Task array to track the returned references from every thread
            var tasks = new Task<Node<string>>[threadCount];

            // Act
            // Hammer the graph with 100 simultaneous requests for the same ID
            Parallel.For(0, threadCount, i =>
            {
                tasks[i] = Task.Run(() => graph.GetOrAddNode(sharedId, $"Data-{i}"));
            });

            var results = await Task.WhenAll(tasks);

            // Assert
            // 1. Structural Integrity: Only one node should exist in the graph
            Assert.Single(graph.Nodes);

            // 2. Reference Integrity: Every thread must have received the EXACT same object reference
            var masterReference = graph.GetNode(sharedId);
            foreach (var node in results)
            {
                Assert.Same(masterReference, node);
            }
        }

        [Fact]
        public void GetTopologyHash_Detects_Minor_Structural_Drift()
        {
            // Arrange: Create two identical graphs
            var g1 = new Graph<string>();
            g1.AddNode("A", "data");
            g1.AddNode("B", "data");
            g1.AddEdge("A", "B", 1.0f);

            var g2 = new Graph<string>();
            g2.AddNode("A", "data");
            g2.AddNode("B", "data");
            g2.AddEdge("A", "B", 1.0f);

            // Act
            var hash1 = g1.CompileCsr().GetTopologyHash();
            var hash2 = g2.CompileCsr().GetTopologyHash();

            // Assert: Identity stability
            Assert.Equal(hash1, hash2);

            // Act: Introduce minor weight drift in G2
            g2.AddEdge("A", "B", 0.0001f); // Weight is now 1.0001
            var hash3 = g2.CompileCsr().GetTopologyHash();

            // Assert: Avalanche effect (hash must diverge)
            Assert.NotEqual(hash1, hash3);
        }
    }
}