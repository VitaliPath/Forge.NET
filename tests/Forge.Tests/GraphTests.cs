using Xunit;
using Forge.Graph;
using System.Linq;
using System.Threading.Tasks;

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
    }
}