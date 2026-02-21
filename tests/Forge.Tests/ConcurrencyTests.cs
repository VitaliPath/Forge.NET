using Xunit;
using Forge.Graph;
using Forge.Core;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Collections.Concurrent;

namespace Forge.Tests
{
    public class ConcurrencyTests
    {
        [Fact]
        public async Task ParallelScan_Obeys_Hardware_Concurrency_Limits()
        {
            // Arrange
            var graph = new Graph<int>();
            for (int i = 0; i < 1000; i++) graph.AddNode(i.ToString(), i);

            var activeThreadIds = new ConcurrentDictionary<int, byte>();
            int totalNodesProcessed = 0;

            // Act: Run the scan. We don't need a gate here because ForgeConcurrency 
            // is now hard-limited. We just need to sample the threads.
            await Task.Run(() =>
            {
                graph.ParallelScanNodes(node =>
                {
                    activeThreadIds.TryAdd(Environment.CurrentManagedThreadId, 0);
                    Interlocked.Increment(ref totalNodesProcessed);

                    // Artificial delay to give the ThreadPool time to try and spawn more threads
                    Thread.Sleep(10);
                });
            });

            // Assert
            // We allow Environment.ProcessorCount + 2 to account for the Main thread 
            // and the Task.Run overhead.
            int maxExpected = Environment.ProcessorCount + 2;
            int actualThreadsUsed = activeThreadIds.Count;

            Assert.True(actualThreadsUsed <= maxExpected,
                $"Thread over-subscription detected! Used {actualThreadsUsed} threads on {Environment.ProcessorCount} cores.");

            Assert.Equal(1000, totalNodesProcessed);
        }
    }
}