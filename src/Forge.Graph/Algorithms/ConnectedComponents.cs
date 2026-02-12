using System;
using System.Collections.Generic;

namespace Forge.Graph.Algorithms
{
    public class ConnectedComponents<T>
    {
        /// <summary>
        /// Standard entry point for dictionary-backed Graph.
        /// </summary>
        public List<List<Node<T>>> Execute(Graph<T> graph, Func<Edge<T>, bool>? predicate = null)
        {
            var islands = new List<List<Node<T>>>();
            var visited = new HashSet<Node<T>>();

            foreach (var node in graph.Nodes)
            {
                if (visited.Contains(node)) continue;

                var currentIsland = new List<Node<T>>();
                var queue = new Queue<Node<T>>();
                
                visited.Add(node);
                queue.Enqueue(node);

                while (queue.Count != 0)
                {
                    var front = queue.Dequeue();
                    currentIsland.Add(front);

                    foreach (var edge in front.Neighbors)
                    {
                        if (predicate != null && !predicate(edge)) continue;

                        if (!visited.Contains(edge.Target))
                        {
                            visited.Add(edge.Target);
                            queue.Enqueue(edge.Target);
                        }
                    }
                }
                islands.Add(currentIsland);
            }
            return islands;
        }

        /// <summary>
        /// High-throughput entry point using the Compiled CSR view.
        /// Predicate now accepts (targetIndex, weight) to maintain performance.
        /// </summary>
        public List<List<string>> Execute(GraphCsr csr, Func<int, double, bool>? predicate = null)
        {
            var islands = new List<List<string>>();
            bool[] visited = new bool[csr.NodeCount];

            for (int i = 0; i < csr.NodeCount; i++)
            {
                if (visited[i]) continue;

                var currentIsland = new List<string>();
                var queue = new Queue<int>();

                visited[i] = true;
                queue.Enqueue(i);

                while (queue.Count > 0)
                {
                    int u = queue.Dequeue();
                    currentIsland.Add(csr.IndexToId[u]);

                    int start = csr.RowPtr[u];
                    int end = csr.RowPtr[u + 1];

                    for (int k = start; k < end; k++)
                    {
                        int v = csr.ColIdx[k];
                        double w = csr.Weights[k];

                        if (predicate != null && !predicate(v, w)) continue;

                        if (!visited[v])
                        {
                            visited[v] = true;
                            queue.Enqueue(v);
                        }
                    }
                }
                islands.Add(currentIsland);
            }
            return islands;
        }
    }
}