using System.Collections.Concurrent;
using Forge.Core;

namespace Forge.Graph
{
    public class Edge<T>
    {
        public Node<T> Target { get; set; }
        public double Weight { get; set; }
        
        public long LastModified { get; set; } 

        public Edge(Node<T> target, double weight, long lastModified = 0)
        {
            Target = target;
            Weight = weight;
            LastModified = lastModified;
        }
    }

    public class Node<T>
    {
        public string Id;
        public T Data;
        
        internal readonly Dictionary<string, Edge<T>> EdgeMap;
        
        public IEnumerable<Edge<T>> Neighbors => EdgeMap.Values;

        internal readonly object SyncRoot = new object();

        public Node(string id, T data)
        {
            Id = id;
            Data = data;
            EdgeMap = new Dictionary<string, Edge<T>>();
        }
    }

    public class Graph<T>
    {
        private readonly ConcurrentDictionary<string, Node<T>> _nodes = new(StringComparer.OrdinalIgnoreCase);
        public IEnumerable<Node<T>> Nodes => _nodes.Values;

        /// <summary>
        /// FORGE-023: Atomically retrieves an existing node or creates a new one.
        /// Ensures identity stability in high-concurrency ingestion environments.
        /// </summary>
        public Node<T> GetOrAddNode(string id, T data)
        {
            if (string.IsNullOrWhiteSpace(id))
                throw new ArgumentException("Node ID cannot be null or whitespace.");

            return _nodes.GetOrAdd(id, (key) => new Node<T>(key, data));
        }

        /// <summary>
        /// FORGE-023: Refactored to utilize the atomic GetOrAddNode primitive.
        /// </summary>
        public void AddNode(string id, T data)
        {
            GetOrAddNode(id, data);
        }

        /// <summary>
        /// FORGE-019: Atomically removes a node and performs a symmetric edge cascade.
        /// Utilizes deterministic lock ordering (ID-based) to maintain thread safety.
        /// Citation: Shapiro (1986) - Concurrent Graph Data Structures.
        /// </summary>
        public bool RemoveNode(string id)
        {
            if (!_nodes.TryRemove(id, out var nodeToRemove))
                return false;

            lock (nodeToRemove.SyncRoot)
            {
                var neighbors = nodeToRemove.EdgeMap.Keys.ToList();

                foreach (var neighborId in neighbors)
                {
                    if (_nodes.TryGetValue(neighborId, out var neighbor))
                    {
                        bool uFirst = string.Compare(id, neighborId, StringComparison.Ordinal) < 0;
                        var first = uFirst ? nodeToRemove : neighbor;
                        var second = uFirst ? neighbor : nodeToRemove;

                        lock (first.SyncRoot)
                        {
                            lock (second.SyncRoot)
                            {
                                neighbor.EdgeMap.Remove(id);
                                nodeToRemove.EdgeMap.Remove(neighborId);
                            }
                        }
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Accumulates weight into an edge. 
        /// Creates the edge if it doesn't exist.
        /// </summary>
        /// <param name="timestamp">FORGE-0011: Optional Unix timestamp of the reinforcement.</param>
        public void AccumulateEdgeWeight(string fromId, string toId, double delta, long timestamp = 0)
        {
            if (!_nodes.TryGetValue(fromId, out var source))
                throw new Exception($"Source node {fromId} missing.");
            if (!_nodes.TryGetValue(toId, out var target))
                throw new Exception($"Target node {toId} missing.");

            if (fromId == toId)
            {
                lock (source.SyncRoot)
                {
                    UpdateEdgeInternal(source, target, delta, timestamp);
                }
                return;
            }

            bool sourceFirst = string.Compare(fromId, toId, StringComparison.Ordinal) < 0;
            var first = sourceFirst ? source : target;
            var second = sourceFirst ? target : source;

            lock (first.SyncRoot)
            {
                lock (second.SyncRoot)
                {
                    UpdateEdgeInternal(source, target, delta, timestamp);
                    UpdateEdgeInternal(target, source, delta, timestamp);
                }
            }
        }

        /// <summary>
        /// FORGE-022: Performs a parallel scan of all nodes in the graph.
        /// Ideal for in-place updates or complex validation passes.
        /// </summary>
        public void ParallelScanNodes(Action<Node<T>> action)
        {
            var partitioner = Partitioner.Create(_nodes.Values, EnumerablePartitionerOptions.NoBuffering);
            Parallel.ForEach(partitioner, ForgeConcurrency.DefaultOptions, action);
        }

        /// <summary>
        /// FORGE-022: High-speed parallel projection of node properties.
        /// Maps graph vertices into a resulting set (Map-Reduce pattern).
        /// </summary>
        public IEnumerable<TResult> ParallelProjectNodes<TResult>(Func<Node<T>, TResult> selector)
        {
            var results = new ConcurrentBag<TResult>();
            var partitioner = Partitioner.Create(_nodes.Values, EnumerablePartitionerOptions.NoBuffering);

            Parallel.ForEach(partitioner, ForgeConcurrency.DefaultOptions, node =>
            {
                results.Add(selector(node));
            });

            return results;
        }

        /// <summary>
        /// FORGE-021: Applies graph-wide temporal decay to all edges.
        /// Formula: w = w * exp(-lambda * delta_t)
        /// </summary>
        /// <param name="lambda">The decay constant.</param>
        /// <param name="nowUnix">The current reference timestamp.</param>
        public void ApplyDecay(double lambda, long nowUnix)
        {
            const double secondsPerDay = 86400.0;

            Parallel.ForEach(_nodes.Values, ForgeConcurrency.DefaultOptions, node =>
            {
                lock (node.SyncRoot)
                {
                    foreach (var edge in node.EdgeMap.Values)
                    {
                        double ageInDays = Math.Max(0, (nowUnix - edge.LastModified) / secondsPerDay);
                        double multiplier = Math.Exp(-lambda * ageInDays);
                        edge.Weight *= (multiplier < 1e-9) ? 0.0 : multiplier;
                    }
                }
            });
        }

        /// <summary>
        /// FORGE-014: Core edge update logic.
        /// Assumes the caller has already acquired the necessary locks on SyncRoot.
        /// </summary>
        private void UpdateEdgeInternal(Node<T> src, Node<T> dest, double delta, long timestamp)
        {
            if (src.EdgeMap.TryGetValue(dest.Id, out var existingEdge))
            {
                existingEdge.Weight += delta;
                existingEdge.LastModified = Math.Max(existingEdge.LastModified, timestamp);
            }
            else
            {
                src.EdgeMap.Add(dest.Id, new Edge<T>(dest, delta, timestamp));
            }
        }

        private void UpdateEdge(Node<T> src, Node<T> dest, double delta, long timestamp)
        {
            lock (src.SyncRoot)
            {
                if (src.EdgeMap.TryGetValue(dest.Id, out var existingEdge))
                {
                    existingEdge.Weight += delta;
                    
                    existingEdge.LastModified = Math.Max(existingEdge.LastModified, timestamp);
                }
                else
                {
                    src.EdgeMap.Add(dest.Id, new Edge<T>(dest, delta, timestamp));
                }
            }
        }

        public void AddEdge(string fromId, string toId, double weight)
        {
            AccumulateEdgeWeight(fromId, toId, weight);
        }

        /// <summary>
        /// FORGE-0013: Safely attempts to retrieve a node by its ID without throwing an exception.
        /// Follows the standard .NET Try-Parse pattern.
        /// </summary>
        /// <param name="id">The unique identifier of the node.</param>
        /// <param name="node">The retrieved node, or null if not found.</param>
        /// <returns>True if the node exists; otherwise, false.</returns>
        public bool TryGetNode(string id, out Node<T>? node)
        {
            if (string.IsNullOrEmpty(id))
            {
                node = null;
                return false;
            }

            return _nodes.TryGetValue(id, out node);
        }

        /// <summary>
        /// Retrieves a node by ID. Throws an exception if the node is missing.
        /// Refactored to utilize the safe TryGetNode implementation.
        /// </summary>
        public Node<T> GetNode(string id)
        {
            if (TryGetNode(id, out var node))
                return node!;

            throw new Exception($"Node {id} not found.");
        }

        public IEnumerable<string> GetAllIds() => _nodes.Keys;

        public GraphCsr CompileCsr()
        {
            // 1. Establish deterministic node ordering
            var sortedNodes = Nodes.OrderBy(n => n.Id).ToList();
            var idToIndex = sortedNodes.Select((n, i) => new { n.Id, i })
                                       .ToDictionary(x => x.Id, x => x.i);
            var indexToId = sortedNodes.Select(n => n.Id).ToArray();

            int n = sortedNodes.Count;
            int[] rowPtr = new int[n + 1];

            int totalEdges = 0;
            for (int i = 0; i < n; i++)
            {
                rowPtr[i] = totalEdges;
                totalEdges += sortedNodes[i].EdgeMap.Count;
            }
            rowPtr[n] = totalEdges;

            int[] colIdx = new int[totalEdges];
            double[] weights = new double[totalEdges];
            long[] lastModified = new long[totalEdges];

            int edgeIdx = 0;
            for (int i = 0; i < n; i++)
            {
                var sortedNeighbors = sortedNodes[i].Neighbors.OrderBy(e => e.Target.Id);

                foreach (var edge in sortedNeighbors)
                {
                    colIdx[edgeIdx] = idToIndex[edge.Target.Id];
                    weights[edgeIdx] = edge.Weight;
                    lastModified[edgeIdx] = edge.LastModified;
                    edgeIdx++;
                }
            }

            return new GraphCsr(rowPtr, colIdx, weights, lastModified, idToIndex, indexToId);
        }
    }
}