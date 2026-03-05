using System.Collections.Concurrent;
using Forge.Core;

namespace Forge.Graph
{
    public enum RelationshipType : byte
    {
        Associative = 0,
        Structural = 1
    }

    public class Edge<T>
    {
        public Node<T> Target { get; set; }
        public float Weight { get; set; }
        public long LastModified { get; set; } 
        public RelationshipType Type { get; set; }

        public Edge(Node<T> target, float weight, long lastModified = 0, RelationshipType type = RelationshipType.Associative)
        {
            Target = target;
            Weight = weight;
            LastModified = lastModified;
            Type = type;
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
        /// FORGE-065: Accumulates weight into an edge and defines its structural nature.
        /// Structural edges (Parent -> Child) are directed, while Associative edges are symmetric.
        /// </summary>
        /// <param name="timestamp">Optional Unix timestamp of the reinforcement.</param>
        /// <param name="type">The nature of the link: Associative (Default) or Structural.</param>
        public void AccumulateEdgeWeight(string fromId, string toId, float delta, long timestamp = 0, RelationshipType type = RelationshipType.Associative)
        {
            if (!_nodes.TryGetValue(fromId, out var source))
                throw new Exception($"Source node {fromId} missing.");
            if (!_nodes.TryGetValue(toId, out var target))
                throw new Exception($"Target node {toId} missing.");

            if (type == RelationshipType.Structural)
            {
                if (IsStructuralPath(toId, fromId))
                {
                    throw new InvalidOperationException($"Circular structural dependency: '{fromId}' and '{toId}' cannot have a parent/child relationship.");
                }
            }

            if (fromId == toId)
            {
                lock (source.SyncRoot)
                {
                    UpdateEdgeInternal(source, target, delta, timestamp, type);
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
                    UpdateEdgeInternal(source, target, delta, timestamp, type);

                    if (type == RelationshipType.Associative)
                    {
                        UpdateEdgeInternal(target, source, delta, timestamp, type);
                    }
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
            float fLambda = (float)lambda;

            Parallel.ForEach(_nodes.Values, ForgeConcurrency.DefaultOptions, node =>
            {
                lock (node.SyncRoot)
                {
                    foreach (var edge in node.EdgeMap.Values)
                    {
                        double ageInDays = Math.Max(0, (nowUnix - edge.LastModified) / secondsPerDay);
                        float multiplier = MathF.Exp(-fLambda * (float)ageInDays);

                        edge.Weight *= (multiplier < 1e-7f) ? 0.0f : multiplier;
                    }
                }
            });
        }

        private bool IsStructuralPath(string startId, string targetId, HashSet<string>? visited = null)
        {
            if (startId == targetId) return true;
            visited ??= new HashSet<string>();
            visited.Add(startId);

            if (_nodes.TryGetValue(startId, out var node))
            {
                foreach (var edge in node.Neighbors.Where(e => e.Type == RelationshipType.Structural))
                {
                    if (!visited.Contains(edge.Target.Id) && IsStructuralPath(edge.Target.Id, targetId, visited))
                        return true;
                }
            }
            return false;
        }

        /// <summary>
        /// FORGE-014: Core edge update logic.
        /// Assumes the caller has already acquired the necessary locks on SyncRoot.
        /// </summary>
        private void UpdateEdgeInternal(Node<T> src, Node<T> dest, float delta, long timestamp, RelationshipType type)
        {
            if (src.EdgeMap.TryGetValue(dest.Id, out var existingEdge))
            {
                existingEdge.Weight += delta;
                existingEdge.LastModified = Math.Max(existingEdge.LastModified, timestamp);
                if (type == RelationshipType.Structural) existingEdge.Type = type;
            }
            else
            {
                src.EdgeMap.Add(dest.Id, new Edge<T>(dest, delta, timestamp, type));
            }
        }

        private void UpdateEdge(Node<T> src, Node<T> dest, float delta, long timestamp)
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

        public void AddEdge(string fromId, string toId, float weight, RelationshipType type = RelationshipType.Associative)
        {
            AccumulateEdgeWeight(fromId, toId, weight, 0, type);
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

        /// <summary>
        /// FORGE-065: Compiles the mutable graph into a high-performance SoA layout.
        /// Updated to include the EdgeTypes byte buffer for structural filtering.
        /// </summary>
        public GraphCsr CompileCsr()
        {
            var sortedNodes = Nodes.OrderBy(n => n.Id).ToList();
            int n = sortedNodes.Count;

            var idToIndex = sortedNodes.Select((n, i) => new { n.Id, i }).ToDictionary(x => x.Id, x => x.i);
            var indexToId = sortedNodes.Select(n => n.Id).ToArray();

            int[] rowPtr = new int[n + 1];
            int totalEdges = 0;
            for (int i = 0; i < n; i++)
            {
                rowPtr[i] = totalEdges;
                totalEdges += sortedNodes[i].EdgeMap.Count;
            }
            rowPtr[n] = totalEdges;

            // Allocate pinned arrays for high-performance SoA
            int[] colIdx = GC.AllocateArray<int>(totalEdges, pinned: true);
            float[] weights = GC.AllocateArray<float>(totalEdges, pinned: true);
            long[] lastModified = GC.AllocateArray<long>(totalEdges, pinned: true);
            byte[] edgeTypes = GC.AllocateArray<byte>(totalEdges, pinned: true);

            int edgeIdx = 0;
            for (int i = 0; i < n; i++)
            {
                var sortedNeighbors = sortedNodes[i].Neighbors.OrderBy(e => e.Target.Id);
                foreach (var edge in sortedNeighbors)
                {
                    colIdx[edgeIdx] = idToIndex[edge.Target.Id];
                    weights[edgeIdx] = edge.Weight;
                    lastModified[edgeIdx] = edge.LastModified;
                    edgeTypes[edgeIdx] = (byte)edge.Type;
                    edgeIdx++;
                }
            }

            return new GraphCsr(rowPtr, colIdx, weights, lastModified, edgeTypes, idToIndex, indexToId);
        }
    }
}