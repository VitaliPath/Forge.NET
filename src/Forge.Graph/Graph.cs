using System.Collections.Concurrent;

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
        private readonly ConcurrentDictionary<string, Node<T>> _nodes = new();
        public IEnumerable<Node<T>> Nodes => _nodes.Values;

        public void AddNode(string id, T data)
        {
            _nodes.TryAdd(id, new Node<T>(id, data));
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

            // 1. Update A -> B
            UpdateEdge(source, target, delta, timestamp);

            // 2. Update B -> A
            UpdateEdge(target, source, delta, timestamp);
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

        public Node<T> GetNode(string id)
        {
            if (_nodes.TryGetValue(id, out var node)) return node;
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