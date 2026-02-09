using System.Collections.Concurrent;

namespace Forge.Graph
{
    public class Edge<T>
    {
        public Node<T> Target { get; set; }
        public double Weight { get; set; }

        public Edge(Node<T> target, double weight)
        {
            Target = target;
            Weight = weight;
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
        public void AccumulateEdgeWeight(string fromId, string toId, double delta)
        {
            if (!_nodes.TryGetValue(fromId, out var source)) 
                throw new Exception($"Source node {fromId} missing.");
            if (!_nodes.TryGetValue(toId, out var target)) 
                throw new Exception($"Target node {toId} missing.");

            // 1. Update A -> B
            UpdateEdge(source, target, delta);

            // 2. Update B -> A (Bi-directional for undirected community detection)
            UpdateEdge(target, source, delta);
        }

        private void UpdateEdge(Node<T> src, Node<T> dest, double delta)
        {
            lock (src.SyncRoot)
            {
                if (src.EdgeMap.TryGetValue(dest.Id, out var existingEdge))
                {
                    existingEdge.Weight += delta;
                }
                else
                {
                    src.EdgeMap.Add(dest.Id, new Edge<T>(dest, delta));
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
    }
}