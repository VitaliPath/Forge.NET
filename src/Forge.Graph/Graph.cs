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
        
        public List<Edge<T>> Neighbors;

        public Node(string id, T data)
        {
            Id = id;
            Data = data;
            Neighbors = new List<Edge<T>>();
        }
    }

    public class Graph<T>
    {
        private Dictionary<string, Node<T>> _nodes = new Dictionary<string, Node<T>>();

        public void AddNode(string id, T data)
        {
            if (!_nodes.ContainsKey(id))
            {
                _nodes.Add(id, new Node<T>(id, data));
            }
        }

        public void AddEdge(string fromId, string toId, double weight)
        {
            if (_nodes.ContainsKey(fromId) && _nodes.ContainsKey(toId))
            {
                var source = _nodes[fromId];
                var target = _nodes[toId];

                var edge = new Edge<T>(target, weight);
                source.Neighbors.Add(edge);
            }
            else
            {
                throw new Exception($"Cannot link {fromId} to {toId}: Node missing.");
            }
        }

        public Node<T> GetNode(string id)
        {
            if (_nodes.ContainsKey(id))
            {
                return _nodes[id];
            }
            throw new Exception($"Node {id} not found.");
        }
        
        public IEnumerable<string> GetAllIds()
        {
            return _nodes.Keys;
        }
    }
}