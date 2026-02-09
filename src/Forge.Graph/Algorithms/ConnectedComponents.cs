namespace Forge.Graph.Algorithms;

public class ConnectedComponents<T>
{
    /// <summary>
    /// Traverses the graph to find disconnected components (islands).
    /// </summary>
    /// <param name="graph">The target graph.</param>
    /// <param name="predicate">An optional filter to ignore specific edges (e.g., weight thresholds).</param>
    public List<List<Node<T>>> Execute(Graph<T> graph, Func<Edge<T>, bool>? predicate = null)
    {
        var islands = new List<List<Node<T>>>();
        var visited = new HashSet<Node<T>>();

        foreach (var node in graph.Nodes)
        {
            if (visited.Contains(node)) continue;

            var currentIsland = new List<Node<T>>();
            visited.Add(node);
            
            Queue<Node<T>> queue = new Queue<Node<T>>();
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
}