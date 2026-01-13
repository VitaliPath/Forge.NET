namespace Forge.Graph.Algorithms;

public class ConnectedComponents<T>
{
    public List<List<Node<T>>> Execute(Graph<T> graph)
    {
        var islands = new List<List<Node<T>>>();
        var visited = new HashSet<Node<T>>();

        foreach (var node in graph.Nodes)
        {
            if (visited.Contains(node))
            {
                continue;
            }

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
