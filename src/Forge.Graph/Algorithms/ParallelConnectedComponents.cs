using System.Collections.Concurrent;

namespace Forge.Graph.Algorithms;

public class ParallelConnectedComponents<T> : IConnectedComponents<T>
{
    /// <summary>
    /// Executes the Parallel DSU algorithm on a GraphCsr snapshot.
    /// Uses deterministic lock ordering and union-by-rank for O(α(N)) performance.
    /// </summary>
    public List<List<string>> Execute(GraphCsr csr, Func<int, double, bool>? predicate = null)
    {
        int n = csr.NodeCount;
        int[] parent = new int[n];
        int[] rank = new int[n];
        object[] locks = new object[n];

        // 1. Initialize Sets
        for (int i = 0; i < n; i++)
        {
            parent[i] = i;
            locks[i] = new object();
        }

        // 2. Parallel Edge Processing
        Parallel.For(0, n, u =>
        {
            int start = csr.RowPtr[u];
            int end = csr.RowPtr[u + 1];

            for (int k = start; k < end; k++)
            {
                int v = csr.ColIdx[k];
                double w = csr.Weights[k];

                // Apply Gravity Threshold
                if (predicate == null || predicate(v, w))
                {
                    Union(u, v, parent, rank, locks);
                }
            }
        });

        // 3. Final Path Compression & Grouping
        var groupedRoots = new ConcurrentDictionary<int, ConcurrentBag<string>>();
        
        Parallel.For(0, n, i =>
        {
            int root = Find(i, parent);
            var bag = groupedRoots.GetOrAdd(root, _ => new ConcurrentBag<string>());
            bag.Add(csr.IndexToId[i]);
        });

        return groupedRoots.Values.Select(bag => bag.ToList()).ToList();
    }

    private int Find(int i, int[] parent)
    {
        while (parent[i] != i)
        {
            // Path Splitting (Greedy path compression for thread-safe find)
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }

    private void Union(int u, int v, int[] parent, int[] rank, object[] locks)
    {
        while (true)
        {
            int rootU = Find(u, parent);
            int rootV = Find(v, parent);

            if (rootU == rootV) return;

            // --- DETERMINISTIC LOCK ORDERING (FORGE-018 Deadlock Prevention) ---
            int first = Math.Min(rootU, rootV);
            int second = Math.Max(rootU, rootV);

            lock (locks[first])
            {
                lock (locks[second])
                {
                    // Re-verify roots inside the lock to handle race conditions
                    if (parent[rootU] != rootU || parent[rootV] != rootV) continue;

                    // Union by Rank
                    if (rank[rootU] < rank[rootV])
                    {
                        parent[rootU] = rootV;
                    }
                    else if (rank[rootU] > rank[rootV])
                    {
                        parent[rootV] = rootU;
                    }
                    else
                    {
                        parent[rootV] = rootU;
                        rank[rootU]++;
                    }
                    return;
                }
            }
        }
    }
}