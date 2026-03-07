using System;
using System.Collections.Generic;
using Forge.Graph;

namespace Forge.Graph.Algorithms;

/// <summary>
/// FORGE-067: Calculates Architectural Elevation (delta) using a bucket-based BFS.
/// Supports STRUCTURAL priority with an optional penalty for ASSOCIATIVE edges.
/// </summary>
public static class DependencyDepthWalker
{
    /// <summary>
    /// Calculates the shortest path distance from roots.
    /// </summary>
    /// <param name="associativePenalty">Weight for Associative edges. Use -1 to ignore them entirely.</param>
    public static int[] Calculate(GraphCsr csr, int[] roots, int associativePenalty = 3)
    {
        int n = csr.NodeCount;
        int[] depths = new int[n];
        Array.Fill(depths, -1);

        if (roots == null || roots.Length == 0) return depths;

        // Buckets for different depth levels to maintain O(V+E)
        var buckets = new List<List<int>>();
        
        foreach (int rootIdx in roots)
        {
            if (rootIdx < 0 || rootIdx >= n) continue;
            depths[rootIdx] = 0;
            GetBucket(buckets, 0).Add(rootIdx);
        }

        int currentDepth = 0;
        int maxReachableDepth = 0;

        // Traverse buckets sequentially
        while (currentDepth <= maxReachableDepth)
        {
            var currentBucket = GetBucket(buckets, currentDepth);
            if (currentBucket.Count == 0)
            {
                currentDepth++;
                continue;
            }

            // Process all nodes at the current depth
            for (int i = 0; i < currentBucket.Count; i++)
            {
                int u = currentBucket[i];

                int start = csr.RowPtr[u];
                int end = csr.RowPtr[u + 1];

                for (int k = start; k < end; k++)
                {
                    bool isStructural = csr.EdgeTypes[k] == (byte)RelationshipType.Structural;
                    
                    // Skip associative if penalty is -1 (Structural Only mode)
                    if (!isStructural && associativePenalty < 0) continue;

                    int weight = isStructural ? 1 : associativePenalty;
                    int v = csr.ColIdx[k];
                    int newDepth = currentDepth + weight;

                    if (depths[v] == -1 || newDepth < depths[v])
                    {
                        depths[v] = newDepth;
                        GetBucket(buckets, newDepth).Add(v);
                        if (newDepth > maxReachableDepth) maxReachableDepth = newDepth;
                    }
                }
            }
            currentDepth++;
        }

        return depths;
    }

    private static List<int> GetBucket(List<List<int>> buckets, int d)
    {
        while (buckets.Count <= d) buckets.Add(new List<int>());
        return buckets[d];
    }
}