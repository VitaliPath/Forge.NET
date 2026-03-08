using System.Buffers;
using System.Collections.Concurrent;

namespace Forge.Graph.Algorithms;

public class NonAcyclicGraphException(string message) : Exception(message);

public static class FlowRankingKernel
{
    /// <summary>
    /// FORGE-069: Calculates the topological flow depth within a node subset.
    /// Only traverses edges where RelationshipType == Structural.
    /// Complexity: O(V_subset + E_subset)
    /// </summary>
    public static int[] Calculate(GraphCsr csr, int[] nodeSubset)
    {
        int n = csr.NodeCount;
        int subsetSize = nodeSubset.Length;
        if (subsetSize == 0) return Array.Empty<int>();

        // 1. Setup Local Mapping for O(1) subset checks
        // Using ArrayPool to minimize GC pressure on high-frequency synthesis
        int[] globalToLocal = ArrayPool<int>.Shared.Rent(n);
        Array.Fill(globalToLocal, -1);
        for (int i = 0; i < subsetSize; i++) globalToLocal[nodeSubset[i]] = i;

        int[] inDegree = new int[subsetSize];
        int[] ranks = new int[subsetSize];
        
        // 2. In-Degree Calculation (Structural Only)
        foreach (int u in nodeSubset)
        {
            int start = csr.RowPtr[u];
            int end = csr.RowPtr[u + 1];

            for (int k = start; k < end; k++)
            {
                // Only follow structural egress within the subset
                if (csr.EdgeTypes[k] == (byte)RelationshipType.Structural)
                {
                    int v = csr.ColIdx[k];
                    int vLocal = globalToLocal[v];
                    if (vLocal != -1) inDegree[vLocal]++;
                }
            }
        }

        // 3. Queue Source Nodes (Rank 0)
        var queue = new Queue<int>();
        for (int i = 0; i < subsetSize; i++)
        {
            if (inDegree[i] == 0) queue.Enqueue(nodeSubset[i]);
        }

        int processedCount = 0;
        // 4. Flow Propagation BFS
        while (queue.Count > 0)
        {
            int u = queue.Dequeue();
            int uLocal = globalToLocal[u];
            processedCount++;

            int start = csr.RowPtr[u];
            int end = csr.RowPtr[u + 1];

            for (int k = start; k < end; k++)
            {
                if (csr.EdgeTypes[k] == (byte)RelationshipType.Structural)
                {
                    int v = csr.ColIdx[k];
                    int vLocal = globalToLocal[v];
                    if (vLocal != -1)
                    {
                        // $D_f(v) = \max(D_f(v), D_f(u) + 1)$
                        ranks[vLocal] = Math.Max(ranks[vLocal], ranks[uLocal] + 1);
                        
                        inDegree[vLocal]--;
                        if (inDegree[vLocal] == 0) queue.Enqueue(v);
                    }
                }
            }
        }

        // 5. Cleanup & Cycle Check
        ArrayPool<int>.Shared.Return(globalToLocal);

        if (processedCount < subsetSize)
        {
            throw new NonAcyclicGraphException("Circular structural dependency detected. The narrative cannot be linearized.");
        }

        return ranks;
    }
}