using System.Numerics;
using Forge.Core;
using Forge.Graph.Algorithms;

namespace Forge.Graph.Algorithms;

public static class CentralityKernel
{
    private const float DampingFactor = 0.85f;
    private const int MaxIterations = 100;
    private const float ConvergenceEpsilon = 1e-7f;

    public static float[] Calculate(GraphCsr csr)
    {
        int n = csr.NodeCount;
        if (n == 0) return Array.Empty<float>();

        float[] scores = new float[n];
        var cc = new ConnectedComponents<string>();
        var islands = cc.Execute(csr);

        foreach (var islandIds in islands)
        {
            var indices = islandIds.Select(id => csr.IdToIndex[id]).ToArray();
            CalculateIslandCentrality(csr, indices, scores);
        }

        return scores;
    }

    private static void CalculateIslandCentrality(GraphCsr csr, int[] islandIndices, float[] globalScores)
    {
        int n = islandIndices.Length;
        if (n == 0) return;

        // --- 1. PRE-FILTER: TRANSPOSE TO PULL-MODEL ---
        int[] gToL = System.Buffers.ArrayPool<int>.Shared.Rent(csr.NodeCount);
        Array.Fill(gToL, -1);
        for (int i = 0; i < n; i++) gToL[islandIndices[i]] = i;

        // Build the Incoming Adjacency List (Transpose)
        int[] inDegree = new int[n];
        foreach (var uGlobal in islandIndices)
        {
            int start = csr.RowPtr[uGlobal];
            int end = csr.RowPtr[uGlobal + 1];
            for (int k = start; k < end; k++)
            {
                int vLocal = gToL[csr.ColIdx[k]];
                if (vLocal != -1) inDegree[vLocal]++;
            }
        }

        int[] pullRowPtr = new int[n + 1];
        for (int i = 0; i < n; i++) pullRowPtr[i + 1] = pullRowPtr[i] + inDegree[i];

        int totalEdges = pullRowPtr[n];
        int[] pullCols = System.Buffers.ArrayPool<int>.Shared.Rent(totalEdges);
        float[] pullW = System.Buffers.ArrayPool<float>.Shared.Rent(totalEdges);
        int[] cursor = new int[n];
        Array.Copy(pullRowPtr, cursor, n);

        foreach (var uGlobal in islandIndices)
        {
            int uLocal = gToL[uGlobal];
            int start = csr.RowPtr[uGlobal];
            int end = csr.RowPtr[uGlobal + 1];
            for (int k = start; k < end; k++)
            {
                int vLocal = gToL[csr.ColIdx[k]];
                if (vLocal != -1)
                {
                    int pos = cursor[vLocal]++;
                    pullCols[pos] = uLocal;
                    pullW[pos] = csr.Weights[k];
                }
            }
        }

        float[] v = new float[n];
        float[] nextV = new float[n];
        Array.Fill(v, 1.0f / MathF.Sqrt(n));
        float jumpProb = (1.0f - DampingFactor) / n;

        // --- 2. HOT LOOP: GATHER ENGINE ---
        for (int iter = 0; iter < MaxIterations; iter++)
        {
            // GATHER: Sequential writes to nextV[i]
            for (int i = 0; i < n; i++)
            {
                float sum = 0;
                int start = pullRowPtr[i];
                int end = pullRowPtr[i + 1];
                // CPU pre-fetcher loves this loop
                for (int k = start; k < end; k++) sum += v[pullCols[k]] * pullW[k];
                nextV[i] = (sum * DampingFactor) + jumpProb;
            }

            // SIMD NORMALIZATION
            var vNormSqTotal = Vector<float>.Zero;
            int vectorWidth = Vector<float>.Count;
            int limit = n - (n % vectorWidth);

            for (int i = 0; i < limit; i += vectorWidth)
            {
                var vNext = new Vector<float>(nextV, i);
                vNormSqTotal += vNext * vNext;
            }

            float normSq = Vector.Dot(vNormSqTotal, Vector<float>.One);
            for (int i = limit; i < n; i++) normSq += nextV[i] * nextV[i];

            float invNorm = 1.0f / MathF.Sqrt(normSq);
            var vInvNorm = new Vector<float>(invNorm);
            var vDiffTotal = Vector<float>.Zero;

            for (int i = 0; i < limit; i += vectorWidth)
            {
                var vNextNorm = new Vector<float>(nextV, i) * vInvNorm;
                vDiffTotal += Vector.Abs(vNextNorm - new Vector<float>(v, i));
                vNextNorm.CopyTo(v, i);
            }

            if (Vector.Dot(vDiffTotal, Vector<float>.One) < ConvergenceEpsilon) break;
        }

        for (int i = 0; i < n; i++) globalScores[islandIndices[i]] = v[i];

        System.Buffers.ArrayPool<int>.Shared.Return(gToL);
        System.Buffers.ArrayPool<int>.Shared.Return(pullCols);
        System.Buffers.ArrayPool<float>.Shared.Return(pullW);
    }
}