using Forge.Core;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Linq;
using System.Threading.Tasks;
using Forge.Graph.Persistence;
using Forge.Graph.Algorithms;


namespace Forge.Graph
{
    /// <summary>
    /// FORGE-064: Represents an edge that crosses a cluster boundary.
    /// </summary>
    public record BoundaryEdge(int SourceIndex, int TargetIndex, float Weight);

    /// <summary>
    /// FORGE-057: High-performance CSR storage. 
    /// Implements Structure-of-Arrays (SoA) for SIMD-ready graph operations.
    /// </summary>
    public readonly struct GraphCsr
    {
        // Internal data buffers representing the SoA layout
        public readonly int[] RowPtr;    // Starting index for each node (Size: N + 1)
        public readonly int[] ColIdx;    // Destination indices (Size: E)
        public readonly float[] Weights;  // Edge gravity (Size: E)
        public readonly long[] LastModified; // Timestamps (Size: E)
        public readonly byte[] EdgeTypes;

        public readonly Dictionary<string, int> IdToIndex;
        public readonly string[] IndexToId;

        public int NodeCount => RowPtr.Length - 1;
        public int EdgeCount => ColIdx.Length;

        /// <summary>
        /// Provides a Tensor view of the weights for gradient-based updates or ML processing.
        /// </summary>
        public Tensor WeightsAsTensor => new Tensor(1, EdgeCount, Weights);

        /// <summary>
        /// FORGE-068: Calculates the Jaccard Similarity between the neighbor sets of two nodes.
        /// Uses a two-finger linear scan over pre-sorted CSR adjacency lists.
        /// Complexity: O(deg(u) + deg(v))
        /// </summary>
        public float CalculateNeighborSimilarity(int u, int v)
        {
            if (u < 0 || u >= NodeCount || v < 0 || v >= NodeCount)
                return 0f;

            int startU = RowPtr[u];
            int endU = RowPtr[u + 1];
            int startV = RowPtr[v];
            int endV = RowPtr[v + 1];

            int countU = endU - startU;
            int countV = endV - startV;

            if (countU == 0 && countV == 0) return 0f;

            int intersection = 0;
            int pU = startU;
            int pV = startV;

            // Two-Finger Linear Scan
            while (pU < endU && pV < endV)
            {
                int valU = ColIdx[pU];
                int valV = ColIdx[pV];

                if (valU == valV)
                {
                    intersection++;
                    pU++;
                    pV++;
                }
                else if (valU < valV)
                {
                    pU++;
                }
                else
                {
                    pV++;
                }
            }

            // Jaccard Formula: |A ∩ B| / |A ∪ B|
            // |A ∪ B| = |A| + |B| - |A ∩ B|
            int union = countU + countV - intersection;
            
            return (float)intersection / union;
        }

        public GraphCsr(int[] rowPtr, int[] colIdx, float[] weights, long[] lastModified, 
                        byte[] edgeTypes, Dictionary<string, int> idToIndex, string[] indexToId)
        {
            RowPtr = rowPtr;
            ColIdx = colIdx;
            Weights = weights;
            LastModified = lastModified;
            EdgeTypes = edgeTypes;
            IdToIndex = idToIndex;
            IndexToId = indexToId;
        }

        #region Persistence (FORGE-063)

        /// <summary>
        /// FORGE-063: Native SoA Persistence. 
        /// Delegates to CsrIOHandler to maintain Separation of Concerns.
        /// </summary>
        public void Save(Stream stream) => CsrIOHandler.WriteToStream(this, stream);

        /// <summary>
        /// FORGE-063: "Warm Brain" re-hydration.
        /// Directly restores SoA buffers from a binary stream.
        /// </summary>
        public static GraphCsr Load(Stream stream) => CsrIOHandler.ReadFromStream(stream);

        #endregion

        #region Topology & Analysis

        /// <summary>
        /// FORGE-062: Generates a deterministic SHA-256 fingerprint of the graph topology.
        /// Hashing order: RowPtr -> ColIdx -> Weights.
        /// </summary>
        public byte[] GetTopologyHash()
        {
            // Zero-copy cast of numeric arrays to byte spans for the HashingBridge
            var rowPtrBytes = MemoryMarshal.AsBytes(RowPtr.AsSpan());
            var colIdxBytes = MemoryMarshal.AsBytes(ColIdx.AsSpan());
            var weightsBytes = MemoryMarshal.AsBytes(Weights.AsSpan());

            return HashingBridge.GenerateHash(rowPtrBytes, colIdxBytes, weightsBytes);
        }

        /// <summary>
        /// FORGE-064: Identifies all edges crossing the perimeter of the specified group.
        /// </summary>
        public List<BoundaryEdge> GetBoundaryEdges(HashSet<int> groupIndices)
        {
            var boundary = new ConcurrentBag<BoundaryEdge>();

            var localRowPtr = this.RowPtr;
            var localColIdx = this.ColIdx;
            var localWeights = this.Weights;

            // Threshold-based parallelism to avoid overhead on small clusters
            if (groupIndices.Count > 500)
            {
                Parallel.ForEach(groupIndices, ForgeConcurrency.DefaultOptions, u =>
                {
                    ScanBoundaryInternal(u, groupIndices, boundary, localRowPtr, localColIdx, localWeights);
                });
            }
            else
            {
                foreach (int u in groupIndices)
                {
                    ScanBoundaryInternal(u, groupIndices, boundary, localRowPtr, localColIdx, localWeights);
                }
            }

            return boundary.ToList();
        }

        private static void ScanBoundaryInternal(
            int u,
            HashSet<int> groupIndices,
            ConcurrentBag<BoundaryEdge> results,
            int[] rowPtr,
            int[] colIdx,
            float[] weights)
        {
            int start = rowPtr[u];
            int end = rowPtr[u + 1];

            for (int k = start; k < end; k++)
            {
                int v = colIdx[k];
                if (!groupIndices.Contains(v))
                {
                    results.Add(new BoundaryEdge(u, v, weights[k]));
                }
            }
        }

        #endregion

        #region Algorithms

        /// <summary>
        /// Calculates the centrality (importance) of nodes using SIMD-accelerated Power Iteration.
        /// </summary>
        public float[] CalculateCentrality() => CentralityKernel.Calculate(this);

        /// <summary>
        /// FORGE-067: Calculates the dependency depth (Architectural Elevation) from entry points.
        /// Defaults to a penalty of 3 for Associative links to prioritize Structural hierarchy.
        /// </summary>
        public int[] CalculateDependencyDepth(int[] roots, int associativePenalty = 3) 
            => DependencyDepthWalker.Calculate(this, roots, associativePenalty);

        #endregion

        /// <summary>
        /// FORGE-057: Applies temporal decay to all edges via SIMD-friendly flat buffer iteration.
        /// </summary>
        public void ApplyDecay(double lambda, long nowUnix)
        {
            const float secondsPerDay = 86400.0f;
            float fLambda = (float)lambda;

            var weights = this.Weights;
            var lastModified = this.LastModified;
            int count = this.EdgeCount;

            Parallel.For(0, count, ForgeConcurrency.DefaultOptions, i =>
            {
                float ageInDays = (float)((nowUnix - lastModified[i]) / secondsPerDay);
                float multiplier = MathF.Exp(-fLambda * MathF.Max(0, ageInDays));

                weights[i] *= (multiplier < 1e-7f) ? 0.0f : multiplier;
            });
        }
    }
}