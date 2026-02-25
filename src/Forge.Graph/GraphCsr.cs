using Forge.Core;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Forge.Graph
{
    /// <summary>
    /// FORGE-057: High-performance CSR storage. 
    /// Implements Structure-of-Arrays (SoA) for SIMD-ready graph operations.
    /// </summary>
    public readonly struct GraphCsr
    {
        public readonly int[] RowPtr;    // Starting index for each node
        public readonly int[] ColIdx;    // Destination indices
        public readonly float[] Weights;  // Edge gravity (SoA)
        public readonly long[] LastModified; // Timestamps (SoA)

        public readonly Dictionary<string, int> IdToIndex;
        public readonly string[] IndexToId;

        public int NodeCount => RowPtr.Length - 1;
        public int EdgeCount => ColIdx.Length;

        public Tensor WeightsAsTensor => new Tensor(1, EdgeCount, Weights);

        public GraphCsr(int[] rowPtr, int[] colIdx, float[] weights, long[] lastModified,
                        Dictionary<string, int> idToIndex, string[] indexToId)
        {
            RowPtr = rowPtr;
            ColIdx = colIdx;
            Weights = weights;
            LastModified = lastModified;
            IdToIndex = idToIndex;
            IndexToId = indexToId;
        }

        /// <summary>
        /// FORGE-062: Generates a deterministic SHA-256 fingerprint of the graph topology.
        /// Hashing order: RowPtr -> ColIdx -> Weights.
        /// </summary>
        public byte[] GetTopologyHash()
        {
            // Zero-copy cast of numeric arrays to byte spans
            var rowPtrBytes = MemoryMarshal.AsBytes(RowPtr.AsSpan());
            var colIdxBytes = MemoryMarshal.AsBytes(ColIdx.AsSpan());
            var weightsBytes = MemoryMarshal.AsBytes(Weights.AsSpan());

            return HashingBridge.GenerateHash(rowPtrBytes, colIdxBytes, weightsBytes);
        }

        /// <summary>
        /// FORGE-057: Refactored Decay to operate on flat buffers.
        /// This allows the JIT to apply SIMD auto-vectorization.
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
                float multiplier = MathF.Exp(-fLambda * ageInDays);

                weights[i] *= (multiplier < 1e-7f) ? 0.0f : multiplier;
            });
        }
    }
}