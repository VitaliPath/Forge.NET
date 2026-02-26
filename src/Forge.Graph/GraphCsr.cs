using Forge.Core;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Forge.Graph
{
    /// <summary>
    /// FORGE-057: High-performance CSR storage. 
    /// Implements Structure-of-Arrays (SoA) for SIMD-ready graph operations.
    /// </summary>
    public readonly struct GraphCsr
    {
        private const uint MagicBytes = 0x46524745; // "FRGE"
        private const int CurrentVersion = 1;
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
        /// FORGE-063: Performs high-speed binary serialization of the CSR Structure-of-Arrays.
        /// </summary>
        public void Save(Stream stream)
        {
            using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

            // 1. Header
            writer.Write(MagicBytes);
            writer.Write(CurrentVersion);
            writer.Write(NodeCount);
            writer.Write(EdgeCount);

            // 2. Numeric Buffers (Bulk Write)
            writer.Write(MemoryMarshal.AsBytes(RowPtr.AsSpan()));
            writer.Write(MemoryMarshal.AsBytes(ColIdx.AsSpan()));
            writer.Write(MemoryMarshal.AsBytes(Weights.AsSpan()));
            writer.Write(MemoryMarshal.AsBytes(LastModified.AsSpan()));

            // 3. String Metadata (Identity Pool)
            foreach (var id in IndexToId)
            {
                writer.Write(id);
            }
        }

        /// <summary>
        /// FORGE-063: Re-hydrates a GraphCsr snapshot directly into SoA buffers.
        /// </summary>
        public static GraphCsr Load(Stream stream)
        {
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

            // 1. Validate Header
            if (reader.ReadUInt32() != MagicBytes) throw new InvalidDataException("Invalid Forge Snapshot Magic Bytes.");
            if (reader.ReadInt32() != CurrentVersion) throw new InvalidDataException("Unsupported GraphCsr Schema Version.");

            int nodeCount = reader.ReadInt32();
            int edgeCount = reader.ReadInt32();

            // 2. Allocate SoA Buffers
            int[] rowPtr = new int[nodeCount + 1];
            int[] colIdx = new int[edgeCount];
            float[] weights = new float[edgeCount];
            long[] lastModified = new long[edgeCount];

            // 3. Bulk Read numeric data
            reader.Read(MemoryMarshal.AsBytes(rowPtr.AsSpan()));
            reader.Read(MemoryMarshal.AsBytes(colIdx.AsSpan()));
            reader.Read(MemoryMarshal.AsBytes(weights.AsSpan()));
            reader.Read(MemoryMarshal.AsBytes(lastModified.AsSpan()));

            // 4. Reconstruct Identity Mappings
            string[] indexToId = new string[nodeCount];
            var idToIndex = new Dictionary<string, int>(nodeCount, StringComparer.OrdinalIgnoreCase);

            for (int i = 0; i < nodeCount; i++)
            {
                string id = reader.ReadString();
                indexToId[i] = id;
                idToIndex.Add(id, i);
            }

            return new GraphCsr(rowPtr, colIdx, weights, lastModified, idToIndex, indexToId);
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