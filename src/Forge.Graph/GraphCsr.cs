using Forge.Core;
using System.Collections.Generic;
using System;
using System.Threading.Tasks;

namespace Forge.Graph
{
    public struct GraphCsr
    {
        public readonly int[] RowPtr;
        public readonly int[] ColIdx;
        public readonly float[] Weights;
        public readonly long[] LastModified;

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

        public void ApplyDecay(double lambda, long nowUnix)
        {
            const double secondsPerDay = 86400.0;
            var weights = this.Weights;
            var lastModified = this.LastModified;
            int count = this.EdgeCount;

            Parallel.For(0, count, i =>
            {
                double ageInDays = Math.Max(0, (nowUnix - lastModified[i]) / secondsPerDay);
                float multiplier = (float)Math.Exp(-lambda * ageInDays);

                weights[i] *= (multiplier < 1e-7f) ? 0.0f : multiplier;
            });
        }
    }
}