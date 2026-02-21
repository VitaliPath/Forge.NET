using Forge.Core;

namespace Forge.Graph
{
    public struct GraphCsr
    {
        public readonly int[] RowPtr;
        public readonly int[] ColIdx;
        public readonly double[] Weights;
        public readonly long[] LastModified;

        public readonly Dictionary<string, int> IdToIndex;
        public readonly string[] IndexToId;

        public int NodeCount => RowPtr.Length - 1;
        public int EdgeCount => ColIdx.Length;

        /// <summary>
        /// FORGE-021: Storage Aliasing. Projects the raw weight buffer as a Tensor.
        /// Changes to this Tensor's Data will directly mutate the Graph weights.
        /// </summary>
        public Tensor WeightsAsTensor => new Tensor(1, EdgeCount, Weights);

        public GraphCsr(int[] rowPtr, int[] colIdx, double[] weights, long[] lastModified,
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
        /// FORGE-021: High-throughput parallel decay for the CSR snapshot.
        /// </summary>
        public void ApplyDecay(double lambda, long nowUnix)
        {
            const double secondsPerDay = 86400.0;

            var weights = this.Weights;
            var lastModified = this.LastModified;
            int count = this.EdgeCount;

            Parallel.For(0, count, i =>
            {
                double ageInDays = Math.Max(0, (nowUnix - lastModified[i]) / secondsPerDay);
                double multiplier = Math.Exp(-lambda * ageInDays);

                weights[i] *= (multiplier < 1e-9) ? 0.0 : multiplier;
            });
        }
    }
}