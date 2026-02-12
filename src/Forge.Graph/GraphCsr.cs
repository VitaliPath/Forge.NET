namespace Forge.Graph
{
    /// <summary>
    /// A read-optimized, immutable snapshot of a Graph in Compressed Sparse Row format.
    /// </summary>
    public struct GraphCsr
    {
        public readonly int[] RowPtr;      // Length: |V| + 1
        public readonly int[] ColIdx;      // Length: |E|
        public readonly double[] Weights;  // Length: |E|
        public readonly long[] LastModified; // Length: |E|

        // Bi-directional mapping for ID lookup
        public readonly Dictionary<string, int> IdToIndex;
        public readonly string[] IndexToId;

        public int NodeCount => RowPtr.Length - 1;
        public int EdgeCount => ColIdx.Length;

        public GraphCsr(
            int[] rowPtr, 
            int[] colIdx, 
            double[] weights, 
            long[] lastModified, 
            Dictionary<string, int> idToIndex, 
            string[] indexToId)
        {
            RowPtr = rowPtr;
            ColIdx = colIdx;
            Weights = weights;
            LastModified = lastModified;
            IdToIndex = idToIndex;
            IndexToId = indexToId;
        }
    }
}