namespace Forge.Graph.Persistence;

/// <summary>
/// Unifies the allocation of pinned SoA buffers for CSR structures.
/// </summary>
internal static class CsrBufferFactory
{
    public static (int[] rowPtr, int[] colIdx, float[] weights, long[] lastModified, byte[] edgeTypes) 
        Allocate(int nodeCount, int edgeCount)
    {
        return (
            new int[nodeCount + 1],
            GC.AllocateArray<int>(edgeCount, pinned: true),
            GC.AllocateArray<float>(edgeCount, pinned: true),
            GC.AllocateArray<long>(edgeCount, pinned: true),
            GC.AllocateArray<byte>(edgeCount, pinned: true)
        );
    }
}