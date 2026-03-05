using Forge.Core;
using System.Runtime.InteropServices;
using System.Text;

namespace Forge.Graph.Persistence;

internal static class CsrIOHandler
{
    private const uint MagicBytes = 0x46524745; // "FRGE"
    private const int SchemaVersion = 1;

    public static void WriteToStream(GraphCsr csr, Stream stream)
    {
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        // 1. Header: [Magic(4)] [Version(4)] [Nodes(4)] [Edges(4)]
        writer.Write(MagicBytes);
        writer.Write(SchemaVersion);
        writer.Write(csr.NodeCount);
        writer.Write(csr.EdgeCount);

        // 2. Direct-to-SoA Buffer Dumps
        writer.Write(MemoryMarshal.AsBytes(csr.RowPtr.AsSpan()));
        writer.Write(MemoryMarshal.AsBytes(csr.ColIdx.AsSpan()));
        writer.Write(MemoryMarshal.AsBytes(csr.Weights.AsSpan()));
        writer.Write(MemoryMarshal.AsBytes(csr.LastModified.AsSpan()));

        // 3. Identity Pool (Metadata)
        foreach (var id in csr.IndexToId)
        {
            writer.Write(id);
        }
    }

    public static GraphCsr ReadFromStream(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        // 1. Validation
        if (reader.ReadUInt32() != MagicBytes) 
            throw new InvalidDataException("Invalid Magic Bytes: Not a Forge Snapshot.");
        
        if (reader.ReadInt32() != SchemaVersion)
            throw new InvalidDataException("Schema Mismatch: Snapshot version is incompatible.");

        int nodeCount = reader.ReadInt32();
        int edgeCount = reader.ReadInt32();

        // 2. Allocation
        int[] rowPtr = new int[nodeCount + 1];
        int[] colIdx = new int[edgeCount];
        float[] weights = new float[edgeCount];
        long[] lastModified = new long[edgeCount];

        // 3. Re-hydration
        reader.Read(MemoryMarshal.AsBytes(rowPtr.AsSpan()));
        reader.Read(MemoryMarshal.AsBytes(colIdx.AsSpan()));
        reader.Read(MemoryMarshal.AsBytes(weights.AsSpan()));
        reader.Read(MemoryMarshal.AsBytes(lastModified.AsSpan()));

        // 4. Identity Mapping
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
}