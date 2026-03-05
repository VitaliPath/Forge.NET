using Forge.Core;
using System.Runtime.InteropServices;
using System.Text;

namespace Forge.Graph.Persistence;

internal static class CsrIOHandler
{
    private const uint MagicBytes = 0x46524745; // "FRGE"
    private const int SchemaVersion = 2; // FORGE-065: Incremented for EdgeTypes support

    public static void WriteToStream(GraphCsr csr, Stream stream)
    {
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        // 1. Header: Updated Version
        writer.Write(MagicBytes);
        writer.Write(SchemaVersion);
        writer.Write(csr.NodeCount);
        writer.Write(csr.EdgeCount);

        // 2. Direct-to-SoA Buffer Dumps (Including EdgeTypes)
        writer.Write(MemoryMarshal.AsBytes(csr.RowPtr.AsSpan()));
        writer.Write(MemoryMarshal.AsBytes(csr.ColIdx.AsSpan()));
        writer.Write(MemoryMarshal.AsBytes(csr.Weights.AsSpan()));
        writer.Write(MemoryMarshal.AsBytes(csr.LastModified.AsSpan()));
        writer.Write(csr.EdgeTypes.AsSpan()); // Raw byte dump for SoA types

        // 3. Identity Pool
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
        
        int version = reader.ReadInt32();
        if (version != SchemaVersion)
            throw new InvalidDataException($"Schema Mismatch: Expected v{SchemaVersion}, found v{version}.");

        int nodeCount = reader.ReadInt32();
        int edgeCount = reader.ReadInt32();

        // 2. Allocation
        int[] rowPtr = new int[nodeCount + 1];
        int[] colIdx = new int[edgeCount];
        float[] weights = new float[edgeCount];
        long[] lastModified = new long[edgeCount];
        byte[] edgeTypes = new byte[edgeCount]; // New buffer

        // 3. Re-hydration (Sequential bulk reads)
        reader.Read(MemoryMarshal.AsBytes(rowPtr.AsSpan()));
        reader.Read(MemoryMarshal.AsBytes(colIdx.AsSpan()));
        reader.Read(MemoryMarshal.AsBytes(weights.AsSpan()));
        reader.Read(MemoryMarshal.AsBytes(lastModified.AsSpan()));
        reader.Read(edgeTypes.AsSpan()); // Read the types directly

        // 4. Identity Mapping
        string[] indexToId = new string[nodeCount];
        var idToIndex = new Dictionary<string, int>(nodeCount, StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < nodeCount; i++)
        {
            string id = reader.ReadString();
            indexToId[i] = id;
            idToIndex.Add(id, i);
        }

        return new GraphCsr(rowPtr, colIdx, weights, lastModified, edgeTypes, idToIndex, indexToId);
    }
}