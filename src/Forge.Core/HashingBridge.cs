using System;
using System.Security.Cryptography;

namespace Forge.Core;

/// <summary>
/// FORGE-062: Unified hashing bridge for Forge structures.
/// Standardizes on SHA-256 and Little-Endian byte-order for cross-platform stability.
/// </summary>
public static class HashingBridge
{
    /// <summary>
    /// Hashes three distinct buffers. Specifically optimized for GraphCsr (RowPtr, ColIdx, Weights).
    /// </summary>
    public static byte[] GenerateHash(ReadOnlySpan<byte> b1, ReadOnlySpan<byte> b2, ReadOnlySpan<byte> b3)
    {
        using var incrementalHash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        
        incrementalHash.AppendData(b1);
        incrementalHash.AppendData(b2);
        incrementalHash.AppendData(b3);
        
        return incrementalHash.GetHashAndReset();
    }

    /// <summary>
    /// General purpose hash for a single buffer.
    /// </summary>
    public static byte[] GenerateHash(ReadOnlySpan<byte> buffer)
    {
        using var incrementalHash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        incrementalHash.AppendData(buffer);
        return incrementalHash.GetHashAndReset();
    }

    /// <summary>
    /// Helper to convert a byte array to a hex string for Knowledge Base storage.
    /// </summary>
    public static string ToHexString(byte[] bytes) => Convert.ToHexString(bytes);
}