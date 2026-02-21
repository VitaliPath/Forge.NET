using System;
using System.Threading.Tasks;

namespace Forge.Core;

/// <summary>
/// FORGE-055: Centralized hardware-aligned concurrency policy.
/// Prevents thread over-subscription and cache-line contention.
/// </summary>
public static class ForgeConcurrency
{
    /// <summary>
    /// Returns ParallelOptions pinned to the physical core count.
    /// Use this for CPU-bound mathematical operations.
    /// </summary>
    public static ParallelOptions DefaultOptions => new()
    {
        MaxDegreeOfParallelism = Environment.ProcessorCount
    };
}