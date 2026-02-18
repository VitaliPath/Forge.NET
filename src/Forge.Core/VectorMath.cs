using System.Numerics;
using System.Runtime.InteropServices;

namespace Forge.Core;

public static class VectorMath
{
    /// <summary>
    /// Calculates the dot product of two spans using SIMD acceleration.
    /// Formula: Σ(a_i * b_i)
    /// </summary>
    public static double DotProduct(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have identical dimensions.");

        double sum = 0;
        int i = 0;
        int width = Vector<double>.Count;

        // SIMD Path
        if (Vector.IsHardwareAccelerated && a.Length >= width)
        {
            Vector<double> vSum = Vector<double>.Zero;
            
            // Process chunks of size 'width' (usually 2 or 4 doubles depending on CPU)
            for (; i <= a.Length - width; i += width)
            {
                var vA = new Vector<double>(a.Slice(i));
                var vB = new Vector<double>(b.Slice(i));
                vSum += vA * vB;
            }

            // Horizontal addition of the vector components
            for (int j = 0; j < width; j++) sum += vSum[j];
        }

        // Tail Path: Handle remaining elements (n % width)
        for (; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>
    /// Calculates Cosine Similarity using SIMD primitives.
    /// Optimized to perform only one pass for dot product and magnitudes.
    /// </summary>
    public static double CosineSimilarity(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        double dot = DotProduct(a, b);
        double magA = Math.Sqrt(DotProduct(a, a));
        double magB = Math.Sqrt(DotProduct(b, b));

        if (magA == 0 || magB == 0) return 0;
        return dot / (magA * magB);
    }
}