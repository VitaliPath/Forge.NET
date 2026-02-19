using System.Numerics;

namespace Forge.Core;

public static class VectorMath
{
    public static double DotProduct(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have identical dimensions.");

        double sum = 0;
        int i = 0;
        int width = Vector<double>.Count;

        if (Vector.IsHardwareAccelerated && a.Length >= width)
        {
            Vector<double> vSum = Vector<double>.Zero;
            for (; i <= a.Length - width; i += width)
            {
                var vA = new Vector<double>(a.Slice(i));
                var vB = new Vector<double>(b.Slice(i));
                vSum += vA * vB;
            }
            sum = Vector.Dot(vSum, Vector<double>.One);
        }

        for (; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    /// <summary>
    /// FORGE-020: Calculates Euclidean Distance (L2 Norm) using SIMD acceleration.
    /// Formula: sqrt(sum((a_i - b_i)^2))
    /// </summary>
    public static double L2Distance(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have identical dimensions.");

        double sumSq = 0;
        int i = 0;
        int width = Vector<double>.Count;

        if (Vector.IsHardwareAccelerated && a.Length >= width)
        {
            Vector<double> vSumSq = Vector<double>.Zero;
            for (; i <= a.Length - width; i += width)
            {
                var vA = new Vector<double>(a.Slice(i));
                var vB = new Vector<double>(b.Slice(i));
                var diff = vA - vB;
                vSumSq += diff * diff;
            }
            sumSq = Vector.Dot(vSumSq, Vector<double>.One);
        }

        for (; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sumSq += diff * diff;
        }

        return Math.Sqrt(sumSq);
    }

    /// <summary>
    /// FORGE-020: Performs in-place SIMD-accelerated normalization.
    /// Formula: v = v / ||v||
    /// </summary>
    public static void Normalize(Span<double> v)
    {
        double mag = Math.Sqrt(DotProduct(v, v));
        if (mag < 1e-9) return;

        double invMag = 1.0 / mag;
        int i = 0;
        int width = Vector<double>.Count;

        if (Vector.IsHardwareAccelerated && v.Length >= width)
        {
            var vInvMag = new Vector<double>(invMag);
            for (; i <= v.Length - width; i += width)
            {
                var vec = new Vector<double>(v.Slice(i));
                (vec * vInvMag).CopyTo(v.Slice(i));
            }
        }

        for (; i < v.Length; i++) v[i] *= invMag;
    }

    public static double CosineSimilarity(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        double dot = DotProduct(a, b);
        double magA = Math.Sqrt(DotProduct(a, a));
        double magB = Math.Sqrt(DotProduct(b, b));

        if (magA == 0 || magB == 0) return 0;
        return dot / (magA * magB);
    }
}