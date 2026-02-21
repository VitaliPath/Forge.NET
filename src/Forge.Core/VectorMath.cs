using System.Numerics;
using System;

namespace Forge.Core;

public static class VectorMath
{
    public static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have identical dimensions.");

        float sum = 0;
        int i = 0;
        int width = Vector<float>.Count;

        if (Vector.IsHardwareAccelerated && a.Length >= width)
        {
            Vector<float> vSum = Vector<float>.Zero;
            for (; i <= a.Length - width; i += width)
            {
                var vA = new Vector<float>(a.Slice(i));
                var vB = new Vector<float>(b.Slice(i));
                vSum += vA * vB;
            }
            // Horizontal sum of the vector
            sum = Vector.Dot(vSum, Vector<float>.One);
        }

        for (; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    public static float L2Distance(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have identical dimensions.");

        float sumSq = 0;
        int i = 0;
        int width = Vector<float>.Count;

        if (Vector.IsHardwareAccelerated && a.Length >= width)
        {
            Vector<float> vSumSq = Vector<float>.Zero;
            for (; i <= a.Length - width; i += width)
            {
                var vA = new Vector<float>(a.Slice(i));
                var vB = new Vector<float>(b.Slice(i));
                var diff = vA - vB;
                vSumSq += diff * diff;
            }
            sumSq = Vector.Dot(vSumSq, Vector<float>.One);
        }

        for (; i < a.Length; i++)
        {
            float diff = a[i] - b[i];
            sumSq += diff * diff;
        }

        return (float)Math.Sqrt(sumSq);
    }

    public static void Normalize(Span<float> v)
    {
        float mag = (float)Math.Sqrt(DotProduct(v, v));
        if (mag < 1e-7f) return;

        float invMag = 1.0f / mag;
        int i = 0;
        int width = Vector<float>.Count;

        if (Vector.IsHardwareAccelerated && v.Length >= width)
        {
            var vInvMag = new Vector<float>(invMag);
            for (; i <= v.Length - width; i += width)
            {
                var vec = new Vector<float>(v.Slice(i));
                (vec * vInvMag).CopyTo(v.Slice(i));
            }
        }

        for (; i < v.Length; i++) v[i] *= invMag;
    }

    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float dot = DotProduct(a, b);
        float magA = (float)Math.Sqrt(DotProduct(a, a));
        float magB = (float)Math.Sqrt(DotProduct(b, b));

        if (magA == 0 || magB == 0) return 0;
        return dot / (magA * magB);
    }
}