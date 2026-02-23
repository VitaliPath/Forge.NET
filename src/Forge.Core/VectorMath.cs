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

    /// <summary>
    /// FORGE-058: SIMD-vectorized single-pass Cosine Similarity.
    /// Calculates Dot Product and both Magnitudes in a single sweep to minimize memory traffic.
    /// </summary>
    public static float CosineSimilarity(float[] vecA, float[] vecB)
    {
        if (vecA.Length != vecB.Length)
            throw new ArgumentException("Vectors must have identical dimensions.");

        float dot = 0.0f;
        float magA = 0.0f;
        float magB = 0.0f;

        int i = 0;
        int width = Vector<float>.Count;

        if (Vector.IsHardwareAccelerated && vecA.Length >= width)
        {
            var vDot = Vector<float>.Zero;
            var vMagA = Vector<float>.Zero;
            var vMagB = Vector<float>.Zero;

            for (; i <= vecA.Length - width; i += width)
            {
                var va = new Vector<float>(vecA, i);
                var vb = new Vector<float>(vecB, i);

                vDot += va * vb;
                vMagA += va * va;
                vMagB += vb * vb;
            }

            dot = Vector.Dot(vDot, Vector<float>.One);
            magA = Vector.Dot(vMagA, Vector<float>.One);
            magB = Vector.Dot(vMagB, Vector<float>.One);
        }

        for (; i < vecA.Length; i++)
        {
            dot += vecA[i] * vecB[i];
            magA += vecA[i] * vecA[i];
            magB += vecB[i] * vecB[i];
        }

        float denominator = (float)(Math.Sqrt(magA) * Math.Sqrt(magB));
        
        return denominator < 1e-10f ? 0.0f : dot / denominator;
    }
}