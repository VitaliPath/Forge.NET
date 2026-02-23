using Xunit;
using Forge.Core;
using System.Numerics;

namespace Forge.Tests;

public class VectorMathTests
{
    [Fact]
    public void DotProduct_IsHardwareAccelerated_OnCurrentPlatform()
    {
        // This is a diagnostic test. If it fails, your Mac/PC is running in "Naive" mode.
        Assert.True(Vector.IsHardwareAccelerated, "SIMD must be enabled for FORGE-016 compliance.");
    }

    [Fact]
    public void DotProduct_Maintains_Parity_With_Tail_Elements()
    {
        var a = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f };
        var b = new float[] { 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
        float result = VectorMath.DotProduct(a, b);
        Assert.Equal(84.0f, result, precision: 5);
    }

    [Fact]
    public void CosineSimilarity_Scale_Test_HighDimension()
    {
        int dim = 1024;
        var dataA = new float[dim];
        var dataB = new float[dim];
        for (int i = 0; i < dim; i++) { dataA[i] = 1.0f; dataB[i] = 1.0f; }
        double sim = VectorMath.CosineSimilarity(dataA, dataB);
        Assert.Equal(1.0, sim, precision: 5);
    }

    [Fact]
    public void L2Distance_Identity_Is_Zero()
    {
        var a = new float[] { 1.2f, 3.4f, 5.6f };
        Assert.Equal(0.0f, VectorMath.L2Distance(a, a), precision: 5);
    }

    [Fact]
    public void L2Distance_UnitCircle_Is_SqrtTwo()
    {
        var a = new float[] { 1.0f, 0.0f };
        var b = new float[] { 0.0f, 1.0f };
        float expected = (float)Math.Sqrt(2.0);
        Assert.Equal(expected, VectorMath.L2Distance(a, b), precision: 5);
    }

    [Fact]
    public void L2Distance_Maintains_Parity_With_Tail_Elements()
    {
        // Dimension 5 (odd, triggers SIMD + Tail path)
        var a = new float[] { 1, 2, 3, 4, 5 };
        var b = new float[] { 5, 4, 3, 2, 1 };

        // sum((1-5)^2 + (2-4)^2 + (3-3)^2 + (4-2)^2 + (5-1)^2)
        // sum(16 + 4 + 0 + 4 + 16) = 40
        double expected = Math.Sqrt(40.0);
        Assert.Equal(expected, VectorMath.L2Distance(a, b), precision: 5);
    }

    [Fact]
    public void CosineSimilarity_Handles_Zero_Vectors_Safely()
    {
        // Arrange
        var a = new float[] { 0, 0, 0 };
        var b = new float[] { 1, 2, 3 };

        // Act
        float result = VectorMath.CosineSimilarity(a, b);

        // Assert
        Assert.Equal(0.0f, result);
    }

    [Fact]
    public void CosineSimilarity_Matches_Scalar_Expectation()
    {
        // Simple 3-4-5 triangle style check
        var a = new float[] { 3.0f, 0.0f };
        var b = new float[] { 0.0f, 4.0f };

        // Orthogonal vectors should be 0
        Assert.Equal(0.0f, VectorMath.CosineSimilarity(a, b), precision: 5);

        var c = new float[] { 1.0f, 1.0f };
        var d = new float[] { 1.0f, 1.0f };
        // Identical vectors should be 1.0
        Assert.Equal(1.0f, VectorMath.CosineSimilarity(c, d), precision: 5);
    }
}