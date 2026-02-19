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
        // 7 is not divisible by 2 or 4 (standard double vector widths).
        // This forces the 'Tail Pass' logic to execute.
        var a = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
        var b = new double[] { 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
        
        // Expected: (1*7)+(2*6)+(3*5)+(4*4)+(5*3)+(6*2)+(7*1) = 84
        double result = VectorMath.DotProduct(a, b);
        
        Assert.Equal(84.0, result, precision: 10);
    }

    [Fact]
    public void CosineSimilarity_Scale_Test_HighDimension()
    {
        // Simulate a real bioinformatics vector (1024 dimensions)
        int dim = 1024;
        var dataA = new double[dim];
        var dataB = new double[dim];
        for (int i = 0; i < dim; i++) { dataA[i] = 1.0; dataB[i] = 1.0; }

        double sim = VectorMath.CosineSimilarity(dataA, dataB);

        // Identical vectors must have similarity of 1.0
        Assert.Equal(1.0, sim, precision: 10);
    }

    [Fact]
    public void L2Distance_Identity_Is_Zero()
    {
        var a = new double[] { 1.2, 3.4, 5.6 };
        Assert.Equal(0.0, VectorMath.L2Distance(a, a), precision: 10);
    }

    [Fact]
    public void L2Distance_UnitCircle_Is_SqrtTwo()
    {
        // Distance from (1,0) to (0,1)
        var a = new double[] { 1.0, 0.0 };
        var b = new double[] { 0.0, 1.0 };

        double expected = Math.Sqrt(2.0);
        Assert.Equal(expected, VectorMath.L2Distance(a, b), precision: 10);
    }

    [Fact]
    public void L2Distance_Maintains_Parity_With_Tail_Elements()
    {
        // Dimension 5 (odd, triggers SIMD + Tail path)
        var a = new double[] { 1, 2, 3, 4, 5 };
        var b = new double[] { 5, 4, 3, 2, 1 };

        // sum((1-5)^2 + (2-4)^2 + (3-3)^2 + (4-2)^2 + (5-1)^2)
        // sum(16 + 4 + 0 + 4 + 16) = 40
        double expected = Math.Sqrt(40.0);
        Assert.Equal(expected, VectorMath.L2Distance(a, b), precision: 10);
    }
}