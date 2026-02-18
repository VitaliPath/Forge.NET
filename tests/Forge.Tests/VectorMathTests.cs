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
}