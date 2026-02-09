using Xunit;
using Forge.Core;

namespace Forge.Tests;

public class TensorTests
{
    [Fact]
    public void SanityCheck_MatMul_Backprop()
    {
        // 1. Setup
        // A is (1, 2) -> [[2, 3]]
        var a = new Tensor(1, 2, new double[] { 2.0, 3.0 });

        // B is (2, 1) -> [[4], [5]]
        var b = new Tensor(2, 1, new double[] { 4.0, 5.0 });

        // 2. Forward
        var c = a.MatMul(b);

        // Check Shape (1, 1)
        Assert.Equal(1, c.Shape[0]);
        Assert.Equal(1, c.Shape[1]);

        // Check Value (2*4 + 3*5 = 23)
        Assert.Equal(23.0, c.Data[0]);

        // 3. Backward
        c.Backward();

        // 4. Check Gradients
        // dL/dA = B^T = [4, 5]
        Assert.Equal(4.0, a.Grad[0]);
        Assert.Equal(5.0, a.Grad[1]);

        // dL/dB = A^T = [2, 3]
        Assert.Equal(2.0, b.Grad[0]); // Flat index 0
        Assert.Equal(3.0, b.Grad[1]); // Flat index 1
    }

    [Fact]
    public void ApplyDecay_HalfLife_ReducesByFiftyPercent()
    {
        // Arrange: Target 50% decay
        // lambda = ln(2) / half_life. For 138.6 days, lambda is ~0.005
        var tensor = new Tensor(1, 2, new double[] { 100.0, 50.0 });
        double lambda = 0.005;
        double halfLife = 138.629; // ln(2) / 0.005

        // Act
        tensor.ApplyDecay(lambda, halfLife);

        // Assert: Exactly 50% reduction within the 0.001% tolerance
        Assert.Equal(50.0, tensor.Data[0], precision: 2);
        Assert.Equal(25.0, tensor.Data[1], precision: 2);
    }

    [Fact]
    public void ApplyDecay_Clamps_NegativeTime()
    {
        // Arrange: Use negative time (simulating clock drift)
        var tensor = new Tensor(1, 1, new double[] { 10.0 });
        
        // Act: Decay with negative time should be clamped to 0 (multiplier = 1.0)
        tensor.ApplyDecay(0.5, -100.0);

        // Assert: Value remains unchanged
        Assert.Equal(10.0, tensor.Data[0]);
    }

    [Fact]
    public void ApplyDecay_Clamps_Subnormals()
    {
        // Arrange: Massive time delta that would result in a subnormal float
        var tensor = new Tensor(1, 1, new double[] { 1.0 });
        
        // Act: lambda=1, t=100 -> exp(-100) is ~3.7e-44 (well below 1e-9)
        tensor.ApplyDecay(1.0, 100.0);

        // Assert: Multiplier clamped to 0.0 to protect CPU performance
        Assert.Equal(0.0, tensor.Data[0]);
    }
}