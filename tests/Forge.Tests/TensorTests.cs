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
}