using Xunit;
using Forge.Core;

namespace Forge.Tests;

public class CoreTests
{
    [Fact]
    public void SanityCheck_Backprop_Scalar_As_Tensor()
    {
        var a = new Tensor(1, 1, new float[] { 2.0f });
        var b = new Tensor(1, 1, new float[] { -3.0f });
        var c = new Tensor(1, 1, new float[] { 10.0f });
        
        var d = a.MatMul(b); 
        var e = d + c; 

        Assert.Equal(4.0, e.Data[0]);

        e.Backward();

        Assert.Equal(-3.0, a.Grad[0]);
        
        Assert.Equal(2.0, b.Grad[0]);
        
        Assert.Equal(1.0, c.Grad[0]);
    }
}