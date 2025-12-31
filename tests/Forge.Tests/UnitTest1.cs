using Forge.Core;

namespace Forge.Tests;

public class CoreTests
{
    [Fact]
    public void SanityCheck_Backprop()
    {
        var a = new Value(2.0, "a");
        var b = new Value(-3.0, "b");
        var c = new Value(10.0, "c");
        var e = (a * b) + c; // -6 + 10 = 4

        e.Backward();

        Assert.Equal(4.0, e.Data);
        Assert.Equal(-3.0, a.Grad);
        Assert.Equal(2.0, b.Grad);
    }
}