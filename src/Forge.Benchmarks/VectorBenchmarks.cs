using BenchmarkDotNet.Attributes;
using Forge.Core;
using Forge.Algorithms;

namespace Forge.Benchmarks;

[MemoryDiagnoser]
public class VectorBenchmarks
{
    private float[] _vecA = null!;
    private float[] _vecB = null!;
    private Tensor _tA = null!;
    private Tensor _tB = null!;
    
    [Params(1024, 4096)]
    public int Size;
    
    [GlobalSetup]
    public void Setup()
    {
        _vecA = new float[Size];
        _vecB = new float[Size];
        var rnd = new Random(42);
        for (int i = 0; i < Size; i++)
        {
            _vecA[i] = (float)rnd.NextDouble();
            _vecB[i] = (float)rnd.NextDouble();
        }

        _tA = new Tensor(1, Size, _vecA);
        _tB = new Tensor(1, Size, _vecB);
    }

    [Benchmark]
    public float DotProduct_SIMD() => VectorMath.DotProduct(_vecA, _vecB);
}