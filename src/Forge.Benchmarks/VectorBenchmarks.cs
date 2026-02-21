using BenchmarkDotNet.Attributes;
using Forge.Core;
using Forge.Algorithms;

namespace Forge.Benchmarks;

[MemoryDiagnoser]
public class VectorBenchmarks
{
    private double[] _vecA = null!;
    private double[] _vecB = null!;
    private Tensor _tA = null!;
    private Tensor _tB = null!;

    [Params(1024, 4096)] 
    public int Size;

    [GlobalSetup]
    public void Setup()
    {
        _vecA = new double[Size];
        _vecB = new double[Size];
        var rnd = new Random(42);
        for (int i = 0; i < Size; i++)
        {
            _vecA[i] = rnd.NextDouble();
            _vecB[i] = rnd.NextDouble();
        }

        _tA = new Tensor(1, Size, _vecA);
        _tB = new Tensor(1, Size, _vecB);
    }

    [Benchmark]
    public double DotProduct_SIMD() => VectorMath.DotProduct(_vecA, _vecB);

    [Benchmark]
    public double CosineSimilarity_SIMD() => BagOfWordsEncoder.CosineSimilarity(_tA, _tB);
}